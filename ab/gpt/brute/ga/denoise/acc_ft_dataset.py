"""Build the fine-tuning dataset for a DENOISING-specialised accuracy predictor.

WHY A SPECIALISED PREDICTOR. The stock ABrain predictor, validated here on 280 held-out ft50 models,
scores rho=+0.226 against the final outcome where the trivial baseline "just use epoch 3" scores
+0.698, and its partial correlation given epoch 3 is +0.056 -- essentially no marginal information.
It is still far better than this project's own zero-cost surrogate (rho=-0.178), and its absolute
calibration is genuinely good (MAE 1.62 dB vs the baseline's 2.98 dB). The most likely explanation
is domain transfer: it was trained on a LEMUR corpus dominated by image CLASSIFICATION, and we ask
it about DENOISING scored as PSNR/48. That is a fixable problem, and fixing it is the point here.

THE TRAP IN THE OBVIOUS DATASET, AND THE FIX.
The DB holds 793 clean single-run denoising curves, 727 after excluding the held-out eval models.
But their LENGTHS are bimodal and lopsided:

    3-5 epochs    701 runs     median gain epoch3 -> best  +0.30 dB
    26-50 epochs   26 runs     median gain epoch3 -> best  +2.01 dB

acc_client.py hardcodes `max_epochs: 50` into every prompt. Training on 701 five-epoch runs whose
"best" is 0.30 dB above epoch 3, while telling the model the budget is 50 epochs, teaches it to
under-predict a real 50-epoch outcome by roughly 1.7 dB -- a systematic bias baked in by the data,
not by the model. Two things prevent that:

  1. EVERY EXAMPLE CARRIES ITS TRUE max_epochs. The prompt template already has the field; only the
     client pins it to 50. Short runs then teach short-horizon extrapolation and long runs teach
     long-horizon, instead of the two contradicting each other.
  2. ft50 IS SPLIT, NOT SPENT ENTIRELY ON THE TEST SET. It is the only large source of genuine
     50-epoch denoising curves, and a predictor with 26 long examples would have no signal where it
     matters most. A fixed, model-disjoint holdout is kept for the honest comparison; the rest is
     training data.

THE HOLDOUT IS BY MODEL AND IS FROZEN. Splitting by run would leak: the same architecture measured
twice would land on both sides. The holdout list is written to disk so every future evaluation uses
the identical set, and the epoch-3 baseline is recomputed on that same subset so the comparison
stays like-for-like.

  python acc_ft_dataset.py --build
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sqlite3

# PATH MUST RESOLVE IN BOTH WORLDS. The same tree is /shared/ssd/home/b-b-ahmed/thesis on the login
# node and /a/mm inside every pod (hostPath mount). This file is imported by acc_ft_train.py, which
# runs IN A POD -- so a hardcoded host path made load_codes() raise FileNotFoundError there. It cost a
# full 95-minute H100 fine-tune: training completed and saved the adapter, then the evaluation stage
# died on this line. Detect the mount instead of assuming either side.
THESIS = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
SWEEP = os.path.join(THESIS, "sweep_out")
DB = os.path.join(THESIS, "nn-dataset", "db", "ab.nn.db")
OUTD = os.path.join(SWEEP, "acc_ft")
SCALE = 48.0
N_HOLDOUT = 100

SYSTEM_PROMPT = """You are a strict JSON generator.
You must output exactly ONE JSON object and nothing else.

The JSON must contain exactly the keys:
best_accuracy
best_epoch

Rules:
best_accuracy must be a float in [0,100] rounded to 2 decimals.
best_epoch must be a positive integer representing the absolute epoch where peak validation accuracy occurs.

Do not explain.
Do not add text.
Stop immediately after the closing brace }.
"""


def user_message(task, dataset, metric, nn_code, e1, e2, e3, max_epochs):
    """Byte-identical to acc_client._build_user_message except max_epochs is a real argument.

    Matching the serving prompt exactly matters: a fine-tune trained on a different template is not
    a drop-in for the client, and the mismatch would show up as unexplained degradation at serving
    time rather than as an obvious error.
    """
    return "\n".join([
        "INPUT",
        f"task: {task}",
        f"dataset: {dataset}",
        f"metric: {metric}",
        "",
        "TRAINING_BUDGET",
        f"max_epochs: {max_epochs}",
        "",
        "EARLY_TRAINING_SIGNAL",
        f"epoch_1_accuracy: {round(float(e1), 6)}",
        f"epoch_2_accuracy: {round(float(e2), 6)}",
        f"epoch_3_accuracy: {round(float(e3), 6)}",
        "",
        "NEURAL_NETWORK_CODE",
        "```python",
        (nn_code or "").strip(),
        "```",
        "",
        "Analyze the training dynamics, architecture complexity, and optimization hyperparameters to estimate the final training outcome.",
        "",
        "Important signals to consider:",
        "- Early learning progress (epoch accuracies)",
        "- Saturation of improvement across epochs",
        "- Architecture depth and complexity",
        "- Optimization scale (learning rate, batch size, effective_lr)",
        "",
        "Using these signals, estimate the final best validation accuracy and the epoch where it occurs.",
        "",
        "The value best_epoch represents the training epoch where the highest validation accuracy is reached.",
        "",
        "Constraints:",
        "- best_epoch must be an integer between 1 and max_epochs.",
        "",
        "OUTPUT (JSON ONLY):",
    ])


def load_codes():
    codes = {}
    for line in open(os.path.join(SWEEP, "tflite_pool", "codes.jsonl")):
        if line.strip():
            r = json.loads(line)
            codes[r["name"]] = r["code"]
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    for name, code in con.execute("select name, code from nn"):
        codes.setdefault(name, code)
    return codes


def from_db():
    """Clean single-run curves. A model with two runs at the same epoch is ambiguous -- the rows
    carry no run id -- so those are dropped rather than interleaved into a fictional curve."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for nn_, ep, acc in con.execute(
            "select nn, epoch, accuracy from stat "
            "where task='img-denoising' and dataset='denoise' and metric='psnr'"):
        per[nn_][ep].append(acc)
    out = []
    for nn_, d in per.items():
        if not {1, 2, 3} <= set(d) or max(len(v) for v in d.values()) > 1:
            continue
        c = {e: v[0] for e, v in d.items()}
        mx = max(c)
        best_ep = max(c, key=lambda e: c[e])
        out.append({"model": nn_, "task": "img-denoising", "dataset": "denoise", "metric": "psnr",
                    "max_epochs": mx, "e1": c[1], "e2": c[2], "e3": c[3],
                    "best_acc": c[best_ep], "best_epoch": best_ep, "src": "db"})
    return out


def from_jsonl():
    """Runs whose per-epoch curve was recorded explicitly, so no run-splitting is needed."""
    out = []
    for pat, ds, metric, src in (
            ("fulltrain50/result_shard*.jsonl", "denoise", "psnr", "ft50"),
            ("beatbar/result_shard*.jsonl", "denoise", "psnr", "beatbar"),
            ("crossdata/result_shard*.jsonl", None, "psnrrgb", "crossdata"),
            ("confirm/result_shard*.jsonl", None, "psnrrgb", "confirm")):
        for f in glob.glob(os.path.join(SWEEP, pat)):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                curve, name = r.get("curve"), (r.get("model") or r.get("key"))
                if not curve or not name:
                    continue
                c = {e: v / SCALE for e, v in curve}
                if not {1, 2, 3} <= set(c):
                    continue
                best_ep = max(c, key=lambda e: c[e])
                out.append({"model": name, "task": "img-denoising",
                            "dataset": ds or r.get("dataset", "denoise"), "metric": metric,
                            "max_epochs": max(c), "e1": c[1], "e2": c[2], "e3": c[3],
                            "best_acc": c[best_ep], "best_epoch": best_ep, "src": src})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default=OUTD)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    codes = load_codes()

    rows = [r for r in (from_db() + from_jsonl()) if r["model"] in codes]
    # one example per (model, source, budget): the same architecture measured under two different
    # protocols is two legitimate training signals, but the same run must not appear twice
    seen, uniq = set(), []
    for r in rows:
        k = (r["model"], r["src"], r["max_epochs"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    rows = uniq
    print(f"{len(rows)} candidate examples from {len(set(r['model'] for r in rows))} models")
    print("  by source:", dict(collections.Counter(r["src"] for r in rows)))
    b = collections.Counter()
    for r in rows:
        k = r["max_epochs"]
        b["3-5" if k <= 5 else "6-25" if k <= 25 else "26-50"] += 1
    print("  by budget:", dict(b))

    # Frozen, model-disjoint holdout drawn from the 50-epoch ft50 runs -- the only place where a
    # long-horizon prediction can actually be scored. Chosen by a hash of the name so the split is
    # deterministic and reproducible without storing a seed.
    ft50 = sorted({r["model"] for r in rows if r["src"] == "ft50" and r["max_epochs"] >= 40})
    ft50.sort(key=lambda n: hashlib.sha1(n.encode()).hexdigest())
    holdout = set(ft50[:N_HOLDOUT])
    train = [r for r in rows if r["model"] not in holdout]
    test = [r for r in rows if r["model"] in holdout and r["src"] == "ft50"]

    def to_chatml(r):
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(
                r["task"], r["dataset"], r["metric"], codes[r["model"]],
                r["e1"] * 100, r["e2"] * 100, r["e3"] * 100, r["max_epochs"])},
            {"role": "assistant", "content": json.dumps(
                {"best_accuracy": round(r["best_acc"] * 100, 2),
                 "best_epoch": int(r["best_epoch"])})}]}

    # val split from TRAIN only, again model-disjoint
    tr_models = sorted({r["model"] for r in train})
    tr_models.sort(key=lambda n: hashlib.sha1(("v" + n).encode()).hexdigest())
    val_models = set(tr_models[:max(20, len(tr_models) // 10)])
    tr = [r for r in train if r["model"] not in val_models]
    va = [r for r in train if r["model"] in val_models]

    for name, part in (("train", tr), ("val", va)):
        p = os.path.join(a.out, f"{name}.jsonl")
        with open(p, "w") as fh:
            for r in part:
                fh.write(json.dumps(to_chatml(r)) + "\n")
        print(f"  {name}: {len(part)} examples, {len(set(x['model'] for x in part))} models -> {p}")
    json.dump({"holdout_models": sorted(holdout), "n_test_rows": len(test),
               "scale_note": "accuracy is dB/48, emitted x100 to match the [0,100] prompt contract"},
              open(os.path.join(a.out, "holdout.json"), "w"), indent=1)
    json.dump(test, open(os.path.join(a.out, "test_rows.json"), "w"), indent=1)
    print(f"  test: {len(test)} rows over {len(holdout)} FROZEN holdout models "
          f"-> {os.path.join(a.out, 'holdout.json')}")
    bt = collections.Counter("3-5" if r["max_epochs"] <= 5 else "6-25" if r["max_epochs"] <= 25
                             else "26-50" for r in tr)
    print(f"\n  train budget mix: {dict(bt)}")
    print("  (each example carries its OWN max_epochs, so short runs cannot teach the model to")
    print("   under-predict a 50-epoch outcome -- the failure the raw DB pool would have caused)")


if __name__ == "__main__":
    main()
