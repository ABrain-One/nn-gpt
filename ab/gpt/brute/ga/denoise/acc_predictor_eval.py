"""Held-out validation of the ABrain accuracy predictor, against the one this project has been using.

WHY THIS IS WORTH RUNNING. The in-house surrogate (ga/denoise/predictor/, a gradient-boosted model
over zero-cost proxies) is not merely unvalidated -- it is measurably ANTI-correlated with truth.
predictor_tracking_history.csv records five timepoints as the sample grew 257 -> 326 models:

    spearman(pred, actual) = -0.155, -0.152, -0.190, -0.188, -0.178      MAE 1.57 -> 1.80 dB

against the +0.693 it reported from k-fold CV on its own 197-model training set. Its dead-model gate
never fires either (mean p_dead 0.004 on models that diverged vs 0.001 on viable ones). Four GA
launchers pass --predictor --predict-keep 0.5, which trains the top half of every generation by that
ranking -- so wherever it was enabled it was preferentially discarding the BETTER half.

THE TWO PREDICTORS SOLVE DIFFERENT PROBLEMS, AND THAT IS THE POINT.
    ours      source -> predicted accuracy            zero-cost, ranks BEFORE any training
    ABrain    source + epoch1 + epoch2 -> final       needs 2 cheap epochs, predicts the outcome
The second is a far more tractable problem: two real measurements anchor the extrapolation instead
of asking a proxy to guess from static graph features. It also changes the pipeline shape from
"skip half untrained" to "train 2 epochs, then continue only the promising ones" -- which is
falsifiable per candidate, because every skipped model has real measurements attached.

THE VALIDATION SET IS ALREADY ON DISK AND IS GENUINELY HELD OUT. 278 ft50 rows carry a full
per-epoch curve at 50 epochs -- matching the predictor's total_epochs=50 -- so epoch 1 and 2 are
inputs and the measured best is the label. None of it was produced for this test.

SCALE. This framework stores accuracy = dB / 48 (ab/nn/metric/psnr.py), and the predictor is trained
on that convention, so dB values are divided by 48 on the way in and multiplied back on the way out.
Getting this backwards would produce a confident, meaningless number.

  python acc_predictor_eval.py --build            # CPU: writes the eval set
  python acc_predictor_eval.py --run              # GPU pod: needs unsloth + the HF repo
  python acc_predictor_eval.py --report
"""
import argparse
import glob
import json
import math
import os
import sys

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
OUTD = os.path.join(SWEEP, "acc_predictor")
EVALSET = os.path.join(OUTD, "evalset.json")
PREDS = os.path.join(OUTD, "predictions.jsonl")
SCALE = 48.0
OURS_SPEARMAN = -0.178          # predictor_tracking.json, n=326, 2026-07-17


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return None if va == 0 or vb == 0 else \
        sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def cmd_build(a):
    codes = {}
    p = os.path.join(SWEEP, "tflite_pool", "codes.jsonl")
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            codes[r["name"]] = r["code"]

    rows, skipped = [], 0
    for f in glob.glob(os.path.join(SWEEP, "fulltrain50", "result_shard*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            name = r.get("key") or r.get("model")
            curve = r.get("curve")
            best = r.get("best_dB") or r.get("max_dB")
            if not (name and curve and best) or name not in codes:
                skipped += 1
                continue
            d = {e: v for e, v in curve}
            if not {1, 2, 3} <= set(d):
                skipped += 1
                continue
            rows.append({"model": name,
                         "task": "img-denoising", "dataset": "denoise", "metric": "psnr",
                         "total_epochs": 50,
                         "dB_ep1": d[1], "dB_ep2": d[2], "dB_ep3": d[3],
                         "actual_best_acc": round(best / SCALE, 6),
                         "actual_best_dB": best,
                         "actual_best_epoch": r.get("best_epoch"),
                         "n_epochs": len(curve)})
    # dedupe by model, keeping the best-scoring run
    byname = {}
    for r in rows:
        if r["model"] not in byname or r["actual_best_dB"] > byname[r["model"]]["actual_best_dB"]:
            byname[r["model"]] = r
    rows = sorted(byname.values(), key=lambda r: r["model"])
    os.makedirs(OUTD, exist_ok=True)
    json.dump({"n": len(rows), "scale": SCALE, "rows": rows}, open(EVALSET, "w"), indent=1)
    dbs = sorted(r["actual_best_dB"] for r in rows)
    print(f"{len(rows)} eval rows ({skipped} skipped for missing curve/code)")
    print(f"  actual best dB: min {dbs[0]:.2f} median {dbs[len(dbs)//2]:.2f} max {dbs[-1]:.2f}")
    print(f"  spread {dbs[-1]-dbs[0]:.2f} dB -- the range a useful predictor must resolve")
    print(f"wrote {EVALSET}")


SCALES = {                       # dB -> whatever convention the predictor was trained on
    "acc100": lambda db: round(db / 48.0 * 100.0, 6),   # LEMUR accuracy as a percentage
    "acc01":  lambda db: round(db / 48.0, 6),           # LEMUR accuracy as stored
    "db":     lambda db: round(db, 6),                  # raw decibels
}


def cmd_run(a):
    sys.path.insert(0, "/shared/ssd/home/b-b-ahmed/thesis/nn-gpt")
    from ab.gpt.util.acc_client import predict_best_accuracy
    conv = SCALES[a.scale]

    ev = json.load(open(EVALSET))
    codes = {}
    for line in open(os.path.join(SWEEP, "tflite_pool", "codes.jsonl")):
        if line.strip():
            r = json.loads(line)
            codes[r["name"]] = r["code"]

    done = set()
    if os.path.exists(PREDS):
        for line in open(PREDS):
            if line.strip():
                done.add(json.loads(line)["model"])
    todo = [r for r in ev["rows"] if r["model"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(ev['rows'])} eval rows | {len(done)} already predicted | running {len(todo)}",
          flush=True)

    for i, r in enumerate(todo, 1):
        rec = {"model": r["model"], "actual_best_acc": r["actual_best_acc"],
               "actual_best_dB": r["actual_best_dB"],
               "actual_best_epoch": r["actual_best_epoch"]}
        try:
            acc, ep = predict_best_accuracy(
                task=r["task"], dataset=r["dataset"], metric=r["metric"],
                nn_code=codes[r["model"]],
                epoch_1_accuracy=conv(r["dB_ep1"]),
                epoch_2_accuracy=conv(r["dB_ep2"]),
                epoch_3_accuracy=conv(r["dB_ep3"]))
            # invert the scale so pred_dB is always comparable to actual_best_dB
            back = {"acc100": lambda v: v / 100.0 * 48.0,
                    "acc01": lambda v: v * 48.0,
                    "db": lambda v: v}[a.scale]
            rec.update(pred_raw=acc, scale=a.scale, pred_dB=round(back(acc), 3),
                       pred_epoch=ep, status="OK")
        except Exception as e:
            rec.update(status="FAIL", error=f"{type(e).__name__}: {str(e)[:160]}")
        with open(PREDS, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
    print(f"wrote {PREDS}")


def cmd_report(a):
    if not os.path.exists(PREDS):
        sys.exit("no predictions yet -- run --run in a GPU pod first")
    rows = [json.loads(l) for l in open(PREDS) if l.strip()]
    ok = [r for r in rows if r.get("status") == "OK" and r.get("pred_dB") is not None]
    print(f"{len(rows)} predictions | {len(ok)} usable")
    if len(ok) < 10:
        for r in rows[:5]:
            if r.get("status") != "OK":
                print("  ", r.get("error"))
        sys.exit("too few usable predictions")

    p = [r["pred_dB"] for r in ok]
    y = [r["actual_best_dB"] for r in ok]
    rs, rp = spearman(p, y), pearson(p, y)
    mae = sum(abs(a - b) for a, b in zip(p, y)) / len(ok)
    bias = sum(a - b for a, b in zip(p, y)) / len(ok)
    print("\n=== ABrain/Accuracy-Prediction, held out on ft50 ===")
    print(f"  Spearman(pred, actual) : {rs:+.3f}")
    print(f"  Pearson                : {rp:+.3f}")
    print(f"  MAE                    : {mae:.3f} dB")
    print(f"  bias (pred - actual)   : {bias:+.3f} dB")
    print(f"\n=== against the in-house zero-cost surrogate ===")
    print(f"  ours    Spearman {OURS_SPEARMAN:+.3f}  (n=326, predictor_tracking.json)")
    print(f"  ABrain  Spearman {rs:+.3f}  (n={len(ok)}, this run)")
    verdict = ("USABLE" if rs >= 0.5 else "WEAK" if rs >= 0.2 else "NOT USABLE")
    print(f"  verdict: {verdict}")

    # The decision the funnel actually makes: keep the top half. Measure THAT, not just rho.
    n = len(ok)
    k = n // 2
    top_pred = {r["model"] for r in sorted(ok, key=lambda r: -r["pred_dB"])[:k]}
    top_true = {r["model"] for r in sorted(ok, key=lambda r: -r["actual_best_dB"])[:k]}
    overlap = len(top_pred & top_true) / max(1, k)
    best_true = max(ok, key=lambda r: r["actual_best_dB"])
    kept_best = best_true["model"] in top_pred
    print(f"\n=== the decision the funnel makes: keep top 50% ===")
    print(f"  overlap with the true top 50% : {overlap:.0%}  (50% = coin flip)")
    print(f"  best model retained?          : {'YES' if kept_best else 'NO'} "
          f"({best_true['actual_best_dB']:.2f} dB)")
    kept = [r["actual_best_dB"] for r in ok if r["model"] in top_pred]
    print(f"  mean actual dB of kept half   : {sum(kept)/len(kept):.3f}")
    print(f"  mean actual dB of all         : {sum(y)/len(y):.3f}")
    print(f"  -> selection gain             : {sum(kept)/len(kept) - sum(y)/len(y):+.3f} dB")

    ep = [(r["pred_epoch"], r["actual_best_epoch"]) for r in ok
          if r.get("pred_epoch") and r.get("actual_best_epoch")]
    if len(ep) >= 10:
        print(f"\n  best_epoch: Spearman {spearman([e[0] for e in ep], [e[1] for e in ep]):+.3f}, "
              f"MAE {sum(abs(a-b) for a,b in ep)/len(ep):.1f} epochs (n={len(ep)})")

    json.dump({"n": len(ok), "spearman": rs, "pearson": rp, "mae_dB": mae, "bias_dB": bias,
               "top50_overlap": overlap, "kept_best": kept_best,
               "selection_gain_dB": sum(kept)/len(kept) - sum(y)/len(y),
               "ours_spearman": OURS_SPEARMAN, "verdict": verdict},
              open(os.path.join(OUTD, "report.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(OUTD, 'report.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale", choices=list(SCALES), default="acc100",
                    help="the predictor's prompt says best_accuracy is in [0,100] and its example "
                         "uses CIFAR percentages, but LEMUR stores accuracy in [0,1]; which "
                         "convention it was TRAINED on is unknown, so it is measured, not assumed")
    ap.add_argument("--preds", default=None, help="override the predictions file (per-scale pilots)")
    a = ap.parse_args()
    global PREDS
    if a.preds:
        PREDS = a.preds
    if a.build:
        cmd_build(a)
    elif a.run:
        cmd_run(a)
    elif a.report:
        cmd_report(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
