"""Pick the model set for the cross-dataset CONFIRMATION run.

WHY THIS SET AND NOT "THE TOP 8". Selecting the 8 best models by real-noise dB and then
re-measuring them would answer the wrong question. The claim under test is about SELECTION:
"if you choose an architecture using synthetic AWGN, what does it cost you on real noise?"
Answering that needs BOTH sides of the comparison in the set --

  * the models AWGN-based selection would hand you   (top-5 by the denoise column), and
  * the models real-noise selection would hand you   (top-5 by MIDD, top-5 by SIDD).

Their union is the set. It happens to be 8 models because the three top-5 lists overlap
heavily -- which is itself a hint that AWGN is not useless, just unreliable at rank 1.

WHY REPEATS ARE MEANINGFUL HERE. Nothing in the training path seeds anything: there is no
torch.manual_seed / np.random.seed in Train.py, api.py or Util.py, and Util.py:179 builds the
train loader with shuffle=True and no generator, so weight init and batch order differ every
run. The DATA is fixed though -- denoise_awgn.py:98 seeds the added noise per image index
(manual_seed(1000003 + idx)) -- so repeats vary the TRAINING and hold the TEST SET constant,
which is exactly the variance this run needs to measure.

WHAT THE 1.90 dB FIGURE ACTUALLY RESTS ON. In the single-draw run the AWGN #1 pick
(LLM_Pg33_seeded-8a7b5df) scored 34.56 on MIDD -- the worst MIDD of all 8 selected models --
while AWGN's #2 and #3 picks scored 36.14 and 36.15, i.e. MIDD ranks 3 and 2. So the entire
regret is carried by ONE model at rank 1. That is the specific thing this run exists to test:
if 8a7b5df regresses toward the mean on re-measurement, regret@1 collapses and the honest
headline becomes regret@3 = 0.31 dB.

  python confirm_select.py
"""
import argparse
import glob
import json
import os

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(SWEEP, "crossdata"))
    ap.add_argument("--out", default=os.path.join(SWEEP, "confirm", "subset.json"))
    ap.add_argument("--min-dB", type=float, default=32.0)
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()

    res = {}
    for f in glob.glob(os.path.join(a.src, "result_shard*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "OK" and r.get("best_dB") is not None:
                res.setdefault(r["model"], {})[r["dataset"]] = r["best_dB"]

    full = {m: v for m, v in res.items() if {"denoise", "midd", "sidd"} <= set(v)}
    work = {m: v for m, v in full.items() if v["denoise"] >= a.min_dB}
    print(f"{len(full)} models with all three datasets | {len(work)} at >= {a.min_dB} dB")

    why = {}
    for col, lbl in (("denoise", "AWGN"), ("midd", "MIDD"), ("sidd", "SIDD")):
        for m in sorted(work, key=lambda m: -work[m][col])[:a.topk]:
            why.setdefault(m, []).append(lbl)

    models = []
    for m in sorted(why, key=lambda m: -work[m]["midd"]):
        v = work[m]
        models.append({"model": m, "awgn_dB": v["denoise"], "midd_dB": v["midd"],
                       "sidd_dB": v["sidd"], "why": sorted(why[m])})
        print(f"  {'+'.join(sorted(why[m])):<16} AWGN {v['denoise']:6.2f}  "
              f"MIDD {v['midd']:6.2f}  SIDD {v['sidd']:6.2f}  {m[:46]}")

    # The k=1 picks are recorded explicitly so the report never has to re-derive "who would
    # AWGN have chosen" from a set that has since been re-measured.
    baseline = {c: max(work, key=lambda m: work[m][c]) for c in ("denoise", "midd", "sidd")}
    out = {"n": len(models), "models": models, "single_draw_baseline": baseline,
           "single_draw_dB": {m: work[m] for m in why}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n{len(models)} models -> {len(models)}x3 datasets x3 repeats = {len(models)*9} runs")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
