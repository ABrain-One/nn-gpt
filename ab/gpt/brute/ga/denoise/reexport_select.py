"""Select and stage models for the published-protocol re-export.

GOAL. Put this corpus on the axes the two reference papers use, so its numbers stop being
internally consistent but externally unanchored:

  MIDD / MAI 2021 protocol   runtime @ 720x480     score with C = 2^64
  MAI 2026 protocol          runtime @ 1088x1920   score with C ~= 7.56e18
  PSNR for both              RGB, full 3200x2432 over the 50 validation frames

LiteDenoiseNet is the anchor and the reason the exercise is worth running: it reimplements the
MAI 2026 student network (1,963,411 params against the paper's 1.96M, cited in its own docstring),
trained on the same 650/50 challenge split, and it has a checkpoint. If the pipeline cannot land
near its published 37.66 dB, no other row in the table is trustworthy either.

WEIGHTS DECIDE WHAT RUNS WHEN, AND THEY ARE UNEVENLY DISTRIBUTED.
  * LATENCY needs no trained weights -- fp32 TFLite latency is a property of the graph -- so every
    selected model can be re-exported and profiled immediately with no GPU.
  * PSNR needs them. There are 1,679 checkpoints across TWO stores that must both be searched:
      sweep_out/ckpt/<name>/best_model.pth        1,660 -- DenoiseGA (1,298), LLM_Pg*_seeded, SMOKEPAR
      nn-dataset/out/ckpt/<name>/best_model.pth      19 -- the hand-written families incl. LiteDenoiseNet
    Coverage is 42% of the profiled set, and it is NOT random: the DnSeedGA family has ZERO
    checkpoints while holding most of the top of the Final Score ranking. So the models that most
    need re-measuring are exactly the ones that must be retrained first.

SOURCES COME FROM codes.jsonl, NOT THE DB. sweep_out/tflite_pool/codes.jsonl (2,869 rows) is what
the 256x256 conversions were actually built from. Re-exporting at a new resolution from the same
text guarantees the graph being re-profiled is the graph that produced the 256x256 numbers; going
back to the DB would reintroduce the chance of a mismatch for no benefit.

  python reexport_select.py --top 16
"""
import argparse
import json
import os
import re

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
CODES = os.path.join(SWEEP, "tflite_pool", "codes.jsonl")
CKPTS = [os.path.join(SWEEP, "ckpt"), "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/out/ckpt"]
ANCHOR = "LiteDenoiseNet"
ANCHOR_dB = 36.511          # DB stat, img-denoising/denoise/psnr, 50-epoch best
ANCHOR_MS = 0.88            # S26 Ultra, Hexagon v81, 256x256 -- results_lite.json
C_2021 = 2 ** 64
C_2026 = 7.56e18


def find_ckpt(name):
    for base in CKPTS:
        p = os.path.join(base, name, "best_model.pth")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SWEEP, "reexport"))
    ap.add_argument("--top", type=int, default=16, help="top N by 256x256 Final Score, plus the anchor")
    a = ap.parse_args()
    src_dir = os.path.join(a.out, "src")
    os.makedirs(src_dir, exist_ok=True)

    codes = {}
    for line in open(CODES):
        if line.strip():
            r = json.loads(line)
            codes[r["name"]] = r["code"]
    print(f"{len(codes)} sources in codes.jsonl")

    rows = []
    p = os.path.join(SWEEP, "aihub", "results_s26u.json")
    for m in json.load(open(p))["models"]:
        if m.get("inference_us") and m.get("psnr_dB"):
            rows.append({"model": m["model"], "psnr_corpus_y": m["psnr_dB"],
                         "ms_256": round(m["inference_us"] / 1000, 3),
                         "npu_frac_256": m.get("npu_time_frac")})
    if find_ckpt(ANCHOR):
        rows.append({"model": ANCHOR, "psnr_corpus_y": ANCHOR_dB, "ms_256": ANCHOR_MS,
                     "npu_frac_256": 1.0})
    for r in rows:
        r["score_256"] = round(2 ** (2 * r["psnr_corpus_y"]) / (C_2021 * r["ms_256"]), 1)
    rows.sort(key=lambda r: -r["score_256"])

    sel, seen = [], set()
    for r in rows:
        if r["model"] == ANCHOR or len(sel) < a.top:
            if r["model"] in seen:
                continue
            seen.add(r["model"])
            r["why"] = "anchor" if r["model"] == ANCHOR else "top-score"
            sel.append(r)
    # keep rank order but guarantee the anchor is present even if it fell outside --top
    if ANCHOR not in seen:
        for r in rows:
            if r["model"] == ANCHOR:
                r["why"] = "anchor"
                sel.append(r)

    missing_src = []
    for r in sel:
        r["ckpt"] = find_ckpt(r["model"])
        r["has_weights"] = r["ckpt"] is not None
        code = codes.get(r["model"])
        if code is None:
            missing_src.append(r["model"])
            r["src"] = None
            continue
        safe = re.sub(r"[^A-Za-z0-9._+-]", "_", r["model"])
        open(os.path.join(src_dir, safe + ".py"), "w").write(code)
        r["src"] = safe + ".py"

    sel = [r for r in sel if r["src"]]
    need = [r["model"] for r in sel if not r["has_weights"]]
    ready = [r["model"] for r in sel if r["has_weights"]]

    out = {"n": len(sel), "models": sel,
           "resolutions": {"midd_mai2021": [480, 720], "mai2026": [1088, 1920],
                           "psnr_fullres": [2432, 3200]},
           "constants": {"C_2021": C_2021, "C_2026": C_2026},
           "ready_fullres": ready, "need_retrain": need}
    json.dump(out, open(os.path.join(a.out, "subset.json"), "w"), indent=1)

    print(f"\n{len(sel)} models staged into {src_dir}")
    print(f"{'#':>3}{'w':>3}{'256ms':>8}{'corpus dB':>11}{'score':>7}  model")
    for i, r in enumerate(sel, 1):
        print(f"{i:>3}{'Y' if r['has_weights'] else '-':>3}{r['ms_256']:>8.2f}"
              f"{r['psnr_corpus_y']:>11.3f}{r['score_256']:>7.0f}  {r['model'][:42]}")
    if missing_src:
        print(f"\nWARNING {len(missing_src)} selected models have no source in codes.jsonl "
              f"and were dropped: {missing_src[:4]}")
    print(f"\nLATENCY re-export (no weights, no GPU): all {len(sel)}")
    print(f"FULL-RES PSNR now  ({len(ready)}): checkpoints exist")
    print(f"FULL-RES PSNR after retrain ({len(need)}): {sorted({m.split('-')[0] for m in need})}")
    print(f"wrote {os.path.join(a.out, 'subset.json')}")


if __name__ == "__main__":
    main()
