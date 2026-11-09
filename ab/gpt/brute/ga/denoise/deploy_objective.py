"""The joint objective: full-resolution RGB PSNR + measured on-device latency + convertibility.

WHY THE OLD OBJECTIVE HAD TO BE REPLACED, NOT TUNED. run3_sft_data.py selects fine-tuning targets
by `stat.accuracy * 48` -- Y-channel PSNR on a single 256x256 centre crop -- and puts that number in
the prompt. Three independent measurements say that signal is unusable:

  * The no-op floor on that metric is 33.021 dB. Across all 20 run3 fine-tune cycles the BEST
    accepted model in any cycle was 33.200 dB, and best-per-cycle showed no trend over 20 cycles
    (33.03 -> 33.20 -> 32.97) with acceptance collapsing to 0 in cycles 18-19. The loop was
    distilling models that barely beat the identity function.
  * The accuracy predictor trained on the same signal scored Spearman -0.178 over 326 production
    models against a claimed +0.693 in cross-validation, and is disabled for that reason.
  * The crop metric is ANTI-CORRELATED with the published protocol at the top of the range. Every
    model above 36 dB on the crop loses 3.3-4.0 dB at full resolution (DenoiseUnet5 37.436 ->
    33.469, the largest drop measured), while models below 34.8 GAIN 2.0-3.6 dB. So optimising the
    crop actively selects against full-frame quality where it matters.

And latency was never in the objective at all: the old builder contains no reference to latency,
inference time or NPU anything. A model's whole reason for existing here is edge deployment.

WHAT THIS USES INSTEAD, and all three are measured, not predicted:
  psnr   full-resolution RGB PSNR over the 50 MAI validation frames (fullres_eval.py). Evaluation
         is deterministic -- 183 accidentally-replicated models agreed to 0.0000 dB -- so rank
         differences are real at any magnitude, unlike the 0.207 dB training noise.
  ms     Qualcomm AI Hub profile, ONE device only. Latencies are NOT comparable across devices
         (rank agreement rho=0.9919 but per-model offset 1.23-2.41x), and the device split is
         uneven by family, so pooling would penalise whichever family drew more of the slower
         phone. S26 Ultra is the default because it has the most coverage.
  conv   the static unconvertible-op rules (npu_gate). A model that cannot become a TFLite graph
         has no latency and cannot ship, so it is excluded outright rather than scored.

THE SCORE IS THE MAI FUNCTIONAL FORM, IN LOG2, AND IT IS ORDINAL ONLY.
The MAI 2026 score is 2^(2*PSNR) / (C * runtime). Taking log2 gives 2*PSNR - log2(runtime) - log2(C),
which is monotone in the same order, does not overflow, and is readable: +1 means twice the score.
The constant is dropped because it only shifts every row equally.

    score = 2 * psnr_db - w * log2(ms)

THIS IS NOT THE PUBLISHED SCORE AND MUST NOT BE REPORTED AS ONE. The published constants are tied
to a resolution (C_2021 = 2^64 at 480x720, C_2026 ~ 7.56e18 at 1088x1920) and our latency is
measured at 256x256, so the absolute value would be meaningless. It ranks our own corpus against
itself; that is all it is for.

WHAT w=1 IMPLIES, STATED BECAUSE IT SURPRISES PEOPLE. Under the published weighting, a 2x latency
increase costs the same as 0.5 dB of PSNR -- so 1 dB of PSNR is worth 4x the runtime. The objective
is therefore PSNR-dominated with latency as a strong tiebreak, by the competition's own choice.
--latency-weight raises w if you want to push harder on speed; the report prints the implied
dB-per-2x-latency trade so the choice is explicit rather than buried.

  python deploy_objective.py --report                 # coverage, top models, trade-off
  python deploy_objective.py --report --min-psnr 33   # only above a floor
"""
import argparse
import glob
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from npu_gate import unconvertible  # noqa: E402

THESIS = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
SWEEP = os.path.join(THESIS, "sweep_out")
RERANK = os.path.join(SWEEP, "rerank")
LATENCY = os.path.join(SWEEP, "aihub", "latency_corpus.json")
DEVICE = "Samsung Galaxy S26 Ultra"

# Published full-resolution RGB no-op: the paper's own "Identity mapping" row. A model below this
# is beaten by returning the input unchanged, so it can never be a fine-tuning target however it
# ranks internally.
NOOP_FULLRES_RGB = 30.88
DEFAULT_LATENCY_WEIGHT = 1.0


def _aliases(name):
    """The corpus carries both `_` and `-` spellings of the same model (see naming.py)."""
    return {name, name.replace("_", "-"), name.replace("-", "_")}


def load_fullres(rerank_dir=None):
    """{model: psnr_rgb_fullres} over every successfully evaluated row, from EVERY eval run.

    Takes the max when a model appears more than once. That is not a judgement call: the 8 shards
    once all evaluated shard 0 and wrote 710 rows for 198 models, and the duplicates were
    bit-identical, so max and first agree wherever it actually happens.

    SCANS ALL OF sweep_out, NOT ONE DIRECTORY. This used to glob `rerank/fullres_shard*.jsonl`
    alone, which silently ignored every other evaluation run -- rerank2 (1,341 models), reexport,
    ft50_fullres and h2h_fullres. It saw 1,578 of 2,955 evaluated models, so the joint objective
    was built on 53% of the available evidence and the deficit was invisible: the coverage line
    printed a plausible number and nothing failed. Any new eval directory is picked up
    automatically now.

    Prefers `psnr_rgb_fullres_shave1` (the published 1-pixel-boundary protocol) and falls back to
    `psnr_rgb_fullres`; the two differ by ~0.003 dB where both exist, but only the first is the
    number the challenge scores.
    """
    out = {}
    if rerank_dir is not None:
        files = sorted(glob.glob(os.path.join(rerank_dir, "fullres_shard*.jsonl")))
    else:
        files = sorted(glob.glob(os.path.join(SWEEP, "**", "fullres*.jsonl"), recursive=True))
    if not files:
        return out
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") not in (None, "OK"):
                continue
            v = r.get("psnr_rgb_fullres_shave1")
            if v is None:
                v = r.get("psnr_rgb_fullres")
            if v is not None and r.get("model"):
                m = r["model"]
                out[m] = max(out.get(m, float("-inf")), float(v))
    return out


def load_latency(path=LATENCY, device=DEVICE):
    """{model: {'ms', 'npu_time_frac'}} for one device only. See the docstring on comparability."""
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    out = {}
    for name, v in (d.items() if isinstance(d, dict) else []):
        # TWO RECORD SHAPES LIVE IN THIS FILE. 2,373 entries carry "status": "OK"; 128 older ones
        # were written without a status field at all, and they are perfectly good measurements --
        # they include the DnSeedGA champions (123a112b at 1.666 ms, 3e4bde5d at 1.917 ms).
        # Requiring status == "OK" silently discarded all 128, which is why the joint objective's
        # target set came back 88% DenoiseGA with ZERO DnSeedGA: not a coverage gap in the data, a
        # filter bug here. A measurement is valid if it HAS a latency; only an explicit non-OK
        # status disqualifies it.
        if not isinstance(v, dict) or not v.get("ms"):
            continue
        if v.get("status") not in (None, "OK"):
            continue
        if device and v.get("device") and v["device"] != device:
            continue
        out[name] = {"ms": float(v["ms"]), "npu_time_frac": v.get("npu_time_frac")}
    return out


def score(psnr_db, ms, w=DEFAULT_LATENCY_WEIGHT):
    """log2 of the MAI functional form, constant dropped. Ordinal within this corpus only."""
    if ms is None or ms <= 0:
        return None
    return 2.0 * psnr_db - w * math.log2(ms)


def db_per_2x_latency(w=DEFAULT_LATENCY_WEIGHT):
    """How much PSNR one may trade for a doubling of runtime under this weight."""
    return w / 2.0


def joint(codes=None, w=DEFAULT_LATENCY_WEIGHT, min_psnr=NOOP_FULLRES_RGB,
          device=DEVICE, require_convertible=True, psnr_map=None):
    """[{model, psnr, ms, npu_time_frac, score}] plus a coverage dict.

    `codes` maps model -> source, used only for the convertibility check. Models with no source
    available are kept but flagged, rather than dropped: absence of source is a bookkeeping gap,
    not evidence the model is unconvertible, and dropping them would silently shrink the set.
    """
    # psnr_map OVERRIDES the full-res scan, and the reason is not convenience. load_fullres()
    # returns whatever full-resolution evaluations exist, and almost all of them are 5-epoch:
    # ft50_fullres and llm_ft50_fullres both hold ZERO rows, so the 50-epoch full-res pass was
    # never completed. A 5-epoch quality number cannot rank converged models -- retraining the
    # champions at 24 epochs inverted their 12-epoch order outright -- so training a generator
    # on 5-epoch targets teaches it to optimise a quantity that does not survive convergence.
    # Callers pass the 50-epoch table here instead, and must pass the matching min_psnr floor:
    # 33.021 dB for the 256-crop Y metric, 30.88 dB for full-res RGB.
    fr = psnr_map if psnr_map is not None else load_fullres()
    lat = load_latency(device=device)
    cov = {"fullres": len(fr), "latency": len(lat), "joined": 0, "below_floor": 0,
           "unconvertible": 0, "no_source": 0, "kept": 0}
    rows = []
    for m, psnr in fr.items():
        hit = next((a for a in _aliases(m) if a in lat), None)
        if hit is None:
            continue
        cov["joined"] += 1
        if psnr < min_psnr:
            cov["below_floor"] += 1
            continue
        src = (codes or {}).get(m)
        if src is None:
            cov["no_source"] += 1
        elif require_convertible:
            bad = unconvertible(src)
            if bad:
                cov["unconvertible"] += 1
                continue
        ms = lat[hit]["ms"]
        rows.append({"model": m, "psnr": psnr, "ms": ms,
                     "npu_time_frac": lat[hit].get("npu_time_frac"),
                     "score": score(psnr, ms, w), "has_source": src is not None})
    cov["kept"] = len(rows)
    rows.sort(key=lambda r: -r["score"])
    return rows, cov


def load_codes_from_db(db=None):
    """{model: code} from the LEMUR DB, for the convertibility half of the objective."""
    import sqlite3
    if db is None:
        try:
            import llm_denoise as L
            db = L.db_path()
        except Exception as e:
            print(f"  (no DB: {type(e).__name__}: {e}); convertibility check skipped")
            return {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("SELECT name, code FROM nn WHERE code IS NOT NULL").fetchall()
        con.close()
    except Exception as e:
        print(f"  (DB read failed: {type(e).__name__}: {e}); convertibility check skipped")
        return {}
    return {n: c for n, c in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--latency-weight", type=float, default=DEFAULT_LATENCY_WEIGHT)
    ap.add_argument("--min-psnr", type=float, default=NOOP_FULLRES_RGB)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default=os.path.join(SWEEP, "deploy_objective.json"))
    a = ap.parse_args()

    codes = load_codes_from_db(a.db)
    rows, cov = joint(codes, w=a.latency_weight, min_psnr=a.min_psnr, device=a.device)

    print(f"COVERAGE  full-res evaluated {cov['fullres']}  |  latency on {a.device} "
          f"{cov['latency']}  |  BOTH {cov['joined']}")
    print(f"  dropped: {cov['below_floor']} below the {a.min_psnr} dB full-res no-op, "
          f"{cov['unconvertible']} unconvertible")
    print(f"  kept: {cov['kept']}   ({cov['no_source']} had no source for the convertibility check)")
    if not rows:
        print("\nNO MODELS SCORED. The joint objective needs BOTH the full-res re-rank and the "
              "AI Hub latency corpus; check that fullres_shard*.jsonl and latency_corpus.json "
              "are populated before using this to build a dataset.")
        return 1
    print(f"\nscore = 2*psnr - {a.latency_weight}*log2(ms)   "
          f"(ordinal only -- NOT the published MAI score; see the module docstring)")
    print(f"trade-off at w={a.latency_weight}: {db_per_2x_latency(a.latency_weight):.2f} dB of PSNR "
          f"per 2x runtime")

    print(f"\nTOP {a.top} on the joint objective:")
    print(f"  {'#':>3} {'model':<46}{'psnr':>8}{'ms':>9}{'npu':>7}{'score':>9}")
    for i, r in enumerate(rows[:a.top], 1):
        nf = f"{r['npu_time_frac']:.2f}" if r["npu_time_frac"] is not None else "-"
        print(f"  {i:>3} {r['model'][:46]:<46}{r['psnr']:>8.3f}{r['ms']:>9.2f}{nf:>7}{r['score']:>9.2f}")

    # The comparison that justifies the whole change: does ranking by the joint objective pick
    # different models than ranking by PSNR alone?
    by_psnr = sorted(rows, key=lambda r: -r["psnr"])
    top_j = {r["model"] for r in rows[:a.top]}
    top_p = {r["model"] for r in by_psnr[:a.top]}
    print(f"\noverlap of top-{a.top} by joint score vs by full-res PSNR alone: "
          f"{len(top_j & top_p)}/{a.top}")
    print(f"  best by PSNR alone:  {by_psnr[0]['model'][:44]} "
          f"{by_psnr[0]['psnr']:.3f} dB @ {by_psnr[0]['ms']:.2f} ms")
    print(f"  best by joint score: {rows[0]['model'][:44]} "
          f"{rows[0]['psnr']:.3f} dB @ {rows[0]['ms']:.2f} ms")

    json.dump({"device": a.device, "latency_weight": a.latency_weight,
               "min_psnr": a.min_psnr, "coverage": cov, "rows": rows},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
