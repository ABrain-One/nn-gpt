"""Activation gate: make the BETTER predictor live, automatically.

The GA funnel loads predictor.joblib. This gate compares the candidate V2 (50-epoch
target, predictor_v2.joblib) against the incumbent V1 (ep5 target, predictor.joblib)
and, if V2 both (a) has enough high-signal labels and (b) generalises better, promotes
V2 to be the live predictor.joblib — after calibrating its output back to the ep5 scale
the funnel treats as fitness, so a live V2 cannot corrupt selection by scale-mixing.

Decision (honest, conservative):
  activate V2  iff  n_v2 >= MIN_N  AND  logo_spearman(V2) >= max(THRESH, logo(V1)+MARGIN)
Rationale: V1 targets ep5, which in production ranks near-randomly (~0.05); V2 targets
50-epoch PSNR (the real objective), so once it has enough labels and clears a real rank
bar it is strictly the better selector. Until then V1 stays live (V2 is not good enough
yet) — nothing regresses.

Run standalone or from the pipeline's --retrain-predictor stage. Idempotent.
"""
import csv, glob, json, os, shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
GA = os.path.dirname(_HERE)
MIN_N = 60          # need this many 50-epoch labels before trusting V2
THRESH = 0.45       # V2 must actually rank (abs LOGO Spearman)
MARGIN = 0.05       # ...and beat V1's generalisation by this much


def _logo(mpath):
    if not os.path.exists(mpath):
        return None, 0
    m = json.load(open(mpath))
    return m["metrics"]["viable_logo"]["spearman"], m.get("n", 0)


def _fit_calibration():
    """ep5 = a*psnr50 + b, from paired labels — maps V2's 50ep output to ep5 scale."""
    csvp = os.path.join(GA, "predictor_labels.csv")
    xs, ys = [], []
    if os.path.exists(csvp):
        for r in csv.DictReader(open(csvp)):
            e5, e50 = r.get("ep5_psnr"), r.get("psnr50")
            if e5 not in (None, "") and e50 not in (None, ""):
                xs.append(float(e50)); ys.append(float(e5))
    if len(xs) < 5:
        return (1.0, 0.0), len(xs)          # identity fallback (too few pairs)
    import numpy as np
    a, b = np.polyfit(xs, ys, 1)
    return (float(a), float(b)), len(xs)


def main():
    v1_logo, v1_n = _logo(os.path.join(_HERE, "metrics.json"))
    v2_logo, v2_n = _logo(os.path.join(_HERE, "metrics_v2.json"))
    v2_path = os.path.join(_HERE, "predictor_v2.joblib")
    print(f"[activate] V1(ep5): LOGO Spearman={v1_logo} n={v1_n}")
    print(f"[activate] V2(50ep): LOGO Spearman={v2_logo} n={v2_n}")

    if v2_logo is None or not os.path.exists(v2_path):
        print("[activate] no V2 yet -> keep V1 live."); return
    bar = max(THRESH, (v1_logo or 0) + MARGIN)
    if v2_n >= MIN_N and v2_logo >= bar:
        import joblib
        calib, npair = _fit_calibration()
        b = joblib.load(v2_path)
        b["calib"] = calib
        live = os.path.join(_HERE, "predictor.joblib")
        if os.path.exists(live):
            shutil.copy2(live, os.path.join(_HERE, "predictor_v1_prev.joblib"))
        joblib.dump(b, live)
        print(f"[activate] *** V2 PROMOTED TO LIVE *** (n={v2_n}>={MIN_N}, "
              f"LOGO {v2_logo:.3f}>={bar:.3f}); ep5-calib a={calib[0]:.3f} b={calib[1]:.3f} "
              f"from {npair} pairs. V1 backed up -> predictor_v1_prev.joblib")
    else:
        why = f"n {v2_n}<{MIN_N}" if v2_n < MIN_N else f"LOGO {v2_logo:.3f}<{bar:.3f}"
        print(f"[activate] V2 not better yet ({why}) -> keep V1 live.")


if __name__ == "__main__":
    main()
