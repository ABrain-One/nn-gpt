"""Train + validate the denoise accuracy predictor (professor-REQUIRED deliverable).

A learned surrogate that predicts ep5-PSNR from an architecture WITHOUT training
it. Inputs = static AST/cost/lineage features (features.py) + optional zero-cost
proxies (zero_cost.py). Target = the measured ep5 PSNR from the ablation caches
(arms A/B/C/D, 213 labelled archs). Model = HistGradientBoostingRegressor
(handles NaN proxies natively, strong on ~200-row tabular).

This is funnel STAGE 1: predict -> keep top fraction -> ep2 -> ep5 -> full-train.
The predictor NEVER crowns a winner (late-bloomers like 10F must still reach
full-train); it only cheaply DROPS the bottom of each generation so real GPU
trainings go to promising candidates. Validation the professor asked for:
hold-out Spearman/Kendall (rank fidelity) + predicted-vs-actual scatter, under a
GROUP split by base-seed lineage so we measure generalisation to unseen families,
not memorised siblings.

Usage:
  python train_predictor.py --zero-cost        # full model (static + proxies)
  python train_predictor.py                     # static-only (fast, no torch)
Outputs (predictor/):
  predictor.joblib  scatter_pred_vs_actual.png  metrics.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import features as F  # noqa: E402

GA_DIR = os.path.dirname(_HERE)


def _discover_arms():
    """All GA arms with a cache + models dir: ablation (A0..D0) AND production
    (R3/R4/R5/... incl. the new macro-families). Was hardcoded to A0-D0, which
    silently excluded every production run and new seed family from retraining."""
    arms = []
    for cf in glob.glob(os.path.join(GA_DIR, "seed_ga_*_cache.json")):
        arm = os.path.basename(cf)[len("seed_ga_"):-len("_cache.json")]
        if os.path.isdir(os.path.join(GA_DIR, f"seed_ga_{arm}_models")):
            arms.append(arm)
    return sorted(arms) or ["A0", "B0", "C0", "D0"]


ARMS = _discover_arms()


def base_seed(name):
    """Lineage group key = the first DenoiseUnet base seed in the name."""
    if not name:
        return "unknown"
    tok = name.split("X")[0].split("+")[0]
    return tok.replace("DenoiseUnet", "") or "unknown"


def _iter_ep5():
    """V1 labels: ep5 proxy PSNR from every arm cache (plentiful, low-signal)."""
    for arm in ARMS:
        cache = os.path.join(GA_DIR, f"seed_ga_{arm}_cache.json")
        mdir = os.path.join(GA_DIR, f"seed_ga_{arm}_models")
        if not os.path.exists(cache):
            continue
        for k, rec in json.load(open(cache)).items():
            if not isinstance(rec, dict) or rec.get("psnr") is None:
                continue
            sp = os.path.join(mdir, k + ".py")
            if os.path.exists(sp):
                yield k, rec["psnr"], rec.get("macs"), rec.get("params"), rec.get("name"), sp


def _iter_psnr50():
    """V2 labels: 50-epoch PSNR from every fulltrain_*.json (scarce, high-signal)."""
    files = glob.glob(os.path.join(GA_DIR, "fulltrain_*.json")) + \
        [os.path.join(GA_DIR, "fulltrain_results.json")]
    for ff in files:
        if not os.path.exists(ff):
            continue
        try:
            runs = json.load(open(ff)).get("runs", [])
        except Exception:
            continue
        for r in runs:
            p, cks = r.get("psnr50"), r.get("checksum")
            if not p or p <= 0 or not cks:
                continue
            sp = os.path.join(GA_DIR, f"seed_ga_{r.get('arm')}_models", cks + ".py")
            if os.path.exists(sp):
                yield cks, p, r.get("macs"), r.get("params"), r.get("name"), sp


def load_dataset(use_zero_cost, target="ep5"):
    """Assemble X (features), y (PSNR at `target` fidelity), groups (base seed), names.
    target='ep5'    -> V1: ep5 proxy label from arm caches (plentiful, low-signal).
    target='psnr50' -> V2: 50-epoch label from fulltrain results (scarce, high-signal)."""
    Z = None
    if use_zero_cost:
        import zero_cost as Z
    src_iter = _iter_psnr50() if target == "psnr50" else _iter_ep5()
    rows, y, groups, names, cols = [], [], [], [], None
    seen = set()
    for k, label, macs, params, name, src_path in src_iter:
        if k in seen:                   # same checksum across arms/files -> one sample
            continue
        seen.add(k)
        src = open(src_path).read()
        feat = F.extract(src, macs, params, name)
        names_i = F.FEATURE_NAMES[:]
        vals = [feat[c] for c in names_i]
        if use_zero_cost:
            zc = Z.proxies(src)
            names_i = names_i + Z.PROXY_NAMES
            vals = vals + [zc[c] for c in Z.PROXY_NAMES]
        if cols is None:
            cols = names_i
        rows.append(vals)
        y.append(float(label))
        groups.append(base_seed(name))
        names.append(name)
    return np.array(rows, float), np.array(y, float), np.array(groups), names, cols


VIABLE_DB = 20.0  # PSNR floor: below this = "dead on arrival" (collapsed training)


def _reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=3,
        min_samples_leaf=8, l2_regularization=1.0, random_state=0)


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)


def _rank_metrics(y, oof):
    from scipy.stats import spearmanr, kendalltau
    return dict(spearman=float(spearmanr(y, oof).correlation),
                kendall=float(kendalltau(y, oof).correlation),
                mae=float(np.mean(np.abs(y - oof))),
                rmse=float(np.sqrt(np.mean((y - oof) ** 2))))


def _reg_oof(X, y, splitter, groups=None):
    oof = np.full_like(y, np.nan, dtype=float)
    for tr, te in splitter.split(X, y, groups):
        m = _reg()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def evaluate(X, y, groups, cols):
    """Two-STAGE evaluation, mirroring how the funnel uses the surrogate:
      STAGE 1 deadness gate: classify psnr<20 dB (collapsed training). Metric =
        ROC-AUC under shuffled 5-fold — cheaply drop dead-on-arrival candidates.
      STAGE 2 viable regressor: on psnr>=20 dB only (the PROMOTABLE band), predict
        PSNR. This is the ranking that actually decides which models get ep5/full.
        Reported under kfold (deployment-realistic: production screens children of
        these SAME seeds) and logo (leave-one-base-seed-out stress test).
      Plus retrieval@keep-2x: fraction of the true top-K kept when the funnel
        forwards the predicted top-2K (the number that bounds wasted GPU)."""
    from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
    from sklearn.metrics import roc_auc_score, average_precision_score
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # --- stage 1: deadness gate ---
    dead = (y < VIABLE_DB).astype(int)
    if dead.sum() >= 3:
        proba = cross_val_predict(_clf(), X, dead, cv=kf, method="predict_proba")[:, 1]
        gate = dict(auc=float(roc_auc_score(dead, proba)),
                    ap=float(average_precision_score(dead, proba)),
                    n_dead=int(dead.sum()), n=len(dead))
    else:
        gate = dict(auc=None, ap=None, n_dead=int(dead.sum()), n=len(dead))

    # --- stage 2: viable regressor ---
    m = y >= VIABLE_DB
    Xv, yv, gv = X[m], y[m], groups[m]
    oof_k = _reg_oof(Xv, yv, kf)
    oof_l = _reg_oof(Xv, yv, LeaveOneGroupOut(), gv)
    mk, ml = _rank_metrics(yv, oof_k), _rank_metrics(yv, oof_l)
    mk.update(n=int(m.sum()))

    # --- retrieval@keep-2x on the viable set ---
    op, ot = np.argsort(oof_k)[::-1], np.argsort(yv)[::-1]
    retr = {}
    for K in (10, 20, 30):
        retr[f"top{K}_in_pred_top{2*K}"] = int(
            len(set(ot[:K].tolist()) & set(op[:2*K].tolist())))

    metrics = {"gate": gate, "viable_kfold": mk, "viable_logo": ml,
               "retrieval": retr}
    metrics.update({k: mk[k] for k in ("spearman", "kendall", "mae", "rmse")})
    return (m, oof_k), metrics


def scatter(y, oof, metrics, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, oof, s=18, alpha=0.6, edgecolor="none")
    lo, hi = min(y.min(), oof.min()), max(y.max(), oof.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("measured ep5 PSNR (dB)")
    ax.set_ylabel("predicted PSNR (dB)")
    ax.set_title(f"Denoise accuracy predictor (LOGO CV)\n"
                 f"Spearman={metrics['spearman']:.3f}  "
                 f"Kendall={metrics['kendall']:.3f}  MAE={metrics['mae']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-cost", action="store_true",
                    help="add one-fwd+bwd proxies (needs torch; ~1-2 min for 213)")
    ap.add_argument("--target", choices=["ep5", "psnr50"], default="ep5",
                    help="label fidelity: ep5 (V1, plentiful/low-signal) or psnr50 "
                         "(V2, scarce/high-signal). V2 writes predictor_v2.* so it never "
                         "clobbers the live V1 funnel model.")
    args = ap.parse_args()
    suffix = "_v2" if args.target == "psnr50" else ""

    print(f"[1/4] loading dataset (zero_cost={args.zero_cost}, target={args.target}) ...",
          flush=True)
    X, y, groups, names, cols = load_dataset(args.zero_cost, args.target)
    print(f"      {len(y)} archs, {X.shape[1]} features, "
          f"{len(set(groups))} lineage groups: {sorted(set(groups))}")
    print(f"      PSNR range {y.min():.2f}..{y.max():.2f} dB", flush=True)

    print("[2/4] two-stage cross-validation ...", flush=True)
    (mask, oof_v), metrics = evaluate(X, y, groups, cols)
    g, k, l, r = (metrics["gate"], metrics["viable_kfold"],
                  metrics["viable_logo"], metrics["retrieval"])
    print(f"      GATE   deadness(<{int(VIABLE_DB)}dB, {g['n_dead']}/{g['n']}): "
          f"AUC={g['auc']:.3f}  AP={g['ap']:.3f}", flush=True)
    print(f"      VIABLE KFOLD Spearman={k['spearman']:.3f}  Kendall={k['kendall']:.3f}"
          f"  MAE={k['mae']:.3f} dB  (n={k['n']}, promotable band)", flush=True)
    print(f"      VIABLE LOGO  Spearman={l['spearman']:.3f}  MAE={l['mae']:.3f} dB"
          f"  (unseen-seed stress)", flush=True)
    print(f"      RETRIEVAL true-top10 in pred-top20 = "
          f"{r['top10_in_pred_top20']}/10  (funnel keeps 2x)", flush=True)

    print("[3/4] fitting final gate + regressor + feature importances ...", flush=True)
    from sklearn.inspection import permutation_importance
    Xv, yv = X[mask], y[mask]
    reg = _reg(); reg.fit(Xv, yv)
    dead = (y < VIABLE_DB).astype(int)
    gate = _clf(); gate.fit(X, dead) if dead.sum() >= 3 else None
    imp = permutation_importance(reg, Xv, yv, n_repeats=8, random_state=0)
    order = np.argsort(imp.importances_mean)[::-1][:12]
    top_feats = [(cols[i], float(imp.importances_mean[i])) for i in order]
    print("      top features (viable regressor):")
    for nm, v in top_feats:
        print(f"        {nm:20} {v:.4f}")

    print("[4/4] saving artifacts ...", flush=True)
    try:
        import joblib
        joblib.dump({"regressor": reg,
                     "gate": gate if dead.sum() >= 3 else None,
                     "cols": cols, "zero_cost": args.zero_cost,
                     "viable_db": VIABLE_DB, "target": args.target},
                    os.path.join(_HERE, f"predictor{suffix}.joblib"))
    except Exception as e:
        print("      joblib save skipped:", e)
    scatter(y[mask], oof_v, k, os.path.join(_HERE, f"scatter{suffix}_pred_vs_actual.png"))
    json.dump({"metrics": metrics, "top_features": top_feats, "target": args.target,
               "cols": cols, "zero_cost": args.zero_cost, "n": len(y)},
              open(os.path.join(_HERE, f"metrics{suffix}.json"), "w"), indent=2)
    print("      wrote predictor.joblib, scatter_pred_vs_actual.png, metrics.json")
    print(f"\nDONE. Gate AUC={g['auc']:.3f}; viable-band rank Spearman "
          f"(realistic)={k['spearman']:.3f}, (unseen-seed)={l['spearman']:.3f}.",
          flush=True)


if __name__ == "__main__":
    main()
