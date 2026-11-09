"""Funnel-integration API for the denoise accuracy predictor (stage 1).

Loads the trained two-stage surrogate (predictor.joblib) and scores a candidate
from its SOURCE + cost, without training it. Used by the GA to cheaply DROP the
bottom of each generation before spending ep5 GPU time. The predictor NEVER
crowns — full-train stays the sole promotion authority (late-bloomers like 10F
must reach it) — it only screens.

score(source, macs, params, name) -> dict:
    p_dead      P(psnr < viable floor)      from the gate classifier
    psnr_pred   predicted PSNR if viable    from the regressor
    score       ranking scalar the GA sorts on:
                (1 - p_dead) * psnr_pred  — pushes likely-dead nets to the bottom
                AND discounts uncertain ones, so the top of the list is the set
                worth training.

rank_and_keep(cands, keep_frac=0.5) -> kept subset (list of the input dicts),
    each annotated with its score — the one call the GA loop makes per generation.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, _HERE)
import features as F  # noqa: E402

_BUNDLE = None


def _load():
    global _BUNDLE
    if _BUNDLE is None:
        import joblib
        _BUNDLE = joblib.load(os.path.join(_HERE, "predictor.joblib"))
    return _BUNDLE


def _row(source, macs, params, name, cols, use_zc):
    f = F.extract(source, macs, params, name)
    vals = dict(f)
    if use_zc:
        import zero_cost as Z
        vals.update(Z.proxies(source))
    return [vals.get(c, float("nan")) for c in cols]


def score(source, macs=None, params=None, name=None):
    """Score one candidate. Returns p_dead, psnr_pred, score (higher=promote)."""
    import numpy as np
    b = _load()
    x = np.array([_row(source, macs, params, name, b["cols"], b["zero_cost"])], float)
    psnr_pred = float(b["regressor"].predict(x)[0])
    # Scale-safety: a live psnr50-target (V2) model is calibrated back to the ep5 scale
    # the GA funnel treats as fitness, so a skipped candidate's predicted fitness never
    # mixes fidelities with the measured-ep5 population. Ranking is unchanged (monotone).
    calib = b.get("calib")
    if calib:
        psnr_pred = calib[0] * psnr_pred + calib[1]
    p_dead = 0.0
    if b.get("gate") is not None:
        p_dead = float(b["gate"].predict_proba(x)[0, 1])
    return {"p_dead": p_dead, "psnr_pred": psnr_pred,
            "score": (1.0 - p_dead) * psnr_pred}


def rank_and_keep(cands, keep_frac=0.5, min_keep=4):
    """cands: list of dicts with keys source, macs, params, name.
    Returns the top keep_frac (each annotated with 'pred'), preserving the rest's
    exclusion. Never keeps fewer than min_keep so a generation can't collapse."""
    scored = []
    for c in cands:
        p = score(c.get("source", ""), c.get("macs"), c.get("params"), c.get("name"))
        d = dict(c)
        d["pred"] = p
        scored.append(d)
    scored.sort(key=lambda d: d["pred"]["score"], reverse=True)
    k = max(min_keep, int(round(len(scored) * keep_frac)))
    return scored[:k]


if __name__ == "__main__":
    # self-test: score a few real ablation models and sanity-check the ordering
    import json
    ga = os.path.dirname(_HERE)
    d = json.load(open(os.path.join(ga, "seed_ga_D0_cache.json")))
    md = os.path.join(ga, "seed_ga_D0_models")
    items = list(d.items())[:6]
    print(f"{'name':36}{'true':>7}{'pred':>7}{'p_dead':>8}{'score':>8}")
    for k, v in items:
        src = open(os.path.join(md, k + ".py")).read()
        s = score(src, v.get("macs"), v.get("params"), v.get("name"))
        print(f"{(v.get('name') or '')[:35]:36}{v.get('psnr',0):7.2f}"
              f"{s['psnr_pred']:7.2f}{s['p_dead']:8.2f}{s['score']:8.2f}")
