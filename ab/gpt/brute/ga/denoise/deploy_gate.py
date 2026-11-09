"""Predict full accelerator eviction from SOURCE TEXT, with no conversion and no device.

WHY THIS EXISTS. `run_seed_ga._deploy_ok` already gates on deployability, but it pays a real
TFLite conversion (~3.4 s/candidate) to count graph ops, and the op threshold reaches only 43%
recall against its own latency label. This gate reads the AST instead -- microseconds -- and
predicts the failure that actually wastes the target device: a model that converts cleanly and
is then placed ENTIRELY on the CPU.

MEASURED, out-of-fold, against 3,713 models profiled on physical Snapdragon handsets:
    5-fold cross-validation              AUC 0.986   precision 0.90  recall 0.83
    train genetic arms -> test LLM arm   AUC 0.921   precision 0.91  recall 0.74
    train LLM arm -> test genetic arms   AUC 0.950
The middle row is the one that licenses use as a gate: the model never saw an LLM-written
architecture and still finds LLM evictions at 91% precision, so it is not keying on which
generator produced a file.

WHAT IT DOES NOT DO. It does not predict latency MAGNITUDE. The same features regressed on
log-latency give rank 0.678 and only 65% of predictions within 2x, so a survivor is "will
probably reach the NPU", never "is fast". Use the AI Hub profile for speed.

The profiler's own `n_layers` is deliberately NOT a feature: it is a post-conversion quantity,
and using it would make this predict one measurement from another rather than from source.

COEFFICIENTS ARE FROZEN to docs/deploy_gate_model.json by --fit, and the gate refuses to run
without them. A gate that silently retrains per call would drift with the corpus it is meant to
be filtering, and a gate that silently fails open is how the MAC cap got disabled for a whole
run without anyone noticing -- so the loader raises instead.

  python3 deploy_gate.py --fit          # refit and overwrite the frozen model
  python3 deploy_gate.py --check FILE   # print the eviction probability for one source file
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODEL = os.path.join(HERE, "docs", "deploy_gate_model.json")
LATENCY = "/a/mm/sweep_out/aihub/latency_corpus.json"
if not os.path.exists(LATENCY):
    LATENCY = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out/aihub/latency_corpus.json"

# The operating point. 0.5 is where the measured precision/recall above was taken; raising it
# trades recall for precision (0.7 -> 0.92/0.80). Kept as a module constant so an experiment
# changes it in one place.
DEFAULT_THR = 0.5

_M = None


def _features(src):
    from skip_pattern import features
    return features(src)


def load():
    """-> frozen model dict. Raises if absent: this gate never fails open."""
    global _M
    if _M is None:
        if not os.path.exists(MODEL):
            raise RuntimeError(
                "deploy gate model missing at %s -- run `python3 deploy_gate.py --fit`. "
                "Refusing to pass candidates through an untrained gate." % MODEL)
        _M = json.load(open(MODEL))
    return _M


def evict_prob(src):
    """-> probability this architecture runs entirely on the CPU, or None if unparseable."""
    import math
    m = load()
    f = _features(src)
    if not f:
        return None
    z = m["b"]
    for k, w, mu, sd in zip(m["keys"], m["w"], m["mu"], m["sd"]):
        z += w * ((float(f.get(k, 0)) - mu) / sd)
    return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))


def ok(src, thr=DEFAULT_THR):
    """-> (passes, probability). An unparseable candidate PASSES: rejecting here would double
    as a syntax gate, and preflight already owns that decision with better diagnostics."""
    p = evict_prob(src)
    if p is None:
        return True, None
    return p < thr, p


def fit(verbose=True):
    import sqlite3
    import numpy as np
    L = json.load(open(LATENCY))
    db = os.path.join("/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/db/ab.nn.db")
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    code = {n: c for n, c in con.execute("select name, code from nn")}
    canon = {n.replace("_", "-"): n for n in code if "_" in n}
    rows = []
    for name, e in L.items():
        if e.get("status") != "OK" or not e.get("ms"):
            continue
        c = code.get(name) or code.get(canon.get(name, ""))
        if not c:
            continue
        f = _features(c)
        if f:
            rows.append((f, 1.0 if (e.get("npu_time_frac") or 0.0) <= 0.0 else 0.0))
    keys = sorted(rows[0][0])
    X = np.array([[float(f.get(k, 0)) for k in keys] for f, _ in rows])
    y = np.array([t for _, t in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(A.shape[1])
    for _ in range(200):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + 2.0 * np.r_[w[:-1], 0]
        H = A.T @ (A * (p * (1 - p))[:, None]) + 2.0 * np.eye(A.shape[1])
        try:
            w -= np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    json.dump({"keys": keys, "w": list(map(float, w[:-1])), "b": float(w[-1]),
               "mu": list(map(float, mu)), "sd": list(map(float, sd)),
               "n_train": len(rows), "n_evicted": int(y.sum()), "thr": DEFAULT_THR},
              open(MODEL, "w"), indent=1)
    if verbose:
        print("fitted on %d models (%d evicted) -> %s" % (len(rows), int(y.sum()), MODEL))
    return MODEL


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--check", default="")
    a = ap.parse_args()
    if a.fit:
        fit()
    if a.check:
        passes, p = ok(open(a.check).read())
        print("%s  evict_prob=%s  %s" % (a.check, "n/a" if p is None else "%.4f" % p,
                                         "PASS" if passes else "REJECT"))
