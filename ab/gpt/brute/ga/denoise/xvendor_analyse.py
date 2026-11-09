"""Does the paper's mechanism hold off Qualcomm? Three tests on the paired cross-vendor runs.

Q1 DISPERSION. Within windows of +-5% of the same MAC cost, what is the interdecile (p90/p10)
   latency ratio on each vendor's silicon? The paper reports 7x-140x on Snapdragon. If MACs
   determined latency this would be ~1 everywhere.
Q2 MECHANISM. Is the spread explained by whether the compiler keeps the whole graph on the
   accelerator? Reported as the residual dispersion after conditioning on residency class,
   which on Snapdragon removes 97%.
Q3 TRANSFER. The frozen gate in docs/deploy_gate_model.json was fitted on Qualcomm handsets
   ONLY and has never seen Tensor or Exynos silicon. Its AUC against each new device's own
   eviction label is therefore an out-of-vendor transfer measurement, not a refit.

A NOTE ON WHAT A NULL RESULT WOULD MEAN. If the gate does not transfer, that is reportable and
does not damage Q1/Q2: the claim "a MAC budget does not predict deployability" would still
hold on three vendors, and the gate would simply be a per-vendor instrument. The hypotheses
are written down here before the data is read so that either outcome is publishable.
"""
import collections
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
OUTDIR = os.path.join(SWEEP, "xvendor")
REF = os.path.join(SWEEP, "aihub", "latency_corpus.json")
REF_DEVICE = "Samsung Galaxy S26 Ultra"
MAC_WINDOW = 0.05


def residency(frac):
    if frac is None:
        return "unknown"
    if frac >= 0.999:
        return "resident"
    if frac <= 0.001:
        return "fully_evicted"
    return "partial"


def quantile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    i = q * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def auc(y, p):
    """Rank-based AUC with tie correction. Returns None if a class is absent."""
    pos = [s for t, s in zip(y, p) if t == 1]
    neg = [s for t, s in zip(y, p) if t == 0]
    if not pos or not neg:
        return None
    pairs = sorted(zip(p, y))
    ranks = {}
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    sp = sum(ranks[k] for k in range(len(pairs)) if pairs[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (sp - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def load_all():
    res = json.load(open(os.path.join(OUTDIR, "results.json")))
    ref = json.load(open(REF))
    macs = json.load(open(os.path.join(SWEEP, "verify_macs.json")))["macs"]
    rows = collections.defaultdict(dict)       # device_key -> model -> rec
    for r in res.values():
        if r.get("status") != "OK" or not r.get("ms"):
            continue
        rows[(r["vendor"], r["chipset"])][r["model"]] = r
    qual = {}
    for n, e in ref.items():
        if isinstance(e, dict) and e.get("status") == "OK" and e.get("ms") \
                and e.get("device") == REF_DEVICE:
            qual[n] = e
    rows[("qualcomm", "snapdragon-8elite(S26U, reference)")] = qual
    return rows, macs


def q1_q2(rows, macs):
    print("=" * 96)
    print("Q1/Q2  fixed-MAC latency dispersion, and how much of it residency explains")
    print("=" * 96)
    print(f"{'vendor':9s} {'chipset':34s} {'n':>5} {'windows':>8} {'p90/p10':>9} "
          f"{'cond.':>8} {'removed':>8}")
    summary = {}
    for key in sorted(rows, key=lambda k: (k[0] != "qualcomm", k[1])):
        recs = rows[key]
        pts = []
        for n, r in recs.items():
            m = macs.get(n)
            if m is None:
                continue
            pts.append((float(m), r["ms"], residency(r.get("npu_time_frac"))))
        if len(pts) < 20:
            print(f"{key[0]:9s} {key[1]:34s} {len(pts):>5}   (too few paired MAC rows)")
            continue
        pts.sort()
        raw, cond = [], []
        nwin = 0
        for i, (m0, _, _) in enumerate(pts):
            win = [p for p in pts if abs(p[0] - m0) <= MAC_WINDOW * m0]
            if len(win) < 8:
                continue
            nwin += 1
            lat = [p[1] for p in win]
            lo, hi = quantile(lat, 0.10), quantile(lat, 0.90)
            if lo > 0:
                raw.append(hi / lo)
            # conditioned: same window, residency class held fixed at the majority class
            cls = collections.Counter(p[2] for p in win).most_common(1)[0][0]
            sub = [p[1] for p in win if p[2] == cls]
            if len(sub) >= 8:
                lo2, hi2 = quantile(sub, 0.10), quantile(sub, 0.90)
                if lo2 > 0:
                    cond.append(hi2 / lo2)
        if not raw:
            print(f"{key[0]:9s} {key[1]:34s} {len(pts):>5}   (no populated MAC windows)")
            continue
        mr = quantile(raw, 0.5)
        mc = quantile(cond, 0.5) if cond else float("nan")
        rem = (1 - (mc - 1) / (mr - 1)) * 100 if (cond and mr > 1) else float("nan")
        print(f"{key[0]:9s} {key[1]:34s} {len(pts):>5} {nwin:>8} {mr:>9.2f} "
              f"{mc:>8.2f} {rem:>7.1f}%")
        summary[f"{key[0]}|{key[1]}"] = {
            "n_paired_macs": len(pts), "n_windows": nwin,
            "median_interdecile_ratio": round(mr, 3),
            "median_ratio_within_residency_class": None if cond != cond else round(mc, 3),
            "pct_dispersion_removed": None if rem != rem else round(rem, 1),
            "max_interdecile_ratio": round(max(raw), 2)}
    print("\n  'p90/p10'  = median over MAC windows of the interdecile latency ratio")
    print("  'cond.'    = same, with residency class held fixed inside the window")
    print("  'removed'  = share of the excess ratio that conditioning on residency explains")
    return summary


def q3(rows):
    print()
    print("=" * 96)
    print("Q3  does the frozen Qualcomm eviction gate transfer to other vendors?")
    print("=" * 96)
    from deploy_gate import evict_prob, load as gload
    m = gload()
    print(f"frozen gate: {m['n_train']} training models, {m['n_evicted']} evicted, "
          f"{len(m['keys'])} AST features -- fitted on Qualcomm only\n")
    db = "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/db/ab.nn.db"
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    code = {n: c for n, c in con.execute("select name, code from nn")}
    canon = {n.replace("_", "-"): n for n in code if "_" in n}
    print(f"{'vendor':9s} {'chipset':34s} {'n':>5} {'evicted':>8} {'rate':>7} {'AUC':>7}")
    out = {}
    for key in sorted(rows, key=lambda k: (k[0] != "qualcomm", k[1])):
        y, p = [], []
        for n, r in rows[key].items():
            c = code.get(n) or code.get(canon.get(n, ""))
            if not c:
                continue
            pr = evict_prob(c)
            if pr is None:
                continue
            y.append(1 if (r.get("npu_time_frac") or 0.0) <= 0.0 else 0)
            p.append(pr)
        if len(y) < 20:
            print(f"{key[0]:9s} {key[1]:34s} {len(y):>5}   (too few with recoverable source)")
            continue
        a = auc(y, p)
        rate = sum(y) / len(y)
        print(f"{key[0]:9s} {key[1]:34s} {len(y):>5} {sum(y):>8} {rate:>6.1%} "
              f"{(a if a is not None else float('nan')):>7.3f}")
        out[f"{key[0]}|{key[1]}"] = {"n": len(y), "n_evicted": int(sum(y)),
                                     "eviction_rate": round(rate, 4),
                                     "auc_transfer": None if a is None else round(a, 4)}
    print("\n  The reference row is in-sample (the gate was fitted on it) and is shown only")
    print("  as the ceiling. The non-Qualcomm rows are out-of-vendor transfer.")
    return out


def main():
    if not os.path.exists(os.path.join(OUTDIR, "results.json")):
        return "no results yet: run xvendor_profile.py --collect first"
    rows, macs = load_all()
    s1 = q1_q2(rows, macs)
    s3 = q3(rows)
    p = os.path.join(OUTDIR, "analysis.json")
    json.dump({"mac_window_frac": MAC_WINDOW, "dispersion": s1, "gate_transfer": s3},
              open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    raise SystemExit(main())
