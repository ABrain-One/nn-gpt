"""Post-sweep analysis of the canonical 50-epoch run. Answers the two questions the
sweep was actually run to settle.

Q1. IS THE 5-EPOCH PROXY VALID?
    julia2.md justifies scoring GA candidates at 5 epochs with "Spearman ep5-vs-ep50 =
    0.90". Mid-sweep that looked wrong: DenoiseMBConv led DenoiseUnet9F by +0.47 dB at
    epoch 5 but trailed it by -0.22 by epoch 24 -- the order reversed. This recomputes
    the correlation directly on THIS corpus, where every model now has both an ep5 and an
    ep50 number under the identical recipe (lr 1e-4 / batch 16). If rho is materially
    below 0.90, the GA is ranking ~2000 candidates on a signal that does not survive to
    the budget we actually care about.

Q2. DOES THE PROFESSOR'S PREDICTOR HOLD UP AT n=23?
    Validated so far on only the 6 full-curve seeds (Spearman 0.84, MAE ~1.2-1.5 dB when
    fed dB). The sweep adds 17 matched (ep1-3 -> ep50) pairs. It also lets us test the
    bias found at n=6: the predictor systematically UNDER-predicts, worst for models with
    poor early epochs (1F opened at 24.38 dB -> -3.30 error). If that holds, it penalises
    late bloomers and must stay screening-only.

Reads the per-pod exported DBs in sweep_out/canonical/ (never the master DB).
Usage: python analyse_canonical.py [sweep_out_dir]
"""
import glob
import os
import sqlite3
import sys

TASK, DATASET, METRIC, SCALE = "img-denoising", "denoise", "psnr", 48.0


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[s[k]] = avg
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def curve(db_path, nn):
    """{epoch: dB} for one model from one DB file."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT epoch, accuracy FROM stat WHERE nn=? AND task=? AND dataset=? "
            "AND metric=?", (nn, TASK, DATASET, METRIC)).fetchall()
        con.close()
    except Exception:
        return {}
    by = {}
    for e, a in rows:
        if a is not None and (e not in by or a > by[e]):
            by[e] = a * SCALE
    return by


def main(outdir):
    dbs = sorted(glob.glob(os.path.join(outdir, "db_*.db")))
    if not dbs:
        print(f"no exported DBs in {outdir} -- sweep not finished?")
        return
    master = "/a/mm/nn-dataset/db/ab.nn.db"
    if not os.path.exists(master):
        master = os.path.join(os.path.dirname(__file__),
                              "../../../../../../nn-dataset/db/ab.nn.db")

    rows = []
    for p in dbs:
        nn = os.path.basename(p)[3:-3]           # db_<NAME>.db
        c50 = curve(p, nn)
        if not c50:
            continue
        c5 = curve(master, nn)                   # the stored 5-epoch run
        ep5 = c5.get(5) or (max(c5.values()) if c5 else None)
        seq = [c50[e] for e in sorted(c50)]
        last5 = sum(seq[-5:]) / len(seq[-5:])
        rows.append({
            "nn": nn, "ep5": ep5,
            "best50": max(c50.values()), "best_ep": max(c50, key=lambda k: c50[k]),
            "n": len(c50), "max_ep": max(c50),
            "e1": c50.get(1), "e2": c50.get(2), "e3": c50.get(3),
            "final": c50[max(c50)],
            "avg": sum(c50.values()) / len(c50), "min": min(c50.values()),
            # STABLE metric. `best50` is a max over 50 epochs, so it rewards VOLATILITY:
            # a noisy curve earns a higher peak from variance alone. Measured on this
            # corpus, DenoiseSubPixel peaks at 35.58 (above 9F's 35.48) but sits at 35.25
            # on last5 vs 9F's 35.43 -- 0.34 dB of its "win" was a transient excursion,
            # and DenoiseParallelMR drops from 4th by peak to LAST by last5. Rank by this.
            "last5": last5,
            "spike": max(c50.values()) - last5,
        })
    rows.sort(key=lambda r: -r["last5"])

    print(f"=== CANONICAL 50-EPOCH RESULTS (lr 1e-4, batch 16) -- {len(rows)} models ===\n")
    h = (f"{'model':<22}{'last5':>8}{'best':>8}{'@ep':>5}{'spike':>7}"
         f"{'avg':>8}{'min':>8}{'ep5':>8}{'gain':>8}")
    print(h); print("-" * len(h))
    for r in rows:
        g = f"{r['last5'] - r['ep5']:+.2f}" if r["ep5"] else "   n/a"
        c = "^" if r["best_ep"] == r["max_ep"] else " "
        print(f"{r['nn']:<22}{r['last5']:>8.2f}{r['best50']:>8.2f}{r['best_ep']:>4}{c}"
              f"{r['spike']:>7.2f}{r['avg']:>8.2f}{r['min']:>8.2f}"
              f"{(r['ep5'] if r['ep5'] else 0):>8.2f}{g:>8}")
    print("\n^ = peak at final epoch (50 still binding).  "
          "spike = best - last5: transient excursion in the peak.")
    big = [r for r in rows if r["spike"] > 0.25]
    if big:
        print(f"  !! {len(big)} model(s) with spike > 0.25 dB -- their `best` overstates "
              f"them: {', '.join(r['nn'] for r in big)}")

    # ---- Q1: does ep5 rank order survive to ep50? ----
    # Computed on BOTH metrics: if rho is much lower on `best50` than on `last5`, the
    # apparent rank instability is spike noise in the peak rather than genuine proxy
    # failure -- a different diagnosis with a different fix.
    print(f"\n=== Q1  ep5 -> ep50 rank correlation ===")
    for key, lbl in (("last5", "last5 (stable)"), ("best50", "best  (peak)")):
        pairs = [(r["ep5"], r[key]) for r in rows if r["ep5"]]
        if len(pairs) < 3:
            continue
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"  vs {lbl:<16} Spearman rho = {rho:+.3f}   (n={len(pairs)})")
    pairs = [(r["ep5"], r["last5"]) for r in rows if r["ep5"]]
    if len(pairs) >= 3:
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"  julia2.md claims 0.90. VERDICT:",
              "consistent with the claim" if rho >= 0.80 else
              "MATERIALLY BELOW 0.90 -- the 5-epoch proxy does not preserve rank on this "
              "corpus, so GA selection at --epoch 5 ranks on a signal that does not "
              "survive to 50 epochs.")
        gains = [r["last5"] - r["ep5"] for r in rows if r["ep5"]]
        print(f"  mean ep5->ep50 gain = {sum(gains)/len(gains):+.2f} dB "
              f"(min {min(gains):+.2f}, max {max(gains):+.2f})")

    # ---- Q2: predictor at the enlarged n ----
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from predictor import prof_acc
        if not prof_acc.available():
            print("\n=== Q2  predictor unreachable -- skipped ==="); return
        # /a/mm is the in-CONTAINER mount; this script also runs on the submit node,
        # where the same tree lives under /shared/ssd/home/... Resolve both, else Q2
        # silently prints a header and no rows.
        nn_dir = next((d for d in (
            "/a/mm/nn-dataset/ab/nn/nn",
            "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/ab/nn/nn",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../../../../../nn-dataset/ab/nn/nn"),
        ) if os.path.isdir(d)), None)
        if not nn_dir:
            print("\n=== Q2 skipped: cannot locate ab/nn/nn ==="); return
        print(f"\n=== Q2  predictor (fed dB) vs measured ep50 ===")
        print(f"{'model':<22}{'pred':>8}{'truth':>8}{'err':>8}{'ep1':>8}")
        print("-" * 54)
        P, T, errs = [], [], []
        for r in rows:
            if not all(r[k] for k in ("e1", "e2", "e3")):
                continue
            src = os.path.join(nn_dir, f"{r['nn']}.py")
            if not os.path.exists(src):
                continue
            p = prof_acc.predict_db(open(src).read(), r["e1"], r["e2"], r["e3"])
            if p is None:
                continue
            P.append(p); T.append(r["last5"]); errs.append(p - r["last5"])
            print(f"{r['nn']:<22}{p:>8.2f}{r['last5']:>8.2f}{p - r['last5']:>+8.2f}"
                  f"{r['e1']:>8.2f}")
        if len(P) >= 3:
            print(f"\n  n={len(P)}  Spearman rho={spearman(T, P):+.3f}  "
                  f"MAE={sum(abs(e) for e in errs)/len(errs):.2f} dB  "
                  f"bias={sum(errs)/len(errs):+.2f} dB")
            neg = sum(1 for e in errs if e < 0)
            print(f"  under-predicts {neg}/{len(errs)} "
                  f"-> {'systematic under-prediction confirmed' if neg > 0.7*len(errs) else 'no strong bias'}")
    except Exception as e:
        print(f"\n=== Q2 skipped: {type(e).__name__}: {e} ===")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/shared/ssd/home/b-b-ahmed/thesis/sweep_out/canonical")
