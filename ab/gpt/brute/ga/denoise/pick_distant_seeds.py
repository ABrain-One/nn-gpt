"""Choose GA seeds that are maximally far apart, instead of by hand.

WHY SEED CHOICE IS THE WHOLE BALLGAME. metric_ablation/DECISION_LEDGER.md records the finding
that decides this: "mutation never escapes the family cluster (embedding-space proof of the
in-family collapse)". Recalibration on 4,174 real pairs agrees -- same-family pairs have a
MEDIAN cosine of 1.000 and a 5th percentile of 0.926, while cross-family pairs sit at a median
of 0.752. A mutation-only GA therefore explores the neighbourhood of its seeds and nothing
else, however many generations it runs.

That makes the seed set the actual search space. Today it is hand-picked per variant -- six
named families chosen for a plausible-sounding reason ("the cheapest legal ones", "distinct
structural motifs"). Nobody measured whether those six are far apart, and the corpus says
several are not: at cosine 0.95 the whole 1,175-model corpus collapses to 227 clusters, and
the largest single cluster holds 172 models.

FARTHEST-POINT SAMPLING, which is the standard answer to "give me k maximally spread items".
Start from the best-scoring model, then repeatedly add whichever candidate is FURTHEST from
everything already chosen -- furthest meaning lowest maximum cosine to the selected set. The
result has a guaranteed property worth reporting: the minimum pairwise separation of the
chosen set, which is exactly the number a hand-picked list cannot claim.

QUALITY IS A FILTER, NOT THE OBJECTIVE. Seeding on the top-k by PSNR is what produces a
monoculture -- run 1 collapsed onto LiteDenoiseNet with 291 of 300 models in one family. So
candidates must clear a floor (edge-legal, beats the no-op baseline) and diversity decides
everything after that. Ranking by score inside the eligible pool would quietly reintroduce the
bias this script exists to remove.

  python pick_distant_seeds.py --k 12 --min-db 33.5
"""
import argparse
import collections
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

TASK = "img-denoising"
IN_SHAPE = (1, 3, 256, 256)
OUT_SHAPE = (3,)
PRM = {"lr": 1e-4, "batch": 16, "epoch_max": 5, "transform": "denoise"}
NOOP_DB = 33.02          # measured: returning the noisy input unchanged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--min-db", type=float, default=33.5,
                    help="quality floor; default is comfortably above the 33.02 no-op")
    ap.add_argument("--mac-cap", type=float, default=9053.0)
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/distant_seeds.json")
    a = ap.parse_args()

    import numpy as np
    from arch_embed import embed

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT n.name, n.code, s.db FROM nn n JOIN "
        "(SELECT nn, MAX(accuracy)*48 db FROM stat WHERE task=? AND accuracy IS NOT NULL "
        " GROUP BY nn) s ON s.nn = n.name WHERE s.db >= ?", (TASK, a.min_db)).fetchall()
    con.close()
    print(f"=== distant-seed selection | {len(rows)} models above {a.min_db} dB "
          f"({a.min_db - NOOP_DB:+.2f} over the no-op floor) ===", flush=True)

    # MAC gate, reusing the cached verdicts so this stays a seconds-long job.
    cache_p = "/a/mm/sweep_out/manifest/mac_gate_cache.json"
    mac_ok = {}
    if os.path.exists(cache_p):
        try:
            mac_ok = json.load(open(cache_p))
        except Exception:
            pass
    import hashlib
    try:
        import edge_cost as EC
        have_ec = True
    except Exception:
        have_ec = False
        print("  WARNING edge_cost unimportable -- MAC cap NOT enforced", flush=True)

    names, vecs, dbs, dropped = [], [], [], 0
    for i, (nm, code, dbv) in enumerate(rows, 1):
        if have_ec:
            h = hashlib.sha256(code.encode()).hexdigest()[:32]
            if h in mac_ok:
                if not mac_ok[h]:
                    dropped += 1
                    continue
            else:
                try:
                    r = EC.measure_cost(code)
                    ok = bool(r.get("ok")) and (r.get("macs", 0) / 1e6) <= a.mac_cap
                except Exception:
                    ok = True
                mac_ok[h] = ok
                if not ok:
                    dropped += 1
                    continue
        try:
            ns = {}
            exec(compile(code, "<m>", "exec"), ns)
            net = ns["Net"](IN_SHAPE, OUT_SHAPE, dict(PRM), "cpu")
            net.eval()
            v = np.asarray(embed(net), dtype=np.float64).ravel()
            n = np.linalg.norm(v)
            if n > 0 and np.isfinite(n):
                names.append(nm)
                vecs.append(v / n)
                dbs.append(dbv)
            del net
        except Exception:
            pass
        if i % 150 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] embedded={len(names)} over-cap={dropped}", flush=True)

    if len(vecs) < a.k:
        print(f"only {len(vecs)} eligible models; cannot pick {a.k}")
        return 1
    w = collections.Counter(v.shape[0] for v in vecs).most_common(1)[0][0]
    keep = [i for i, v in enumerate(vecs) if v.shape[0] == w]
    M = np.stack([vecs[i] for i in keep])
    names = [names[i] for i in keep]
    dbs = [dbs[i] for i in keep]
    print(f"  {M.shape[0]} eligible, embedding dim {w}\n", flush=True)

    # ---- farthest-point sampling on cosine.
    # Seeded from the best scorer only so the run is deterministic; every subsequent pick is
    # chosen purely for distance, never for score -- that separation is the point.
    sel = [int(np.argmax(dbs))]
    worst = M[sel] @ M.T                      # max cosine to the selected set, per candidate
    worst = worst.max(axis=0)
    for _ in range(a.k - 1):
        nxt = int(np.argmin(worst))
        sel.append(nxt)
        worst = np.maximum(worst, M[nxt] @ M.T)

    S = M[sel]
    C = S @ S.T
    np.fill_diagonal(C, -1)
    print(f"=== {a.k} maximally distant seeds ===")
    print(f"  {'#':>2}  {'model':<46}{'dB':>7}{'max cos to others':>19}")
    for r, i in enumerate(sel, 1):
        mc = float(C[r - 1].max())
        print(f"  {r:2d}  {names[i][:46]:<46}{dbs[i]:7.2f}{mc:19.3f}")
    print(f"\n  worst pair among the chosen : {C.max():.3f}")
    print(f"  mean pairwise cosine        : {C[C > -1].mean():.3f}")
    print(f"  (cross-family pairs measure ~0.752 median; same-family ~1.000)")

    # ---- what a hand-picked list actually scores, for comparison.
    hand = ["DenoiseSelfAttn", "DenoiseMBConv", "DenoiseChanShuffle",
            "DenoiseSplitterNetF", "DenoiseSubPixel", "DenoisePoolDeconv"]
    idx = [names.index(h) for h in hand if h in names]
    if len(idx) > 1:
        H = M[idx]
        HC = H @ H.T
        np.fill_diagonal(HC, -1)
        print(f"\n=== the hand-picked 'struct' seed set, measured ===")
        print(f"  {len(idx)} of {len(hand)} found above the quality floor")
        print(f"  worst pair : {HC.max():.3f}")
        print(f"  mean       : {HC[HC > -1].mean():.3f}")
        print(f"  => selection improves worst-pair separation by "
              f"{HC.max() - C.max():+.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"k": a.k, "min_db": a.min_db,
               "seeds": [{"name": names[i], "db": dbs[i]} for i in sel],
               "worst_pair_cosine": float(C.max()),
               "mean_pair_cosine": float(C[C > -1].mean())},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("\nUse with:  SEEDS_DB=\"" + " ".join(names[i] for i in sel[:6]) + "\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
