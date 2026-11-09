"""Pick 40 architectures SPANNING the AWGN performance range for the cross-dataset study.

THE QUESTION IS RANK TRANSFER, SO SPREAD IS THE WHOLE DESIGN. "Do our top models also win on
real noise?" cannot be answered by training our top models: if every candidate scores within
a narrow band, the correlation is destroyed by range restriction regardless of what is true.
That is not hypothetical here -- the beatbar subset was chosen as the top 12 by predicted
50-epoch PSNR, all landed inside a 0.5 dB window at epoch 5, and the predictor's rank
correlation against the outcome came out at rho = -0.416. It ranked them backwards, and the
cause was the selection, not the predictor.

So models are taken at EVEN INTERVALS THROUGH THE SORTED RANK ORDER, not from the top. That
guarantees maximum spread in the independent variable and even coverage of it, which is what
a correlation estimate needs.

SPEARMAN, NOT PEARSON, IS THE TARGET STATISTIC. The corpus contains models that trained
cleanly to nothing (one sits at 9.65 dB, flat across all 50 epochs). Under Pearson a single
such point manufactures a large correlation on its own. Rank correlation is unaffected, and
the selection deliberately keeps a few low performers rather than trimming to the "working"
band -- excluding them would itself be a range restriction.

  python crossdata_select.py --n 40 --out crossdata_subset.json
"""
import argparse
import glob
import json
import os

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"


def awgn_scores():
    """model -> best 50-epoch AWGN dB, from ft50 and beatbar."""
    out = {}
    for pat in ("fulltrain50/result_shard*.jsonl", "beatbar/result_shard*.jsonl"):
        for f in glob.glob(os.path.join(SWEEP, pat)):
            for line in open(f):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                name = r.get("key") or r.get("model")
                db = r.get("max_dB") or r.get("best_dB")
                if name and db and db > (out.get(name) or 0):
                    out[name] = db
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--db", default="/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/db/ab.nn.db")
    ap.add_argument("--out", default=os.path.join(SWEEP, "crossdata", "subset.json"))
    a = ap.parse_args()

    import sqlite3
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    have_code = {n for (n,) in con.execute("select name from nn")}

    scores = {k: v for k, v in awgn_scores().items() if k in have_code}
    ranked = sorted(scores.items(), key=lambda kv: kv[1])       # ascending
    if len(ranked) < a.n:
        raise SystemExit(f"only {len(ranked)} models have a 50-epoch AWGN score; need {a.n}")

    # even strides through the rank order, endpoints included
    step = (len(ranked) - 1) / (a.n - 1)
    idx = sorted({int(round(i * step)) for i in range(a.n)})
    picked = [ranked[i] for i in idx]

    def fam(n):
        for p in ("LLM_", "DenoiseGA-", "DnSeedGA-", "DenoiseSkip-"):
            if n.startswith(p):
                return p
        return "other"

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = [{"model": m, "awgn_dB": round(d, 3), "family": fam(m)} for m, d in picked]
    json.dump({"n_pool": len(ranked), "n_picked": len(rows),
               "awgn_min": rows[0]["awgn_dB"], "awgn_max": rows[-1]["awgn_dB"],
               "models": rows}, open(a.out, "w"), indent=1)

    import collections
    print(f"pool {len(ranked)} models with a 50-epoch AWGN score -> picked {len(rows)}")
    print(f"AWGN span {rows[0]['awgn_dB']:.2f} .. {rows[-1]['awgn_dB']:.2f} dB "
          f"({rows[-1]['awgn_dB']-rows[0]['awgn_dB']:.2f} dB of spread)")
    print(f"families: {dict(collections.Counter(r['family'] for r in rows))}")
    print(f"\n{'awgn dB':>9}  model")
    for r in rows:
        print(f"{r['awgn_dB']:9.3f}  {r['model'][:52]}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
