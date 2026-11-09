"""Does a noisier benchmark separate architectures better? Same models, both benchmarks.

WHY THE awgn25 GA RUN DID NOT ANSWER THIS. It generated 111 of its own models and scored them
only on denoise_awgn, while the 1,464 models with MAI scores are almost entirely different
models -- the overlap was ONE model. Comparing a 4.80 dB spread on one population to 1.42 dB
on another confounds the benchmark with the population, and the awgn population was also far
less converged (the run died at generation 4 of 30), which inflates spread on its own.

THE CONTROL IS THE WHOLE POINT: one fixed set of architectures, trained on each benchmark
under the SAME budget, compared model-for-model. Both corpora already standardise on 5 epochs
(1,614 MAI models and all 111 awgn models sit at epoch 5), so the budget matches without
re-running the MAI side.

Only the noise level differs -- denoise_awgn is the same MAI clean images with synthetic AWGN
and per-index-seeded validation noise. That makes sigma the single independent variable, which
a different dataset (SIDD, CBSD68) would not.

WHAT WOULD MAKE sigma=25 THE BETTER OBJECTIVE, stated before the data arrives so the result
cannot be reinterpreted afterwards:
  1. a wider p05-p95 spread on the SAME models, and
  2. a larger max-minus-median -- headroom ABOVE the pack, not just a longer tail below it, and
  3. rank correlation with MAI well below 1.0, or it is measuring the same thing more loudly.
Failing (2) while passing (1) means the benchmark punishes weak models rather than rewarding
strong ones, which is the wrong direction for a search objective.

  python benchmark_headroom.py --shard 0 --num-shards 4 --n 200 --dataset denoise_awgn
"""
import argparse
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import naming  # noqa: E402  (needs HERE on sys.path first)

SCALE = 48.0                       # LEMUR accuracy -> dB
EPOCH_STD = 5                      # the budget both corpora already use


def pick(db, n, epoch, seed=0):
    """Stratified by MAI score, plus every hand-written model.

    STRATIFIED, NOT RANDOM. 78.8% of the corpus sits within 0.5 dB of the median, so a uniform
    sample is almost entirely median models and would show no spread on either benchmark by
    construction. Even coverage of the score range is what makes the two spreads comparable.

    The hand-written models are forced in because they are the corpus ceiling -- 37.52 dB
    against the GA's 35.35 -- and a benchmark's headroom claim is exactly a claim about what
    happens at the top.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT n.name, n.code, MAX(s.accuracy) a FROM nn n JOIN stat s ON s.nn = n.name "
        "WHERE s.task='img-denoising' AND s.dataset='denoise' AND s.epoch=? "
        "GROUP BY n.name ORDER BY a", (epoch,)).fetchall()
    con.close()

    def hand(nm):
        # naming.family_prefixes expands "LLM-" to cover the legacy "LLM_" rows;
        # without it every pre-rename generated model reads as hand-written and
        # gets force-included in the benchmark set.
        generated = naming.family_prefixes("LLM-") + (
            "g1_", "DenoiseGA-", "DnSeedGA-", "DenoiseSkip-",
            "DnSkip-", "DnNewFam-", "DnSearch-")
        return not (nm.startswith(generated) or naming.arm_of(nm))

    forced = [r for r in rows if hand(r[0])]
    rest = [r for r in rows if not hand(r[0])]
    k = max(1, len(rest) // max(n - len(forced), 1))
    picked = forced + rest[::k][:max(n - len(forced), 0)]
    seen, out = set(), []
    for r in picked:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    # "denoise-awgn" matches ab/nn/loader/denoise-awgn.py. The old "denoise_awgn"
    # spelling is what produced the 111 unparseable stat dirs withdrawn in review.
    ap.add_argument("--dataset", default="denoise-awgn")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--epoch", type=int, default=EPOCH_STD)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default="/a/mm/sweep_out/headroom")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    sel = pick(a.db, a.n, a.epoch)
    # INTERLEAVED, not contiguous: the list is sorted by score, so contiguous blocks would put
    # every weak model in one shard and a lost shard would delete one end of the range.
    mine = sel[a.shard::a.num_shards]
    print(f"=== headroom shard {a.shard}/{a.num_shards}: {len(mine)} of {len(sel)} "
          f"models on '{a.dataset}' @ {a.epoch}ep ===", flush=True)

    import ab.nn.api as api
    prm = {"lr": a.lr, "batch": a.batch, "transform": "denoise",
           "epoch": a.epoch, "num_workers": 4}
    resf = os.path.join(a.out_dir, f"shard{a.shard}.json")
    out = json.load(open(resf)) if os.path.exists(resf) else {}

    t0 = time.time()
    for i, (nm, code, acc) in enumerate(mine, 1):
        if nm in out and out[nm].get("new_db") is not None:
            continue
        rec = {"mai_db": round(acc * SCALE, 3)}
        t1 = time.time()
        try:
            # save_to_db=False: this is a MEASUREMENT of the benchmark, and writing 200 rows
            # of a second dataset into the corpus would silently blend two noise levels into
            # every downstream PSNR summary that does not filter on dataset.
            _, na, _, _ = api.check_nn(code, "img-denoising", a.dataset, "psnr", prm,
                                       save_to_db=False, prefix=f"HR{a.shard}")
            rec["new_db"] = round(na * SCALE, 3)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:140]}"
        rec["sec"] = round(time.time() - t1)
        out[nm] = rec
        json.dump(out, open(resf, "w"), indent=2)
        if i % 5 == 0 or i == len(mine):
            done = sum(1 for v in out.values() if v.get("new_db") is not None)
            print(f"  [{i}/{len(mine)}] ok={done} {time.time()-t0:.0f}s  "
                  f"last {nm[:34]} MAI {rec['mai_db']} -> "
                  f"{rec.get('new_db', rec.get('error','?'))}", flush=True)
    print(f"DONE shard {a.shard} -> {resf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
