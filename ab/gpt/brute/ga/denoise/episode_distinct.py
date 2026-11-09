"""Recompute distinctness for EVERY generation episode, offline. Fixes the label, not a nicety.

WHY THE STORED LABEL IS NOT ENOUGH. `ok=True` in preflight.json means "passed every stage
including the diversity gate", and the gate's documented contract is that an UNTRACEABLE
candidate is admitted -- "no architectural opinion, never reject". Measured on our own data:
301 of the 705 desirable episodes (42.7%) carry no arch_uid, so their distinctness was never
actually tested. They are "valid, distinctness unknown", and training on them as though they
were "valid AND distinct" teaches the model that untraceable output is desirable.

THE FIX IS AVAILABLE BECAUSE THE AST HASH NEEDS NOTHING TO RUN. It parses source; it does not
build or trace, so it covered 19,397 of 19,397 LEMUR zoo models where torch.fx managed 75.3%.
Every episode has candidate.py on disk, including the 3,406 that failed at build or forward.

WHAT IS EMITTED PER EPISODE:
    ast_hash      exact identity over every method (None only if the source will not parse)
    fx_cos        graded cosine to the nearest corpus model, when the model traces
    novel_exact   ast_hash unseen in the corpus AND unseen earlier in this pass
    novel_graded  novel_exact AND fx_cos < --cos-max, when fx_cos exists

REFERENCE SET IS THE CORPUS, NOT THE BATCH. A candidate that re-enacts a model already in the
dataset is a duplicate no matter what else was generated that day, so master is loaded first
and episodes are then walked in generation order -- the same greedy semantics the live gate
uses, so the recomputed label is comparable to the stored one rather than a different measure.
"""
import argparse
import collections
import signal
import glob
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/incoming"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


class _Timeout:
    """Per-candidate wall-clock ceiling.

    OMITTING THIS COST A RUN. load_class imports the candidate module, which executes its
    top-level code, and the fx trace can loop on a pathological graph -- neither burns enough
    CPU to trip a CPU limit, and neither raises. The first attempt at this script stalled at
    episode 750 of 5,875 and sat there for sixteen minutes looking identical to slow progress.
    """

    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._fire)
        signal.alarm(self.seconds)
        return self

    def _fire(self, *_):
        raise TimeoutError(f"exceeded {self.seconds}s")

    def __exit__(self, *exc):
        signal.alarm(0)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/a/mm/sweep_out")
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--cos-max", type=float, default=0.95)
    ap.add_argument("--timeout", type=int, default=45,
                    help="per-candidate ceiling on import + fingerprint")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default="/a/mm/sweep_out/manifest/episode_distinct")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    recs = []
    for p in sorted(glob.glob(os.path.join(a.root, "llm_evolve*/gen*/pre/preflight.json"))):
        root = p.rsplit("/pre/", 1)[0]
        try:
            d = json.load(open(p))
        except Exception:
            continue
        # KEY BY RUN/GEN/ID, NEVER BY ID ALONE. Candidate ids are `g<gen>_<idx>_<strategy>`
        # and every run restarts its generation numbering, so `g1_0000_seeded_constraint_based`
        # exists in par, par2, par3, par4 and par5. Measured: 6,012 records collapse to 2,545
        # distinct ids -- a 57.7% silent loss for any dict keyed on id, with nothing raised.
        rel = os.path.relpath(root, a.root).replace(os.sep, "/")
        for x in d:
            f = os.path.join(root, "run", "cand", x.get("id", ""), "candidate.py")
            if os.path.exists(f):
                recs.append((f"{rel}/{x.get('id')}", f, str(x.get("ok")) == "True",
                             x.get("stage"), x.get("arch_uid")))
    # Sorted by id so the greedy walk is deterministic and reproducible across shards.
    recs.sort(key=lambda r: r[0] or "")
    print(f"=== {len(recs)} episodes with source on disk ===", flush=True)

    from diversity_gate import DiversityGate
    gate = DiversityGate(store_path=None, graded=True, cos_max=a.cos_max,
                         ast_exact=True, log=lambda *_: None)

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    corpus = con.execute(
        "SELECT name, code FROM nn WHERE name IN "
        "(SELECT DISTINCT nn FROM stat WHERE task='img-denoising') ORDER BY name").fetchall()
    con.close()
    t0 = time.time()
    for i, (nm, code) in enumerate(corpus, 1):
        try:
            with _Timeout(a.timeout):
                _, fp = gate.check(code)
            gate.add(nm, fp)
        except BaseException as ex:
            if isinstance(ex, KeyboardInterrupt):
                raise
        if i % 400 == 0:
            print(f"  seeded {i}/{len(corpus)} {time.time()-t0:.0f}s", flush=True)
    print(f"  reference: {len(gate.accepted)} hashes, {len(gate.embeds)} vectors "
          f"({time.time()-t0:.0f}s)\n", flush=True)

    lo = a.shard * len(recs) // a.num_shards
    hi = (a.shard + 1) * len(recs) // a.num_shards
    mine = recs[lo:hi]
    out = {}
    stats = collections.Counter()
    t1 = time.time()
    for i, (cid, f, ok, stage, old_uid) in enumerate(mine, 1):
        code = open(f).read()
        rec = {"ok": ok, "stage": stage, "had_arch_uid": old_uid not in (None, "None")}
        try:
            with _Timeout(a.timeout):
                h, e = gate._fingerprint(code)
        except BaseException as ex:
            # BaseException: the alarm fires wherever the interpreter happens to be, and
            # imported candidate code may raise SystemExit. Neither should end the pass.
            if isinstance(ex, KeyboardInterrupt):
                raise
            h, e = None, None
            rec["error"] = f"{type(ex).__name__}"
        rec["ast_hash"] = h
        if h is None:
            stats["unhashable"] += 1
            rec["novel_exact"] = None
        else:
            rec["novel_exact"] = h not in gate.accepted
            if rec["novel_exact"]:
                stats["novel_exact"] += 1
            else:
                stats["exact_dup"] += 1
        if e is not None and gate.embeds:
            import numpy as np
            sims = [float(np.dot(e, ae)) for ae in gate.embeds]
            rec["fx_cos"] = round(max(sims), 4)
            rec["novel_graded"] = bool(rec.get("novel_exact")) and \
                max(sims) < a.cos_max
        else:
            rec["fx_cos"] = None
            rec["novel_graded"] = rec.get("novel_exact")
        # GREEDY, like the live gate: an accepted episode joins the reference set so a later
        # episode that merely repeats it is correctly called a duplicate.
        if rec.get("novel_exact") and h is not None:
            gate.add(cid, (h, e))
        out[cid] = rec
        if i % 250 == 0 or i == len(mine):
            print(f"  [{i}/{len(mine)}] {time.time()-t1:.0f}s  "
                  f"novel={stats['novel_exact']} dup={stats['exact_dup']} "
                  f"unhashable={stats['unhashable']}", flush=True)
            json.dump(out, open(os.path.join(a.out_dir, f"shard{a.shard}.json"), "w"))

    json.dump(out, open(os.path.join(a.out_dir, f"shard{a.shard}.json"), "w"), indent=1)

    # The number this exists to produce: how the recomputed label differs from the stored one
    # on the positives, which is the set training would consume.
    des = [r for r in out.values() if r["ok"]]
    unver = [r for r in des if not r["had_arch_uid"]]
    print(f"\n=== POSITIVES ({len(des)}) ===")
    print(f"  previously unverified (no arch_uid) {len(unver)}")
    print(f"  now hashable                        "
          f"{sum(1 for r in unver if r['ast_hash'])}/{len(unver)}")
    print(f"  of those, actually NOVEL            "
          f"{sum(1 for r in unver if r.get('novel_exact'))}")
    print(f"  of those, DUPLICATES (mislabelled)  "
          f"{sum(1 for r in unver if r.get('novel_exact') is False)}")
    print(f"\nwrote {a.out_dir}/shard{a.shard}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
