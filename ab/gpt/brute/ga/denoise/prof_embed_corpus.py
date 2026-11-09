"""Embed one CONTIGUOUS SHARD of a LEMUR task under the AST metric. Pairs with prof_cluster.py.

The AST counterpart to embed_corpus.py, over the SAME row set, so the two runs are directly
comparable. Everything that made the fx run expensive is absent here: no model is
instantiated, no forward is run, no shapes are needed, no pretrained weights are downloaded.
Only the module is imported and its source parsed.

That is also why coverage differs. embed_corpus.py lost 4,774 models to torch.fx TraceError
and 12 to build failures -- 24.6% of the zoo with no architectural identity at all. Here the
only way to fail is unparseable or unimportable source, so the population being measured is
NOT the same population. prof_cluster.py therefore reports two slices and never one number:
the full AST coverage, and the intersection with what fx managed, which is the only
apples-to-apples comparison against the 10,479 figure.

SHARDING IS FOR ROBUSTNESS, NOT SPEED. uid_full.py measured 22.7 models/sec on denoise, so
19,397 models is ~15 minutes in one process. But importing arbitrary LEMUR source executes
arbitrary top-level code, and a single model that hangs or OOMs would take the whole run with
it. Contiguous blocks plus --flush-every bound that to one block.

IMPORTS ARE EVICTED AFTER EACH MODEL. load_class registers the module in sys.modules and
leaves it there; 19k live modules, each holding class objects that reference torch, is a leak
big enough to matter. The module is dropped and its temp file unlinked once the graph is
built -- the graph itself holds no reference to the class.

  python prof_embed_corpus.py --shard 0 --num-shards 8 --task img-classification --out DIR
"""
import argparse
import gc
import json
import os
import signal
import sqlite3
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/incoming"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


class _Timeout:
    """Per-model wall-clock ceiling.

    WALL CLOCK, NOT RLIMIT_CPU, and the difference is the point: the failure mode here is a
    module that BLOCKS -- a top-level download against a dead mirror, a lock -- which burns no
    CPU and so would never trip a CPU limit. preflight.py wants the opposite (reproducible
    accounting of compute) and correctly uses RLIMIT_CPU there.
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
    ap.add_argument("--db", default="/a/mm/nn-gpt/db/ab.nn.db")
    ap.add_argument("--task", default="img-classification")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=60,
                    help="per-model ceiling on import+parse")
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # THE SAME QUERY AS embed_corpus.py, deliberately. It selects on `stat` so the row set is
    # identical -- a different WHERE would change the denominator and quietly invalidate every
    # comparison this run exists to make. `dataset` is selected and ignored: unused by the AST
    # metric, kept so the two row lists are provably the same object.
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT n.name, n.code, s.dataset FROM nn n JOIN "
        "(SELECT nn, MIN(dataset) dataset FROM stat WHERE task=? GROUP BY nn) s "
        "ON s.nn = n.name ORDER BY n.name", (a.task,)).fetchall()
    con.close()
    if a.limit:
        rows = rows[:a.limit]

    lo = a.shard * len(rows) // a.num_shards
    hi = (a.shard + 1) * len(rows) // a.num_shards
    mine = rows[lo:hi]
    print(f"=== AST embed shard {a.shard}/{a.num_shards} | rows [{lo}:{hi}] "
          f"= {len(mine)} of {len(rows)} ===", flush=True)

    import numpy as np
    import ast_model_graph as P
    from prof.uid_full import load_class, source_graph_ext

    tmp = tempfile.mkdtemp(prefix=f"astuid{a.shard}_")
    names, hashes, vecs = [], [], []
    failed = {}
    t0 = time.time()

    def flush(final=False):
        widths = sorted({v.shape[0] for v in vecs})
        np.savez_compressed(
            os.path.join(a.out, f"shard{a.shard}.npz"),
            names=np.array(names, dtype=object),
            hashes=np.array(hashes, dtype=object),
            vecs=(np.stack(vecs) if vecs and len(widths) == 1
                  else np.zeros((0, 0), dtype=np.float32)),
            ragged=np.array([] if len(widths) <= 1 else widths))
        with open(os.path.join(a.out, f"fail{a.shard}.json"), "w") as f:
            # `build` is left empty and `trace` carries everything: prof_cluster.py and
            # cluster_corpus.py both read this shape, and the AST metric has exactly one
            # failure class, so splitting it two ways would invent a distinction.
            json.dump({"build": {}, "trace": failed, "widths": widths,
                       "elapsed_s": time.time() - t0, "complete": final,
                       "done": len(names) + len(failed), "assigned": len(mine)},
                      f, indent=2)

    def _safe(nm):
        """load_class's own sanitisation, mirrored so eviction targets the exact key.

        Recomputed rather than guessed: matching sys.modules by prefix would evict a
        DIFFERENT model's module whenever two names share a truncated stem, and the index
        prefix makes that likely rather than rare.
        """
        return "m_" + "".join(c if c.isalnum() else "_" for c in nm)[:60]

    for i, (name, code, _ds) in enumerate(mine, 1):
        # Index first: names are truncated to 60 chars, and LEMUR has families whose names
        # agree well past that, so the index is what keeps the module key unique.
        modname = f"s{a.shard}i{i}_{name}"
        key = _safe(modname)
        try:
            with _Timeout(a.timeout):
                K = load_class(tmp, modname, code)
                g = source_graph_ext(K)
                if not g.keys:
                    # Every empty graph hashes identically, so admitting one would fabricate
                    # a duplicate group rather than record a miss.
                    raise ValueError("empty graph (source not recoverable)")
                h = P.wl_hash(g)
                v = np.asarray(P.wl_embed(*P.adjacency(g), P.featurize(g)),
                               dtype=np.float32).ravel()
            n = np.linalg.norm(v)
            if n == 0 or not np.isfinite(n):
                raise ValueError("degenerate embedding")
            names.append(name)
            hashes.append(h)
            vecs.append(v / n)
        except BaseException as e:
            # BaseException, not Exception: the alarm fires wherever the interpreter happens
            # to be, and arbitrary imported code may raise SystemExit. Neither should end the
            # shard.
            if isinstance(e, KeyboardInterrupt):
                raise
            failed[name] = f"{type(e).__name__}: {e}"[:120]
        finally:
            sys.modules.pop(key, None)
            p = os.path.join(tmp, key + ".py")
            if os.path.exists(p):
                os.unlink(p)
        if i % a.flush_every == 0:
            flush()
            gc.collect()
        if i % 200 == 0 or i == len(mine):
            el = time.time() - t0
            print(f"  [{i}/{len(mine)}] ok={len(names)} failed={len(failed)} "
                  f"{el:.0f}s ({i/max(el,1e-9):.1f}/s)", flush=True)

    flush(final=True)
    print(f"\n  hashed {len(names)}  failed {len(failed)}  "
          f"in {time.time()-t0:.0f}s")
    print(f"  wrote {os.path.join(a.out, f'shard{a.shard}.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
