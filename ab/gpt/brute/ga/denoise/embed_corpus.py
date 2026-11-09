"""Embed one CONTIGUOUS SHARD of a LEMUR task's models. Pairs with cluster_corpus.py.

Splits the work that embed_benchmark.py measured at 4.4 models/sec, i.e. ~74 minutes for
19,397 classification models in one process. Sixteen shards bring that to ~5 minutes, and it
is pure CPU torch.fx tracing, so it consumes no GPU quota.

CONTIGUOUS BLOCKS, NOT `i % num_shards`. The interleave aliases with any periodic structure
in the row order -- in the LLM generator the same mistake put every candidate of one strategy
in a single shard, so losing one shard deleted an entire experimental arm. Blocks make a lost
shard a lost contiguous range, which is obvious and re-runnable.

TORCH_HOME MUST BE SHARED STORAGE. LEMUR classification models call torchvision constructors
that download pretrained weights on instantiation -- 5 MB to 109 MB each. The container's
default cache is /home/ab/.cache/torch, which is ephemeral, so sixteen shards would each
re-download the same weights and sharding would make the job SLOWER, not faster. The weights
are pure waste to us -- we hash the GRAPH and never look at a parameter -- but suppressing
them needs a per-model source change, so caching is the fix available from outside.

SHAPES ARE PER-DATASET AND REFUSED WHEN UNKNOWN. Feeding a CIFAR model ImageNet's shapes does
not reliably crash; it can BUILD, and then emit a confident hash for an architecture nobody
trained. A wrong-but-successful build is indistinguishable downstream from a real one, so an
unknown dataset is an error, never a default.

  python embed_corpus.py --shard 0 --num-shards 16 --task img-classification --out DIR
"""
import argparse
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

SHAPES = {
    "cifar-10":      (3, 32, 32, 10),
    "cifar-100":     (3, 32, 32, 100),
    "svhn":          (3, 32, 32, 10),
    "mnist":         (1, 28, 28, 10),
    "celeba-gender": (3, 64, 64, 2),
    "imagenette":    (3, 224, 224, 10),
    "places365":     (3, 224, 224, 365),
}
PRM = {"lr": 1e-3, "momentum": 0.9, "batch": 16, "dropout": 0.1, "epoch_max": 5}

# THE TRACE BATCH IS AT LEAST 2, DELIBERATELY.
#
# 975 of the 1,271 build failures were one error: "Expected more than 1 value per channel
# when training, got input size [1, C, 1, 1]". That is BatchNorm refusing a batch of 1 while
# the module is in train() mode. It fires during __init__, before net.eval() can run, because
# a very common LEMUR pattern is a dummy forward inside the constructor to infer the
# classifier's flattened input width. At batch 1 every such model dies.
#
# Batch size does not change the computation GRAPH -- it is a leading dimension, not
# topology -- so tracing at 2 is faithful to the architecture and merely satisfies BatchNorm.
MIN_TRACE_BATCH = 2


class _bn_tolerant:
    """Let BatchNorm survive the batch-1 dummy forward that many constructors do.

    RAISING THE TRACE BATCH DID NOT FIX THIS, and the error message says why: after the
    change the shape is still torch.Size([1, 512, 1, 1]). These models do not use the
    in_shape we pass for their shape-inference pass -- they build their OWN probe tensor with
    a hardcoded batch of 1:

        dummy = torch.zeros(1, C, H, W)
        self.fc = nn.Linear(self.features(dummy).numel(), n_classes)

    so nothing an external caller passes can reach it. 866 models die this way, all with
    [1, C, 1, 1]: a globally-pooled feature map entering BatchNorm while the module is still
    in train() mode, before net.eval() is reachable.

    F.batch_norm's guard is a single helper, _verify_batch_size, so suppressing exactly that
    is narrower than patching BatchNorm itself. It is also SAFE FOR OUR PURPOSE in a way
    worth stating: the constructor uses only .numel()/.shape from that dummy forward, and
    BatchNorm does not change shape, so the garbage statistics it computes at batch 1 are
    never read. Nor does it touch the traced graph -- topology comes from module structure,
    and ShapeProp later runs under net.eval(), where this helper is not called at all.
    """

    def __enter__(self):
        import torch.nn.functional as F
        self._orig = getattr(F, "_verify_batch_size", None)
        if self._orig is not None:
            F._verify_batch_size = lambda size: None
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            import torch.nn.functional as F
            F._verify_batch_size = self._orig
        return False


def _coerce(v):
    """prm values arrive from SQLite as text. A model doing `int(prm['patch_size'])` is
    fine either way, but one doing `prm['dropout'] * 0.5` needs a number. Coerce what
    parses, leave the rest as the string it is."""
    if not isinstance(v, str):
        return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-gpt/db/ab.nn.db")
    ap.add_argument("--task", default="img-classification")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prm-map", default="/a/mm/sweep_out/manifest/prm_map.json",
                    help="per-model REAL hyperparameters from build_prm_map.py. Without it "
                         "every model gets an invented prm dict and any model reading a key "
                         "outside it dies with KeyError -- 285 of them did.")
    ap.add_argument("--flush-every", type=int, default=50,
                    help="persist partial results this often; an OOM cannot be caught, so "
                         "this bounds how much a killed shard loses")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
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
    print(f"=== embed shard {a.shard}/{a.num_shards} | rows [{lo}:{hi}] "
          f"= {len(mine)} of {len(rows)} ===", flush=True)
    print(f"  TORCH_HOME={os.environ.get('TORCH_HOME', '<unset -- WILL RE-DOWNLOAD>')}",
          flush=True)

    prm_map = {}
    if a.prm_map and os.path.exists(a.prm_map):
        with open(a.prm_map) as f:
            prm_map = json.load(f)
        print(f"  real hyperparameters for {len(prm_map)} models from {a.prm_map}",
              flush=True)
    else:
        print(f"  WARNING no prm map at {a.prm_map} -- falling back to an invented prm; "
              f"expect KeyError on models reading keys outside it", flush=True)

    import numpy as np
    import torch                                       # noqa: F401
    from arch_embed import embed, exact

    names, hashes, vecs = [], [], []
    build_fail, trace_fail = {}, {}
    t0 = time.time()

    def flush(final=False):
        """Persist what we have so far.

        AN OOM CANNOT BE CAUGHT. The kernel kills the process, so a shard that writes only at
        the end loses everything it had computed -- shard 11 of the first 32-way run died at
        model 300 of 606 and its 277 successful embeddings went with it. Some classification
        models genuinely need more memory than a batch-1 64x64 forward suggests, so the
        possibility cannot be designed away; the loss can be. Flushing every --flush-every
        models bounds the damage to that interval, and the partial file is a valid input to
        cluster_corpus.py, which merges whatever shards it finds.
        """
        widths = sorted({v.shape[0] for v in vecs})
        np.savez_compressed(
            os.path.join(a.out, f"shard{a.shard}.npz"),
            names=np.array(names, dtype=object),
            hashes=np.array(hashes, dtype=object),
            vecs=(np.stack(vecs) if vecs and len(widths) == 1
                  else np.zeros((0, 0), dtype=np.float32)),
            ragged=np.array([] if len(widths) <= 1 else widths))
        with open(os.path.join(a.out, f"fail{a.shard}.json"), "w") as f:
            json.dump({"build": build_fail, "trace": trace_fail, "widths": widths,
                       "elapsed_s": time.time() - t0, "complete": final,
                       "done": len(names) + len(build_fail) + len(trace_fail),
                       "assigned": len(mine)}, f, indent=2)
    for i, (name, code, ds) in enumerate(mine, 1):
        shp = SHAPES.get(ds)
        if shp is None:
            build_fail[name] = f"unknown dataset {ds}"
            continue
        c, h, w, ncls = shp
        net = None
        try:
            ns = {}
            exec(compile(code, "<m>", "exec"), ns)
            if "Net" not in ns:
                raise ValueError("no class Net")
            # Real prm where we have it, the invented one only as a floor for keys the
            # model's own record happens to lack (lr/momentum never change the graph).
            p = dict(PRM)
            p.update({k: _coerce(v) for k, v in prm_map.get(name, {}).items()})
            bs = max(MIN_TRACE_BATCH, int(p.get("batch", MIN_TRACE_BATCH) or 2))
            with _bn_tolerant():
                net = ns["Net"]((bs, c, h, w), (ncls,), p, "cpu")
            net.eval()
        except Exception as e:
            build_fail[name] = f"{type(e).__name__}: {e}"[:120]
            continue
        try:
            hh = exact(net)
            vv = np.asarray(embed(net), dtype=np.float32).ravel()
            n = np.linalg.norm(vv)
            if n == 0 or not np.isfinite(n):
                raise ValueError("degenerate embedding")
            names.append(name)
            hashes.append(hh)
            vecs.append(vv / n)
        except Exception as e:
            trace_fail[name] = f"{type(e).__name__}: {e}"[:120]
        finally:
            del net
        if i % a.flush_every == 0:
            flush()
        if i % 100 == 0 or i == len(mine):
            el = time.time() - t0
            print(f"  [{i}/{len(mine)}] ok={len(names)} build_fail={len(build_fail)} "
                  f"trace_fail={len(trace_fail)} {el:.0f}s "
                  f"({i/max(el,1e-9):.1f}/s)", flush=True)

    flush(final=True)
    widths = sorted({v.shape[0] for v in vecs})
    print(f"\n  embedded {len(names)}  build_fail {len(build_fail)}  "
          f"trace_fail {len(trace_fail)}  in {time.time()-t0:.0f}s")
    if len(widths) > 1:
        # Ragged widths make cosines between vectors meaningless. Reported, not truncated.
        print(f"  WARNING ragged embedding widths {widths}: vectors NOT saved for this shard")
    print(f"  wrote {os.path.join(a.out, f'shard{a.shard}.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
