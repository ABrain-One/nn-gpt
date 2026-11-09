"""How long would WL-hash + cosine diversity take over LEMUR's 19,397 classification models?

MEASURE, DO NOT EXTRAPOLATE. The denoise corpus embeds at 26.7 models/sec (1175 in 44 s),
which naively puts 19,397 models at ~12 minutes. That extrapolation is worthless, because
cost is per GRAPH NODE and the two corpora are not the same size of model: denoise models
are ~64 parameter tensors and a median 5,050 bytes of source, while the classification set
has a median of 3,740 bytes but a p95 of 24,407 and a max of 178,375 -- a two-order spread
that a single average hides completely. A handful of ViT/ConvNeXt-scale graphs can dominate
the total.

So this samples the REAL distribution and reports the tail, not just the mean.

THE HARDER PROBLEM IS NOT SPEED, IT IS THE BUILD CONTRACT. arch_uid.py hardcodes denoise:
in_shape (1,3,256,256), out_shape (3,), prm with transform='denoise'. Classification models
span 7 datasets with different resolutions and class counts (cifar-10: 16,633 models at 10
classes; places365: 336 at 365). Handing a CIFAR model ImageNet's shapes does not reliably
crash -- it can BUILD, and then produce a confident hash for an architecture that was never
trained. A wrong-but-successful build is worse than a failure, because nothing downstream
can tell it apart from a real one. Shapes are therefore derived per model from its own
dataset, and any model whose dataset is unknown is REFUSED rather than defaulted.

Reports, from a stratified sample across the code-size range:
    models/sec, and the cost split by model size decile
    projected wall time for all 19,397, with the tail weighted properly
    TraceError rate -- the denoise corpus fails 0.85%; classification will be worse
    build failures, separated from trace failures, because they need different fixes

  python embed_benchmark.py --sample 200
"""
import argparse
import collections
import json
import os
import sqlite3
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# (C, H, W, n_classes) per dataset. Refused rather than guessed when absent -- see docstring.
SHAPES = {
    "cifar-10":      (3, 32, 32, 10),
    "cifar-100":     (3, 32, 32, 100),
    "svhn":          (3, 32, 32, 10),
    "mnist":         (1, 28, 28, 10),
    "celeba-gender": (3, 64, 64, 2),
    "imagenette":    (3, 224, 224, 10),
    "places365":     (3, 224, 224, 365),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-gpt/db/ab.nn.db")
    ap.add_argument("--task", default="img-classification")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--total", type=int, default=19397)
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/embed_benchmark.json")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT n.name, n.code, s.dataset FROM nn n JOIN "
        "(SELECT nn, MIN(dataset) dataset FROM stat WHERE task=? GROUP BY nn) s "
        "ON s.nn = n.name", (a.task,)).fetchall()
    con.close()
    print(f"=== embed benchmark | {len(rows)} {a.task} models in the DB ===", flush=True)

    # STRATIFIED BY CODE SIZE. A uniform random sample of 200 from a distribution whose p95
    # is 6x its median will, most of the time, contain no large model at all -- and the large
    # models are exactly what decides the total.
    rows.sort(key=lambda r: len(r[1]))
    step = len(rows) / a.sample
    sample = [rows[min(int(i * step), len(rows) - 1)] for i in range(a.sample)]
    sizes = [len(r[1]) for r in sample]
    print(f"  sampled {len(sample)}, stratified across code size "
          f"({min(sizes)}..{max(sizes)} bytes)", flush=True)

    import torch                                          # noqa: F401
    from arch_embed import embed, exact

    ok, build_fail, trace_fail, timings = 0, [], [], []
    t_all = time.time()
    for i, (name, code, ds) in enumerate(sample, 1):
        shp = SHAPES.get(ds)
        if shp is None:
            build_fail.append((name, f"unknown dataset {ds}"))
            continue
        c, h, w, ncls = shp
        t0 = time.time()
        net = None
        try:
            ns = {}
            exec(compile(code, "<m>", "exec"), ns)
            if "Net" not in ns:
                raise ValueError("no class Net")
            net = ns["Net"]((1, c, h, w), (ncls,),
                            {"lr": 1e-3, "momentum": 0.9, "batch": 16,
                             "dropout": 0.1, "epoch_max": 5}, "cpu")
            net.eval()
        except Exception as e:
            build_fail.append((name, f"{type(e).__name__}: {e}"[:90]))
            continue
        try:
            exact(net)
            embed(net)
            timings.append((len(code), time.time() - t0))
            ok += 1
        except Exception as e:
            trace_fail.append((name, f"{type(e).__name__}: {e}"[:90]))
        finally:
            del net
        if i % 25 == 0 or i == len(sample):
            el = time.time() - t_all
            print(f"  [{i}/{len(sample)}] ok={ok} build_fail={len(build_fail)} "
                  f"trace_fail={len(trace_fail)} {el:.0f}s", flush=True)

    if not timings:
        print("\nNo model embedded successfully -- the build contract is wrong, not slow.")
        for n, e in (build_fail + trace_fail)[:12]:
            print(f"   {n[:44]}: {e}")
        return 1

    per = [t for _s, t in timings]
    total_s = sum(per)
    rate = len(per) / total_s
    print(f"\n=== THROUGHPUT ===")
    print(f"  embedded            {ok}/{len(sample)}")
    print(f"  mean per model      {statistics.mean(per)*1000:.0f} ms")
    print(f"  median              {statistics.median(per)*1000:.0f} ms")
    print(f"  p95                 {sorted(per)[int(len(per)*0.95)]*1000:.0f} ms")
    print(f"  max                 {max(per)*1000:.0f} ms")
    print(f"  rate                {rate:.1f} models/sec")
    print(f"  (denoise corpus measured 26.7 models/sec)")

    # Cost by size decile -- shows whether the tail dominates.
    print(f"\n=== cost by code-size decile ===")
    timings.sort()
    d = max(1, len(timings) // 10)
    print(f"  {'decile':>7}{'bytes':>10}{'ms/model':>10}")
    for k in range(0, len(timings), d):
        chunk = timings[k:k + d]
        if chunk:
            print(f"  {k//d + 1:7d}{statistics.median(s for s, _ in chunk):10.0f}"
                  f"{statistics.mean(t for _s, t in chunk)*1000:10.0f}")

    # PROJECTION weighted by the real size distribution, not the sample mean. The sample is
    # stratified, so its mean per-model cost already reflects the population's size mix.
    est = a.total / rate
    fail_rate = (len(build_fail) + len(trace_fail)) / len(sample)
    print(f"\n=== PROJECTION for {a.total} models ===")
    print(f"  single process          {est/60:6.1f} min  ({est/3600:.2f} h)")
    for n in (4, 8, 16):
        print(f"  {n:2d}-way sharded         {est/60/n:6.1f} min")
    print(f"  expected failures       {fail_rate:.1%} "
          f"({int(fail_rate*a.total)} of {a.total})")
    print(f"    build failures        {len(build_fail)/len(sample):.1%}"
          f"  -- shape/contract problems, FIXABLE")
    print(f"    trace failures        {len(trace_fail)/len(sample):.1%}"
          f"  -- torch.fx limits, need a documented fallback")
    print(f"\n  clustering is O(N x reps) and negligible next to this: at 19k models and "
          f"~10k reps\n  that is ~50 GFLOP, seconds in numpy.")

    if build_fail:
        print(f"\n=== build failures ({len(build_fail)}) ===")
        for k, v in collections.Counter(
                e.split(":")[0] for _n, e in build_fail).most_common(8):
            print(f"  {v:4d}  {k}")
    if trace_fail:
        print(f"\n=== trace failures ({len(trace_fail)}) ===")
        for k, v in collections.Counter(
                e.split(":")[0] for _n, e in trace_fail).most_common(8):
            print(f"  {v:4d}  {k}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"sample": len(sample), "ok": ok, "rate_per_sec": rate,
                   "mean_ms": statistics.mean(per) * 1000,
                   "projected_seconds_total": est, "total_models": a.total,
                   "build_failures": dict(build_fail[:200]),
                   "trace_failures": dict(trace_fail[:200])}, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
