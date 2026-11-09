"""Calibrate a source-only Full HD latency predictor.

At 256x256 the denoise corpus is latency-bimodal: residency on the accelerator decides
almost everything and MACs are near-useless (AUC 0.651). At 1088x1920 every profiled model
is npu_layer_frac=1.0 -- nothing is evicted -- so the compute term should dominate instead.
This script tests that directly: MACs (counted at the dataset's 256x256 dims, which for
conv nets scale linearly to Full HD) against measured Full HD milliseconds.

If the fit is tight, the GA can gate candidates on predicted Full HD latency for free,
in-loop, with no AI Hub round trip.
"""
import json, os, sys, math

T = "/a/mm"
sys.path.insert(0, T + "/nn-gpt/ab/gpt/brute/ast")
sys.path.insert(0, T + "/nn-gpt/ab/gpt/brute/ga/denoise")
from edge_cost import measure_cost

FH = T + "/sweep_out/fullhd"
out = {}
fh = json.load(open(FH + "/fullhd_results.json"))
for fn in sorted(os.listdir(FH + "/src")):
    if not fn.endswith(".py"):
        continue
    name = fn[:-3]
    rec = fh.get(name)
    # fullhd_results keys are model names; a couple of sources are named differently
    if rec is None:
        cand = [k for k in fh if k.split("-")[0] == name.split("-")[0] and name[-12:] in k]
        rec = fh.get(cand[0]) if cand else None
    if not isinstance(rec, dict) or rec.get("status") != "OK":
        print("  no FullHD ms for %s" % name, flush=True)
        continue
    c = measure_cost(open(os.path.join(FH, "src", fn)).read())
    if not c.get("ok"):
        print("  MAC count FAILED for %s" % name, flush=True)
        continue
    out[name] = {"macs": c["macs"], "params": c["params"], "fullhd_ms": rec["ms"]}
    print("  %-44s macs=%10d params=%8d  %7.2f ms" %
          (name[:44], c["macs"], c["params"], rec["ms"]), flush=True)

json.dump(out, open(FH + "/cost_fit.json", "w"), indent=1)

# --- fit and report ---
xs = [v["macs"] / 1e6 for v in out.values()]
ys = [v["fullhd_ms"] for v in out.values()]
n = len(xs)
print("\nn = %d models with both MACs and Full HD latency" % n)
if n >= 3:
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    pr = sxy / math.sqrt(sxx * sum((y - my) ** 2 for y in ys))
    # Spearman
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(s):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    mrx, mry = sum(rx) / n, sum(ry) / n
    sp = (sum((p - mrx) * (q - mry) for p, q in zip(rx, ry)) /
          math.sqrt(sum((p - mrx) ** 2 for p in rx) * sum((q - mry) ** 2 for q in ry)))
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    mae = sum(abs(r) for r in res) / n
    print("fit:  fullhd_ms = %.4f * MMac + %.3f" % (b, a))
    print("Pearson r = %.4f   (r^2 = %.4f)" % (pr, pr * pr))
    print("Spearman rho = %.4f" % sp)
    print("MAE = %.2f ms   max |resid| = %.2f ms" % (mae, max(abs(r) for r in res)))
    print("\nMMac budget for a 46.1 ms Full HD cap:  %.0f MMac" % ((46.1 - a) / b))
    print("MMac budget for a 50.05 ms cap (our instrument): %.0f MMac" % ((50.05 - a) / b))
    print("\n%-44s %9s %9s %8s" % ("model", "MMac", "meas ms", "resid"))
    for (k, v), r in zip(out.items(), res):
        print("%-44s %9.1f %9.2f %+8.2f" % (k[:44], v["macs"] / 1e6, v["fullhd_ms"], r))
    json.dump({"slope_ms_per_MMac": b, "intercept_ms": a, "pearson": pr, "spearman": sp,
               "mae_ms": mae, "n": n,
               "mmac_cap_46_1ms": (46.1 - a) / b, "mmac_cap_50_05ms": (50.05 - a) / b},
              open(FH + "/cost_model.json", "w"), indent=1)
    print("\nwrote cost_model.json")
