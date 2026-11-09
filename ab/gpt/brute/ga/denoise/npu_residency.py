"""What actually pushes a converted model OFF the NPU, and onto the CPU.

THE QUESTION, AND WHY IT IS NOT THE ONE npu_gate ALREADY ANSWERED. npu_gate.py has two findings
and neither covers this. Its ban rules predict CONVERTIBILITY -- whether a TFLite graph can be
produced at all -- which affects 97 of 1,582 corpus models (6.1%). Its operator table was tested
against LATENCY (>10 ms) and scored 36% recall, correctly rejected as a gate. But the dominant LLM
failure is a third thing: the model converts fine, gets profiled, and then executes on the CPU.
On the S26 Ultra, 37% of 321 profiled LLM models run entirely off the NPU at a median 42.36 ms,
against 2% and 1.52 ms for DnSeedGA. Every one of those 84 recorded fallbacks names CPU.

Nobody has looked at `npu_time_frac` as a label. That is what this does.

THE TRAP THIS IS BUILT TO AVOID -- FAMILY CONFOUNDING. Pooled across families, any operator that
happens to be an LLM idiom will "predict" CPU fallback beautifully, because LLM models fall back
far more often than GA models do. The rule would then be a family detector wearing an operator's
clothes, and it would fail the moment it met a GA model that used the same op. This is the same
class of error as the 12/12-perfect operator rule that collapsed to 36% recall when re-tested
against the label that mattered.

So the primary analysis is WITHIN FAMILY, and the LLM family is where it can be done: 118 fully
off, 119 fully on, 84 partial -- balanced enough to compare directly. A candidate rule is then
re-tested on DenoiseGA as a HELD-OUT family. An operator that separates off from on inside LLM and
also inside DenoiseGA is a plausible cause. One that works only inside LLM is an idiom.

PARTIALS ARE EXCLUDED FROM THE PRIMARY CONTRAST, deliberately. A model at npu_time_frac 0.57 is
neither case and mixing it into either arm blurs the very contrast being measured; there are 84 of
them in LLM and they get their own reported line so their exclusion is visible rather than quiet.

WHAT A RESULT HERE IS WORTH. If a small set of operators separates off from on within both
families, it belongs in the prompt as concrete guidance and in the gate as a WARNING (never a hard
ban -- the ban bar is zero false positives against a definite outcome, and residency is
probabilistic). If nothing separates, that is also an answer: the fallback is driven by something
the source text does not expose, and the only route left is measurement per candidate, which is
what npu_gate's --profile path already does.

  python npu_residency.py --report
  python npu_residency.py --report --min-n 8      # require an op in >=8 models per class
"""
import argparse
import collections
import json
import math
import os
import sqlite3
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from npu_gate import compute_ops  # noqa: E402  AST-scoped: __init__ + forward, no train-only methods

THESIS = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
SWEEP = os.path.join(THESIS, "sweep_out")
DEVICE = "Samsung Galaxy S26 Ultra"
OFF, FULL = 0.001, 0.999


def _borrow(names=("wilson", "newcombe")):
    """Reuse ablation_prompt.py's interval maths; see ablation_prompt_deploy.py for why the import
    has to be a prefix-exec rather than a plain import (that module runs a report at import time)."""
    src = open(os.path.join(_HERE, "ablation_prompt.py")).read()
    marker = "# ---- load"
    ns = {"__name__": "_ap_helpers"}
    exec(compile(src[:src.index(marker)], "ablation_prompt.py(prefix)", "exec"), ns)
    return tuple(ns[n] for n in names)


wilson, newcombe = _borrow()


def family(name):
    if name.startswith("LLM"):
        return "LLM"
    if name.startswith("DnSeedGA"):
        return "DnSeedGA"
    if name.startswith("DenoiseGA"):
        return "DenoiseGA"
    return "hand"


def _aliases(n):
    return {n, n.replace("_", "-"), n.replace("-", "_")}


def load():
    """[{name, family, code, frac, ms, n_layers}] for every model with BOTH a source and a
    per-layer compute-unit breakdown on one device."""
    codes = {}
    p = os.path.join(SWEEP, "tflite_pool", "codes.jsonl")
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                codes[r["name"]] = r["code"]
    try:
        import llm_denoise as L
        con = sqlite3.connect(f"file:{L.db_path()}?mode=ro", uri=True)
        for n, c in con.execute("SELECT name, code FROM nn WHERE code IS NOT NULL"):
            codes.setdefault(n, c)
        con.close()
    except Exception as e:
        print(f"  (DB sources unavailable: {type(e).__name__}: {e})")

    lp = os.path.join(SWEEP, "aihub", "latency_corpus.json")
    lat = json.load(open(lp)) if os.path.exists(lp) else {}
    rows = []
    for n, v in lat.items():
        if not (isinstance(v, dict) and v.get("status") == "OK" and v.get("ms")):
            continue
        if v.get("device") and v["device"] != DEVICE:
            continue
        if v.get("npu_time_frac") is None:
            continue
        hit = next((a for a in _aliases(n) if a in codes), None)
        if hit is None:
            continue
        rows.append({"name": n, "family": family(n), "code": codes[hit],
                     "frac": float(v["npu_time_frac"]), "ms": float(v["ms"]),
                     "n_layers": v.get("n_layers")})
    return rows


def contrast(rows, min_n):
    """Per-operator off-vs-full comparison inside one family."""
    off = [r for r in rows if r["frac"] <= OFF]
    full = [r for r in rows if r["frac"] >= FULL]
    if not off or not full:
        return [], len(off), len(full)
    ops_off = [compute_ops(r["code"]) for r in off]
    ops_full = [compute_ops(r["code"]) for r in full]
    allops = {o for s in ops_off + ops_full for o in s}
    out = []
    for op in allops:
        k1 = sum(1 for s in ops_off if op in s)
        k2 = sum(1 for s in ops_full if op in s)
        if k1 + k2 < min_n:
            continue
        d, lo, hi = newcombe(k1, len(off), k2, len(full))
        out.append({"op": op, "k_off": k1, "n_off": len(off), "k_full": k2, "n_full": len(full),
                    "diff": d, "lo": lo, "hi": hi, "resolved": (lo > 0 or hi < 0)})
    out.sort(key=lambda r: -r["diff"])
    return out, len(off), len(full)


def depth_only_separation(rows):
    """How well does n_layers ALONE separate off-NPU from fully-on? (AUC, threshold-free.)

    This has to be measured before any operator is blamed. If depth alone separates the classes
    nearly perfectly, an operator table computed across the whole family is largely re-describing
    depth: a 377-layer graph is more likely to contain any given operator than a 70-layer one.
    """
    off = [r["n_layers"] for r in rows if r["frac"] <= OFF and r["n_layers"]]
    ful = [r["n_layers"] for r in rows if r["frac"] >= FULL and r["n_layers"]]
    if not off or not ful:
        return None
    # AUC = P(random off-model is deeper than a random full-model), ties count half.
    wins = sum(1 for a in off for b in ful if a > b) + 0.5 * sum(1 for a in off for b in ful if a == b)
    return wins / (len(off) * len(ful)), off, ful


def depth_stratified(rows, ops_of_interest, min_cell):
    """Off-vs-full operator contrast INSIDE depth bands, so depth cannot do the separating.

    Bands are quartiles of the pooled n_layers distribution for the family. An operator that still
    separates within a band, where both classes have similar depth, is a candidate cause. One whose
    effect vanishes inside every band was a proxy for depth.
    """
    have = [r for r in rows if r["n_layers"]]
    if len(have) < 20:
        return {}
    depths = sorted(r["n_layers"] for r in have)
    qs = [depths[int(f * (len(depths) - 1))] for f in (0.25, 0.5, 0.75)]

    def band(n):
        return 0 if n <= qs[0] else (1 if n <= qs[1] else (2 if n <= qs[2] else 3))

    out = {}
    for b in range(4):
        rs = [r for r in have if band(r["n_layers"]) == b]
        off = [r for r in rs if r["frac"] <= OFF]
        ful = [r for r in rs if r["frac"] >= FULL]
        if len(off) < min_cell or len(ful) < min_cell:
            out[b] = {"n_off": len(off), "n_full": len(ful), "skipped": True, "ops": {}}
            continue
        oo = [compute_ops(r["code"]) for r in off]
        of = [compute_ops(r["code"]) for r in ful]
        cell = {}
        for op in ops_of_interest:
            k1 = sum(1 for s in oo if op in s)
            k2 = sum(1 for s in of if op in s)
            d, lo, hi = newcombe(k1, len(off), k2, len(ful))
            cell[op] = {"k_off": k1, "n_off": len(off), "k_full": k2, "n_full": len(ful),
                        "diff": d, "lo": lo, "hi": hi, "resolved": (lo > 0 or hi < 0)}
        out[b] = {"n_off": len(off), "n_full": len(ful), "skipped": False, "ops": cell,
                  "depth_range": (min(r["n_layers"] for r in rs), max(r["n_layers"] for r in rs))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--min-cell", type=int, default=5,
                    help="minimum models per class inside a depth band for it to be tested")
    ap.add_argument("--min-n", type=int, default=6,
                    help="minimum models carrying an op across the two classes")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(SWEEP, "aihub", "npu_residency.json"))
    a = ap.parse_args()

    rows = load()
    if not rows:
        sys.exit("no models with both source and npu_time_frac -- run bulk_profile.py --collect/--report")
    byfam = collections.defaultdict(list)
    for r in rows:
        byfam[r["family"]].append(r)

    print(f"{len(rows)} models with source + per-layer compute units on {DEVICE}\n")
    print(f"{'family':<12}{'n':>6}{'off':>6}{'partial':>9}{'full':>6}"
          f"{'p50 ms off':>12}{'p50 ms full':>13}")
    for f, rs in sorted(byfam.items(), key=lambda kv: -len(kv[1])):
        off = sorted(r["ms"] for r in rs if r["frac"] <= OFF)
        ful = sorted(r["ms"] for r in rs if r["frac"] >= FULL)
        par = sum(1 for r in rs if OFF < r["frac"] < FULL)
        print(f"{f:<12}{len(rs):>6}{len(off):>6}{par:>9}{len(ful):>6}"
              f"{(off[len(off)//2] if off else float('nan')):>12.2f}"
              f"{(ful[len(ful)//2] if ful else float('nan')):>13.2f}")
    print("\nThat last pair is the cost of the fallback, and it is the reason this matters:")
    print("running off the NPU is not a small penalty, it is the difference between shipping")
    print("and not shipping.")

    results = {}
    for fam in ("LLM", "DenoiseGA"):
        rs = byfam.get(fam, [])
        res, n_off, n_full = contrast(rs, a.min_n)
        results[fam] = res
        print(f"\n{'='*94}\nWITHIN {fam}: {n_off} off-NPU vs {n_full} fully-on "
              f"({sum(1 for r in rs if OFF < r['frac'] < FULL)} partial excluded)")
        if not res:
            print("  too few models in one class to contrast")
            continue
        print(f"  {'operator':<22}{'in off':>12}{'in full':>12}{'difference':>13}{'  95% CI':>18}")
        for r in res[:a.top]:
            mark = " *" if r["resolved"] else ""
            print(f"  {r['op']:<22}{r['k_off']:>5}/{r['n_off']:<6}{r['k_full']:>5}/{r['n_full']:<6}"
                  f"{r['diff']:>+12.1%}   [{r['lo']:+.1%},{r['hi']:+.1%}]{mark}")
        nres = sum(1 for r in res if r["resolved"])
        print(f"  {nres} of {len(res)} operators resolve (interval excludes 0); * marks them")

    # THE GENERALISATION TEST. An operator is only a candidate cause if it points the same way in a
    # family that was not used to find it. Same sign AND resolved in LLM, same sign in DenoiseGA.
    print(f"\n{'='*94}\nGENERALISATION: operators that point the SAME way in both families")
    llm = {r["op"]: r for r in results.get("LLM", [])}
    ga = {r["op"]: r for r in results.get("DenoiseGA", [])}
    both = []
    for op, r in llm.items():
        if op not in ga or not r["resolved"]:
            continue
        if (r["diff"] > 0) == (ga[op]["diff"] > 0):
            both.append((op, r["diff"], ga[op]["diff"], ga[op]["resolved"]))
    both.sort(key=lambda t: -abs(t[1]))
    if not both:
        print("  NONE. Every operator that separates inside LLM either does not appear often")
        print("  enough in DenoiseGA to test, or points the other way there. That means the")
        print("  separating operators are LLM IDIOMS, not causes of CPU fallback, and no static")
        print("  operator rule should be added to the prompt or the gate on this evidence.")
    else:
        print(f"  {'operator':<22}{'LLM diff':>12}{'GA diff':>12}{'GA resolved':>14}")
        for op, d1, d2, res2 in both:
            print(f"  {op:<22}{d1:>+12.1%}{d2:>+12.1%}{str(res2):>14}")
        print("\n  Positive difference = more common in OFF-NPU models = candidate culprit.")
        print("  These are candidates for PROMPT GUIDANCE and for a gate WARNING. They are not")
        print("  ban material: the ban bar is zero false positives against a definite outcome,")
        print("  and NPU residency is probabilistic, not definite.")

    # Depth, as a non-operator explanation that has to be ruled out before blaming operators.
    print(f"\n{'='*94}\nIS IT JUST DEPTH? n_layers by class, per family")
    depth_auc = {}
    for fam in ("LLM", "DenoiseGA"):
        rs = byfam.get(fam, [])
        got = depth_only_separation(rs)
        if not got:
            continue
        auc, off, ful = got
        depth_auc[fam] = auc
        off, ful = sorted(off), sorted(ful)
        print(f"  {fam:<12} off-NPU median {off[len(off)//2]:>4}  "
              f"fully-on median {ful[len(ful)//2]:>4}  (n={len(off)}/{len(ful)})  "
              f"depth-alone AUC {auc:.3f}")
    print("  AUC is P(a random off-NPU model is deeper than a random fully-on one). 0.5 = depth")
    print("  says nothing; 1.0 = depth separates the classes perfectly on its own.")

    # DEPTH-STRATIFIED CONTRAST -- the test that decides whether the operator table above means
    # anything. Inside a depth band the two classes have comparable graph sizes, so an operator
    # effect that survives there is not depth in disguise.
    print(f"\n{'='*94}\nOPERATORS INSIDE DEPTH BANDS (quartiles of n_layers, per family)")
    strat_out = {}
    for fam in ("LLM", "DenoiseGA"):
        res = results.get(fam, [])
        interest = [r["op"] for r in res if r["resolved"]][:8]
        if not interest:
            continue
        bands = depth_stratified(byfam.get(fam, []), interest, a.min_cell)
        strat_out[fam] = bands
        print(f"\n  --- {fam}: {', '.join(interest)}")
        for b in sorted(bands):
            info = bands[b]
            if info.get("skipped"):
                print(f"    band {b}: n_off={info['n_off']} n_full={info['n_full']} "
                      f"-- too few in one class to test")
                continue
            lo_d, hi_d = info["depth_range"]
            surv = [op for op, c in info["ops"].items() if c["resolved"] and c["diff"] > 0]
            print(f"    band {b} (layers {lo_d}-{hi_d}, {info['n_off']} off / {info['n_full']} full): "
                  f"{len(surv)}/{len(interest)} still resolve -> {', '.join(surv) if surv else 'NONE'}")
    print("\n  An operator listed as surviving in a band separates off-NPU from fully-on among")
    print("  models of SIMILAR DEPTH, which is the only version of this claim worth acting on.")

    json.dump({"device": DEVICE, "n": len(rows),
               "families": {f: len(v) for f, v in byfam.items()},
               "within_family": results, "depth_auc": depth_auc, "depth_stratified": strat_out,
               "generalising": [{"op": o, "llm_diff": d1, "ga_diff": d2} for o, d1, d2, _ in both]},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
