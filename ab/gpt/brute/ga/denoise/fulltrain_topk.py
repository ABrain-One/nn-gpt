"""Gate 1 — full-train the top-K per ablation arm at 50 epochs (the PROMOTION AUTHORITY).

The ep5 proxy only SCREENS; it cannot crown (10F/7F late-bloomer blind spot). This
driver takes each arm's cached candidates, picks the top-K by edge-aware eff-dB
(plus the best deployable-within-budget one, since that's the thesis edge headline),
and trains each at epoch 50 to get the real dB. Output = the authoritative table
that confirms whether the proxy ranking D>C>B>A holds, to bring the professor
BEFORE committing the production run.

Resumable: results are cached by checksum (fulltrain_results.json), so a re-run
after preemption trains only what's missing, and an architecture shared across
arms trains once. save_to_db OFF. Run via launch_fulltrain.sh.
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ast")

TASK, DATASET, METRIC = "img-denoising", "denoise", "psnr"
PSNR_SCALE = 48.0
BUDGET_MMAC = 9052.61  # median seed MACs (9F) — same auto-budget the ablation used


def _select(arm, k, log):
    """Return an ordered list of (checksum, rec) to full-train for this arm:
    top-K by eff-dB (distinct checksums) + the best deployable within-budget one."""
    from edge_deploy import deployability
    cache_path = os.path.join(_HERE, f"seed_ga_{arm}_cache.json")
    mdir = os.path.join(_HERE, f"seed_ga_{arm}_models")
    if not os.path.exists(cache_path):
        log(f"  [{arm}] no cache — skipped")
        return []
    d = json.load(open(cache_path))
    recs = [(kk, v) for kk, v in d.items()
            if isinstance(v, dict) and v.get("psnr", 0) and v["psnr"] > 0
            and os.path.exists(os.path.join(mdir, kk + ".py"))]
    recs.sort(key=lambda t: t[1].get("eff_db", 0.0), reverse=True)

    chosen, seen = [], set()
    for kk, v in recs[:k]:
        chosen.append((kk, v)); seen.add(kk)

    # best deployable within-budget (the edge headline) — add if not already chosen
    for kk, v in recs:
        if kk in seen:
            continue
        macs_m = (v.get("macs") or 0) / 1e6
        if macs_m <= 0 or macs_m > BUDGET_MMAC:
            continue
        src = open(os.path.join(mdir, kk + ".py")).read()
        if deployability(src)["deployable"]:
            v = dict(v); v["_edge_pick"] = True
            chosen.append((kk, v))
            log(f"  [{arm}] + edge pick {v.get('name')} ({macs_m:.0f} MMac)")
            break
    return chosen


def _train(arm, key, rec, epoch, batch, lr, workers, log):
    """Full-train one candidate at `epoch`; return the measured result dict."""
    import ab.nn.api as api
    from edge_deploy import deployability
    src = open(os.path.join(_HERE, f"seed_ga_{arm}_models", key + ".py")).read()
    dep = deployability(src)
    prm = {"lr": lr, "batch": batch, "transform": "denoise",
           "epoch": epoch, "num_workers": workers}
    t0 = time.time()
    try:
        _n, acc, _a2t, _q = api.check_nn(src, TASK, DATASET, METRIC, prm,
                                         save_to_db=False, prefix="DnFullTrain")
        psnr = acc * PSNR_SCALE
    except Exception as e:
        psnr = 0.0
        log(f"    DEAD {rec.get('name')} -> {type(e).__name__}: {str(e)[:80]}")
    dt = time.time() - t0
    return {"name": rec.get("name"), "arm": arm, "checksum": key,
            "psnr50": psnr, "psnr5_proxy": rec.get("psnr"),
            "eff5_proxy": rec.get("eff_db"), "macs": rec.get("macs"),
            "params": rec.get("params"), "deployable": dep["deployable"],
            "hard": dep["hard"], "edge_pick": rec.get("_edge_pick", False),
            "epoch": epoch, "train_sec": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["A0", "B0", "C0", "D0"])
    ap.add_argument("--k", type=int, default=3, help="top-K by eff-dB per arm")
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--include-checksums", nargs="*", default=[],
                    help="explicit extra checksums to full-train (from arms[0] namespace); "
                         "used to append distinct non-top-K winners for diversity")
    ap.add_argument("--results", default=os.path.join(_HERE, "fulltrain_results.json"))
    args = ap.parse_args()

    def log(m):
        print(m, flush=True)

    done = {}
    if os.path.exists(args.results):
        try:
            done = {r["checksum"]: r for r in json.load(open(args.results))["runs"]}
            log(f"resume: {len(done)} already full-trained")
        except Exception:
            done = {}

    log(f"=== Full-train top-{args.k} per arm @ {args.epoch} epochs "
        f"(budget {BUDGET_MMAC:.0f} MMac) ===")
    plan = []
    for arm in args.arms:
        for key, rec in _select(arm, args.k, log):
            plan.append((arm, key, rec))

    # explicit extra candidates (e.g. distinct non-Lite winners for diversity),
    # looked up in the first arm's namespace; appended if not already selected.
    if args.include_checksums:
        arm = args.arms[0]
        inc_cache = json.load(open(os.path.join(_HERE, f"seed_ga_{arm}_cache.json")))
        inc_mdir = os.path.join(_HERE, f"seed_ga_{arm}_models")
        have = {k for _a, k, _r in plan}
        for cks in args.include_checksums:
            if cks in have:
                log(f"  [{arm}] include {cks[:12]} — already in top-K, skipped")
                continue
            rec = inc_cache.get(cks)
            if rec and os.path.exists(os.path.join(inc_mdir, cks + ".py")):
                rec = dict(rec); rec["_included"] = True
                plan.append((arm, cks, rec)); have.add(cks)
                log(f"  [{arm}] + included (diversity) {rec.get('name')}")
            else:
                log(f"  [{arm}] include {cks[:12]} — not found, skipped")
    log(f"plan: {len(plan)} (arm,candidate) slots across {len(args.arms)} arms\n")

    runs = list(done.values())
    for i, (arm, key, rec) in enumerate(plan, 1):
        if key in done:
            log(f"[{i}/{len(plan)}] {arm} {rec.get('name')} — cached "
                f"(psnr50 {done[key]['psnr50']:.3f})")
            continue
        log(f"[{i}/{len(plan)}] {arm} {rec.get('name')} — training {args.epoch}ep ...")
        r = _train(arm, key, rec, args.epoch, args.batch, args.lr,
                   args.num_workers, log)
        runs.append(r); done[key] = r
        json.dump({"budget_mmac": BUDGET_MMAC, "epoch": args.epoch, "runs": runs},
                  open(args.results, "w"), indent=2)
        log(f"    -> psnr50 {r['psnr50']:.3f} dB "
            f"(proxy ep5 {r['psnr5_proxy']:.3f}) | {(r['macs'] or 0)/1e6:.0f} MMac "
            f"| deployable={r['deployable']} | {r['train_sec']:.0f}s")

    # ---- report: per-arm best@50ep + proxy-vs-full rank check ----
    log("\n=== RESULT (full-train @%dep) ===" % args.epoch)
    log(f"{'arm':4}{'name':34}{'psnr50':>8}{'proxyep5':>9}{'MMac':>8}{'deploy':>8}")
    by_arm = {}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)
    for arm in args.arms:
        for r in sorted(by_arm.get(arm, []), key=lambda x: x["psnr50"], reverse=True):
            tag = " *edge" if r.get("edge_pick") else ""
            log(f"{arm:4}{(r['name'] or '')[:33]:34}{r['psnr50']:8.3f}"
                f"{(r['psnr5_proxy'] or 0):9.3f}{(r['macs'] or 0)/1e6:8.0f}"
                f"{('YES' if r['deployable'] else 'no'):>8}{tag}")
    log(f"\nResults -> {args.results}")


if __name__ == "__main__":
    main()
