"""Hypervolume and IGD+ on the quality/latency front, per producer. NO EXTERNAL DEPS.

WHY THIS EXISTS. PEL-NAS (arXiv:2510.01472) reports Pareto hypervolume and IGD; this corpus
reports a front and no scalar, which makes the two bodies of work hard to compare on the one
axis a reviewer will reach for. With exactly two objectives both metrics are exact in closed
form, so pymoo is unnecessary -- and pymoo is not installed on the login node anyway.

WHAT THE AXES ARE, AND THE MISMATCH IN THEM. Quality is full-resolution RGB PSNR on the
published protocol (50 frames at 3200x2432). Latency is measured on-device at 3x256x256.
These are different input sizes: the pairing is the corpus's own, inherited from the fact
that quality and latency were collected by different harnesses, and it is stated rather than
hidden. It is defensible only because latency RANKS consistently across resolution within a
residency regime (sec:fullhd, r=0.949) -- not because the numbers refer to the same forward
pass. A Full HD latency column would be the clean fix and does not exist for 2,812 models.

THE REFERENCE POINT IS A CHOICE AND IT CHANGES THE ANSWER, so it is reported as a sweep
rather than a single number. Quality reference is the NO-OP FLOOR, 30.88 dB on this protocol:
a model that cannot beat returning its input unchanged has no value, so it should contribute
zero hypervolume rather than a small positive amount. Latency reference is swept over
deployment budgets, because "which arm is better" is a different question inside 10 ms than
inside 50 ms, and a single reference point silently picks one.

IGD+ RATHER THAN IGD. IGD punishes an arm for not covering parts of the reference front even
when its own points dominate them, which inverts verdicts; IGD+ only counts the dominated
component. The reference front is the union of all arms (standard when the true front is
unknown), so IGD+ reads as "distance from the best front anyone reached".

  python pareto_metrics.py --out /a/mm/sweep_out/pareto_metrics.json
"""
import argparse
import json
import math
import os
import re

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
NOOP_FULLRES_RGB = 30.88          # published protocol floor; see sec:baselines
LAT_BUDGETS = (5.0, 10.0, 20.0, 50.0)


def canon(name):
    """Join key. fullres_union uses hyphens, latency_corpus uses underscores for some
    families (LLM_Denoise- vs DnSeed2Act-), so neither side can be the canonical form."""
    return re.sub(r"[_-]+", "-", name).lower()


PRODUCERS = (
    ("DnSeed2Fast", "GA frontier-seeded"),
    ("DnSeed2Act", "GA operator-weighted"),
    ("DnSeed2Wide", "GA dimension-weighted"),
    ("DnSeedGA2", "GA replication"),
    ("DnSeedGA", "GA seeded"),
    ("DnSeed3", "GA full-res fitness"),
    ("DenoiseGA", "GA main"),
    ("DnSearch", "GA other"),
    ("DnSkip", "GA other"),
    ("DenoiseSkip", "GA other"),
    ("LLM", "LLM"),
)


def producer(name):
    for pre, lab in PRODUCERS:
        if name.startswith(pre):
            return lab
    return "hand/seed"


def front(pts):
    """pts: [(lat, psnr, name)] -> non-dominated set, sorted by latency ascending.
    Minimise latency, maximise PSNR. Ties broken so an identical-latency pair keeps only
    the better PSNR."""
    best = {}
    for lat, q, n in pts:
        if lat not in best or q > best[lat][1]:
            best[lat] = (lat, q, n)
    out, top = [], -1e18
    for lat in sorted(best):
        lat, q, n = best[lat]
        if q > top:                     # strictly better than anything faster
            out.append((lat, q, n))
            top = q
    return out


def hypervolume(fr, lat_ref, q_ref):
    """Exact 2-D HV of the region dominated by `fr` w.r.t. (lat_ref, q_ref).

    fr is sorted by latency ascending, so PSNR is ascending too (anything else would be
    dominated). The dominated region is the union of [lat_i, lat_ref] x [q_ref, q_i], which
    telescopes into one pass. Units are dB*ms."""
    hv, prev_q = 0.0, q_ref
    for lat, q, _ in fr:
        if lat >= lat_ref or q <= q_ref:
            continue                     # outside the box: contributes nothing
        if q > prev_q:
            hv += (q - prev_q) * (lat_ref - lat)
            prev_q = q
    return hv


def igd_plus(approx, ref, lat_span, q_span):
    """Mean modified distance from each reference point to the approximation set.

    Axes are normalised by the global data range because one is ms and the other dB; an
    unnormalised Euclidean distance would be a latency metric with a rounding error in it."""
    if not approx or not ref:
        return None
    tot = 0.0
    for rl, rq, _ in ref:
        d = min(
            math.hypot(max(al - rl, 0.0) / lat_span, max(rq - aq, 0.0) / q_span)
            for al, aq, _ in approx)
        tot += d
    return tot / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default=os.path.join(SWEEP, "fullres_union.json"))
    ap.add_argument("--latency", default=os.path.join(SWEEP, "aihub/latency_corpus.json"))
    ap.add_argument("--device", default="", help="restrict to one device (recommended)")
    ap.add_argument("--out", default=os.path.join(SWEEP, "pareto_metrics.json"))
    # The floor belongs to the METRIC, not to the tool: 30.88 dB is the full-res RGB no-op and
    # 33.021 dB is the 256-crop Y no-op. Passing the crop quality file with the full-res floor
    # would credit every model with ~2.1 dB of hypervolume it has not earned.
    ap.add_argument("--noop", type=float, default=NOOP_FULLRES_RGB)
    # EDGE-LEGALITY IS NOT OPTIONAL. Without it the front is topped by hand-written models
    # that the search is forbidden to produce -- DenoiseUnet5/Unet7 run 3.3x over the cap --
    # so every generated arm is scored against a bar the thesis itself excludes. The paper's
    # own best_hand_legal is LiteDenoiseNet at 3,529.5 MMac for exactly this reason.
    ap.add_argument("--macs", default=os.path.join(SWEEP, "verify_macs.json"),
                    help="json with {macs: {model: MMac}, cap_MMac: N}; '' disables the filter")
    a = ap.parse_args()

    q = {canon(k): (k, v) for k, v in json.load(open(a.quality)).items()}
    legal, cap = None, None
    if a.macs and os.path.exists(a.macs):
        mj = json.load(open(a.macs))
        cap = float(mj.get("cap_MMac") or 9053.0)
        legal = {canon(k) for k, v in mj["macs"].items() if v is not None and float(v) <= cap}
        print("edge-legal filter: %d models at or under %.0f MMac" % (len(legal), cap))
    lat = json.load(open(a.latency))
    pts, skipped = [], {"no_quality": 0, "no_ms": 0, "wrong_device": 0}
    for name, rec in lat.items():
        if a.device and rec.get("device") != a.device:
            skipped["wrong_device"] += 1
            continue
        if rec.get("ms") is None:
            skipped["no_ms"] += 1
            continue
        hit = q.get(canon(name))
        if hit is None:
            skipped["no_quality"] += 1
            continue
        if legal is not None and canon(name) not in legal:
            skipped["over_mac_cap"] = skipped.get("over_mac_cap", 0) + 1
            continue
        pts.append((float(rec["ms"]), float(hit[1]), hit[0]))

    print("paired on both axes : %d" % len(pts))
    print("  dropped: %s" % skipped)
    if not pts:
        return 2
    lats = [p[0] for p in pts]
    qs = [p[1] for p in pts]
    lat_span = max(lats) - min(lats) or 1.0
    q_span = max(qs) - min(qs) or 1.0
    print("  latency %.3f - %.3f ms   quality %.3f - %.3f dB" %
          (min(lats), max(lats), min(qs), max(qs)))

    gfront = front(pts)
    print("\nGLOBAL FRONT: %d points" % len(gfront))
    for lat, qq, n in gfront:
        print("  %8.3f ms  %7.3f dB  %-46s [%s]" % (lat, qq, n[:46], producer(n)))

    by = {}
    for p in pts:
        by.setdefault(producer(p[2]), []).append(p)

    print("\nHYPERVOLUME (dB*ms, reference quality = %.2f dB no-op floor)" % a.noop)
    hdr = "  %-22s %6s " % ("producer", "n") + " ".join(
        "%11s" % ("HV@%gms" % b) for b in LAT_BUDGETS) + "   %8s %7s" % ("IGD+", "onfront")
    print(hdr)
    rows = {}
    gset = {n for _, _, n in gfront}
    for lab in sorted(by, key=lambda L: -len(by[L])):
        grp = by[lab]
        fr = front(grp)
        hv = {b: hypervolume(fr, b, a.noop) for b in LAT_BUDGETS}
        ig = igd_plus(fr, gfront, lat_span, q_span)
        oncnt = sum(1 for _, _, n in fr if n in gset)
        rows[lab] = {"n": len(grp), "front": len(fr), "hv": hv, "igd_plus": ig,
                     "on_global_front": oncnt}
        print("  %-22s %6d " % (lab, len(grp)) +
              " ".join("%11.1f" % hv[b] for b in LAT_BUDGETS) +
              "   %8.4f %7d" % (ig if ig is not None else float("nan"), oncnt))
    gl = {b: hypervolume(gfront, b, a.noop) for b in LAT_BUDGETS}
    print("  %-22s %6d " % ("ALL (reference)", len(pts)) +
          " ".join("%11.1f" % gl[b] for b in LAT_BUDGETS) + "   %8.4f %7d" % (0.0, len(gfront)))

    json.dump({"n_paired": len(pts), "device": a.device or "all (POOLED)",
               "noop_floor_dB": a.noop, "lat_budgets_ms": list(LAT_BUDGETS),
               "quality_axis": "full-res RGB PSNR, 50 frames at 3200x2432",
               "latency_axis": "on-device ms at 3x256x256",
               "global_front": [{"ms": l, "dB": qq, "model": n, "producer": producer(n)}
                                for l, qq, n in gfront],
               "global_hv": gl, "by_producer": rows},
              open(a.out, "w"), indent=1)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    raise SystemExit(main())
