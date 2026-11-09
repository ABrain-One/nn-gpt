"""Hypervolume and IGD+ on the quality/latency front, per producer. NO EXTERNAL DEPS.

v2 OF pareto_metrics.py. The HV and IGD+ mathematics in v1 are correct and are reused
verbatim. v2 fixes three defects in how v1 LABELLED and GROUPED the points, each of which
changed a per-producer number that the paper would have quoted.

DEFECT 1 -- THE "hand/seed" ARM WAS NOT HAND-DESIGNED. v1's producer() ended in a bare
`return "hand/seed"`, so every name that matched no GA/LLM prefix fell into it. Three
families fall through that hole:
  * bl0-<uid> / bl1-<uid>  -- REPLICATE RUNS of GA architectures. bl0-3e4bde5d... and
    DnSeedGA-3e4bde5d... are the same architecture (ga_vs_rw_fullres.json scores the
    DnSeedGA- name at 36.906); the bl prefix is a replicate index, not a designer.
  * DenoiseRW-<uid>        -- the RANDOM-WALK control arm, a generator, not a human.
  * bare hand names        -- DenoiseUnet*, LiteDenoiseNet, DenoiseMoE, ... the real humans.
At full resolution this put GA replicates and random-walk models into the arm that the
paper's automation-versus-human claim is measured against. They now get their own labels.

DEFECT 2 -- ONE ARCHITECTURE COULD OCCUPY SEVERAL FRONT POINTS. LiteDenoiseNet appears as
_ep40.._ep50 and _best, and every bl0-/bl1- name repeats a DnSeedGA- architecture. A front
over ARCHITECTURES must carry each architecture once, so v2 collapses on arch_uid2 (the
32-hex suffix, else the epoch-stripped base name) keeping the best quality. The identity is
arch_uid2, consistent with the rest of the pipeline.

DEFECT 3 -- quality_axis WAS A HARDCODED STRING. v1 wrote "full-res RGB PSNR, 50 frames at
3200x2432" into every output, including the two runs fed crop_psnr_union.json, so
pareto_crop_s26u.json and pareto_legal_s26u.json both claim an axis they do not carry. v2
derives the label from the quality file and cross-checks it against --noop, and refuses to
run if they disagree, because the floor is what converts the axis into hypervolume.

NOT CHANGED, AND STILL TRUE OF v2: quality is measured at a different input size from
latency (see v1's header); the reference front is the union of all arms; IGD+ rather than
IGD; the reference point is swept over latency budgets rather than fixed.

  python pareto_metrics_v2.py --quality .../fullres_union.json --noop 30.88 \
      --device 'Samsung Galaxy S26 Ultra' --out .../pareto_v2_fullres_s26u.json
"""
import argparse
import json
import math
import os
import re

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
NOOP_FULLRES_RGB = 30.88
NOOP_CROP_Y = 33.021
# v1 swept 5/10/20/50 ms, but the whole global front lies below 2 ms, so every one of
# those references sits outside the data and the four columns differ by a constant --
# they rank the arms identically and measure nothing. Budgets that actually cut through
# the front are what separate "owns the sub-millisecond regime" from "owns the quality
# end", so the sweep now starts inside the data and keeps 5 ms for continuity with v1.
LAT_BUDGETS = (0.5, 1.0, 2.0, 5.0)

# Which floor belongs with which quality file. Getting this pair wrong is worth ~2.1 dB of
# unearned hypervolume per model, so it is checked rather than trusted.
AXES = {
    "fullres_union.json": ("full-res RGB PSNR, 50 frames at 3200x2432", NOOP_FULLRES_RGB),
    "crop_psnr_union.json": ("Y-channel PSNR, 256x256 centre crops", NOOP_CROP_Y),
    # Same metric and floor as crop_psnr_union, but restricted to models whose
    # training actually REACHED epoch 50 (max(stat.epoch) >= 50 over all shard DBs).
    # crop_psnr_union pools 5-epoch screening scores with a handful of 50-epoch hand
    # numbers, which makes a per-producer front a budget comparison rather than an
    # architecture one. This axis removes that confound by construction.
    "matched50_crop_psnr.json": ("Y-channel PSNR, 256x256 centre crops, matched 50-epoch budget",
                                 NOOP_CROP_Y),
}

UID = re.compile(r"([0-9a-f]{32})")
EPOCH_SUFFIX = re.compile(r"_(?:ep\d+|best|final)$")


def canon(name):
    return re.sub(r"[_-]+", "-", name).lower()


def arch_id(name):
    """Collapse replicates and per-epoch checkpoints of ONE architecture to one key."""
    m = UID.search(name.lower())
    if m:
        return m.group(1)
    return EPOCH_SUFFIX.sub("", name).lower()


# Order matters: the longer/more specific prefixes come first.
PRODUCERS = (
    ("bl0-", "GA replicate"),
    ("bl1-", "GA replicate"),
    ("DenoiseRW", "random walk (control)"),
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
    ("DnNewFam", "GA other"),          # ga-legacy per dataset_manifest.GA_LEGACY_RE
    ("LLM", "LLM"),
)

# Classifying the fallback. Every GENERATED name in this corpus carries its arch_uid2 as a
# 32-hex suffix -- DenoiseGA-<uid>, DnSeedGA-<uid>, LLM_*-<uid>, bl0-<uid>, DenoiseRW-<uid>.
# Hand-written models are bare module names (DenoiseUnet5, LiteDenoiseNet, DenoiseKPN), with
# at most an _ep<N>/_best checkpoint suffix. So "no arch_uid2" IS the hand-written test, and
# it needs no maintained whitelist -- an earlier prefix list silently missed 14 hand-written
# variants (DenoiseAxial, DenoiseKPN, DenoiseWaveletMS, ...) on the first run.
def producer(name):
    for pre, lab in PRODUCERS:
        if name.startswith(pre):
            return lab
    if not UID.search(name.lower()):
        return "hand-designed"
    return "unclassified"


def front(pts):
    best = {}
    for lat, q, n in pts:
        if lat not in best or q > best[lat][1]:
            best[lat] = (lat, q, n)
    out, top = [], -1e18
    for lat in sorted(best):
        lat, q, n = best[lat]
        if q > top:
            out.append((lat, q, n))
            top = q
    return out


def hypervolume(fr, lat_ref, q_ref):
    hv, prev_q = 0.0, q_ref
    for lat, q, _ in fr:
        if lat >= lat_ref or q <= q_ref:
            continue
        if q > prev_q:
            hv += (q - prev_q) * (lat_ref - lat)
            prev_q = q
    return hv


def igd_plus(approx, ref, lat_span, q_span):
    if not approx or not ref:
        return None
    tot = 0.0
    for rl, rq, _ in ref:
        tot += min(math.hypot(max(al - rl, 0.0) / lat_span, max(rq - aq, 0.0) / q_span)
                   for al, aq, _ in approx)
    return tot / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default=os.path.join(SWEEP, "fullres_union.json"))
    ap.add_argument("--latency", default=os.path.join(SWEEP, "aihub/latency_corpus.json"))
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=os.path.join(SWEEP, "pareto_v2.json"))
    ap.add_argument("--noop", type=float, default=None)
    ap.add_argument("--macs", default=os.path.join(SWEEP, "verify_macs.json"))
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep per-epoch and replicate names as separate points (v1 behaviour)")
    a = ap.parse_args()

    base = os.path.basename(a.quality)
    axis_label, want_noop = AXES.get(base, (None, None))
    if axis_label is None:
        return "unknown quality file %r: add it to AXES with its no-op floor" % base
    if a.noop is None:
        a.noop = want_noop
    elif abs(a.noop - want_noop) > 1e-9:
        return ("floor/axis mismatch: %s needs --noop %.3f, got %.3f. The floor defines the "
                "hypervolume reference; a wrong pairing silently credits every model."
                % (base, want_noop, a.noop))
    print("quality axis : %s" % axis_label)
    print("no-op floor  : %.3f dB" % a.noop)

    q = {canon(k): (k, v) for k, v in json.load(open(a.quality)).items()}
    legal, cap = None, None
    if a.macs and os.path.exists(a.macs):
        mj = json.load(open(a.macs))
        cap = float(mj.get("cap_MMac") or 9053.0)
        macs = mj["macs"]
        legal = {canon(k) for k, v in macs.items() if v is not None and float(v) <= cap}
        # A per-epoch checkpoint carries no MAC row of its own; it inherits the base
        # architecture's. Without this, every LiteDenoiseNet_ep* was dropped as illegal.
        for k, v in list(macs.items()):
            if v is not None and float(v) <= cap:
                legal.add(canon(EPOCH_SUFFIX.sub("", k)))
        print("edge-legal   : %d names at or under %.0f MMac" % (len(legal), cap))

    lat = json.load(open(a.latency))
    pts, skipped = [], {"no_quality": 0, "no_ms": 0, "wrong_device": 0, "over_mac_cap": 0}
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
        if legal is not None:
            cn = canon(name)
            if cn not in legal and canon(EPOCH_SUFFIX.sub("", name)) not in legal:
                skipped["over_mac_cap"] += 1
                continue
        pts.append((float(rec["ms"]), float(hit[1]), hit[0]))

    print("paired       : %d   dropped %s" % (len(pts), skipped))

    n_raw = len(pts)
    if not a.no_dedupe:
        bya = {}
        for lat_, qq, n in pts:
            k = arch_id(n)
            if k not in bya or qq > bya[k][1]:
                bya[k] = (lat_, qq, n)
        pts = list(bya.values())
        print("deduped      : %d -> %d distinct architectures (arch_uid2)" % (n_raw, len(pts)))
    if not pts:
        return "no paired points"

    lats = [p[0] for p in pts]
    qs = [p[1] for p in pts]
    lat_span = max(lats) - min(lats) or 1.0
    q_span = max(qs) - min(qs) or 1.0

    gfront = front(pts)
    print("\nGLOBAL FRONT: %d points" % len(gfront))
    for lat_, qq, n in gfront:
        print("  %8.3f ms  %7.3f dB  %-46s [%s]" % (lat_, qq, n[:46], producer(n)))

    by = {}
    for p in pts:
        by.setdefault(producer(p[2]), []).append(p)

    print("\nHYPERVOLUME (dB*ms, reference quality = %.3f dB)" % a.noop)
    print("  %-22s %6s " % ("producer", "n") +
          " ".join("%11s" % ("HV@%gms" % b) for b in LAT_BUDGETS) +
          "   %8s %7s" % ("IGD+", "onfront"))
    rows = {}
    gset = {n for _, _, n in gfront}
    for lab in sorted(by, key=lambda L: -len(by[L])):
        grp = by[lab]
        fr = front(grp)
        hv = {b: hypervolume(fr, b, a.noop) for b in LAT_BUDGETS}
        ig = igd_plus(fr, gfront, lat_span, q_span)
        oncnt = sum(1 for _, _, n in fr if n in gset)
        rows[lab] = {"n": len(grp), "front": len(fr), "hv": hv, "igd_plus": ig,
                     "on_global_front": oncnt,
                     "best_dB": max(p[1] for p in grp),
                     "fastest_ms": min(p[0] for p in grp)}
        print("  %-22s %6d " % (lab, len(grp)) +
              " ".join("%11.1f" % hv[b] for b in LAT_BUDGETS) +
              "   %8.4f %7d" % (ig if ig is not None else float("nan"), oncnt))
    gl = {b: hypervolume(gfront, b, a.noop) for b in LAT_BUDGETS}
    print("  %-22s %6d " % ("ALL (reference)", len(pts)) +
          " ".join("%11.1f" % gl[b] for b in LAT_BUDGETS) + "   %8.4f %7d" % (0.0, len(gfront)))

    if "unclassified" in rows:
        names = sorted({p[2] for p in by["unclassified"]})[:12]
        print("\nWARNING %d unclassified points -- classify before quoting: %s"
              % (rows["unclassified"]["n"], names))

    json.dump({"n_paired_raw": n_raw, "n_architectures": len(pts),
               "deduped_on": "arch_uid2" if not a.no_dedupe else None,
               "device": a.device or "all (POOLED)", "noop_floor_dB": a.noop,
               "lat_budgets_ms": list(LAT_BUDGETS),
               "quality_file": base, "quality_axis": axis_label,
               "latency_axis": "on-device ms at 3x256x256",
               "mac_cap_MMac": cap, "generator": "pareto_metrics_v2.py",
               "global_front": [{"ms": l, "dB": qq, "model": n, "producer": producer(n)}
                                for l, qq, n in gfront],
               "global_hv": gl, "by_producer": rows},
              open(a.out, "w"), indent=1)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    raise SystemExit(main())
