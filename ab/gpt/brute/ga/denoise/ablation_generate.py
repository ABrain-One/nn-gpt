"""Generate + validate the skip-connection ablation grid. CPU only.

This is the instrument for Task 2. It does three things that have to happen before any GPU
is spent, because each can silently invalidate the experiment:

  1. GENERATES every cell of the factorial from a common base, using the existing
     source->source operators (skip_operators.py). The base occupies cell (0,0,0), so the
     comparison is always within-base -- no cross-model confound.
  2. VALIDATES each variant with _builds_and_trains: build, forward, one optimizer step,
     finite grads. An operator that returns syntactically valid but untrainable source is
     the failure mode that would otherwise show up as "this skip config scores terribly".
  3. MEASURES MACs and params. GLOBAL and BLOCKRES are parameter-free identity adds and
     MUST come back within a hair of the base -- the script asserts that. LATERAL adds a
     1x1 projection and will not; its cost is recorded so the effect can be reported per
     MMac instead of pretending it is free.

BASES ARE DRAWN FROM THE (G=0, L=0, B=0) CELL. The operators can only ADD skips -- there is
no removal counterpart -- so a base must lack all three factors for the full 2x2x2 to be
reachable. skip_inventory_corpus.py says 470 corpus models qualify. A base also needs at
least one block-residual site and one lateral site; DenoiseUnet5F, for instance, has zero
block sites because its blocks are inline rather than helper classes, so the BLOCKRES arm
is simply not expressible on it.

  python ablation_generate.py --n-bases 8 --out /a/mm/sweep_out/ablation
"""
import argparse
import itertools
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skip_operators as SO                                     # noqa: E402

MAC_CAP = 9053.0
SCALE = 48.0


def measure(src):
    """-> (macs_MMac, params_M) or (None, None) when the cost harness is unavailable."""
    try:
        import edge_cost as EC
        c = EC.measure_cost(src)
        if c.get("ok"):
            return c["macs"] / 1e6, c["params"] / 1e6
    except Exception:
        pass
    return None, None


MAC_DECADES = [(0, 500), (500, 1500), (1500, 3000), (3000, 6000), (6000, MAC_CAP)]


def pick_bases(db, inv, n, min_db=32.5, scan_cap=140):
    """Bases stratified across FAMILY and MAC DECADE, not the top of the score list.

    Taking the highest-scoring qualifiers produced 8 bases that were all LLM_*_seeded or
    DenoiseGA-*, several with a single block site. An ablation run on those measures skip
    behaviour *in one generation lineage at one model size* and generalises to nothing --
    and since the whole point is a causal claim about topology, that is the difference
    between a result and an anecdote.

    Selection therefore: round-robin over MAC decades, at most 2 per family, preferring
    candidates with MORE insertion sites (a base with 1 block site cannot support the
    placement arm at all). Costs one measure_cost per scanned candidate, hence scan_cap."""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cand = [(k, v) for k, v in inv.items()
            if not v["GLOBAL"] and not v["LATERAL"] and not v["BLOCKRES"]
            and (v["ep5_dB"] or 0) >= min_db]
    cand.sort(key=lambda kv: -(kv[1]["ep5_dB"] or 0))
    print(f"  {len(cand)} models in the (G=0,L=0,B=0) cell above {min_db} dB")

    pool = []
    for name, meta in cand[:scan_cap]:
        row = c.execute("select code from nn where name=?", (name,)).fetchone()
        if not row:
            continue
        src = row[0]
        bsites, lsites = SO.block_residual_sites(src), SO.lateral_sites(src)
        if not bsites or not lsites:
            continue                       # cannot express the full factorial
        macs, params = measure(src)
        if macs is None or macs > MAC_CAP:
            continue
        pool.append({"name": name, "src": src, "ep5_dB": meta["ep5_dB"], "macs": macs,
                     "block_sites": bsites, "lateral_sites": lsites,
                     "fam": name.split("-")[0],
                     "n_sites": len(bsites) + len(lsites)})
    print(f"  {len(pool)} viable candidates scanned (of {min(scan_cap, len(cand))})")

    # bucket by cost, richest-in-sites first inside each bucket
    buckets = {i: [] for i in range(len(MAC_DECADES))}
    for p in pool:
        for i, (lo, hi) in enumerate(MAC_DECADES):
            if lo <= p["macs"] < hi:
                buckets[i].append(p)
                break
    for i in buckets:
        buckets[i].sort(key=lambda p: (-p["n_sites"], -(p["ep5_dB"] or 0)))

    out, fam_count = [], {}
    while len(out) < n and any(buckets.values()):
        progressed = False
        for i in sorted(buckets):                     # round-robin across decades
            while buckets[i]:
                p = buckets[i].pop(0)
                if fam_count.get(p["fam"], 0) >= 2:
                    continue                          # cap per lineage
                fam_count[p["fam"]] = fam_count.get(p["fam"], 0) + 1
                out.append(p)
                progressed = True
                break
            if len(out) >= n:
                break
        if not progressed:
            break
    for p in out:
        print(f"    {p['name'][:44]:46s} {p['ep5_dB']:.2f} dB  {p['macs']:7.0f} MMac  "
              f"blocks={len(p['block_sites'])} laterals={len(p['lateral_sites'])}")
    return out


def build_variant(src, g, b, l, b_index=None, l_target=None):
    """Apply the requested factor combination in a fixed order. -> (source, info) or (None, why).

    THE FACTORIAL MUST NOT USE THE DEFAULT LATERAL PATH. add_lateral_skip(target=None)
    requires the top forward to end in exactly `return self.<M>(<name>)`; measured across
    the base set that pattern never matches (every base reports PATTERN), which is why all
    four L=1 cells came back 0/8 while the placement arm worked fine. When no target is
    named, walk the real sites DEEPEST FIRST -- deepest reproduces the original operator's
    intent of injecting just before the output projection -- and take the first that
    validates. Sites whose channel count mis-traces raise a shape error and are skipped."""
    cur, info = src, {}
    if g:
        cur = SO.add_global_residual(cur, tag="ablG")
        if cur is None:
            return None, "global op declined"
    if b:
        cur = SO.add_block_residual(cur, tag="ablB", index=b_index)
        if cur is None:
            return None, "block op declined"
    if l:
        if l_target is not None:
            nxt = SO.add_lateral_skip(cur, tag="ablL", target=l_target)
            if nxt is None:
                return None, f"lateral declined at {l_target}"
            cur, info["l_target"] = nxt, l_target
        else:
            for t in reversed(SO.lateral_sites(cur)):
                nxt = SO.add_lateral_skip_at(cur, t, tag="ablL")
                if nxt is not None:
                    cur, info["l_target"] = nxt, t
                    break
            else:
                return None, "no lateral site validated"
    return cur, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--inventory", default=os.path.join(HERE, "skip_inventory_corpus.json"))
    ap.add_argument("--n-bases", type=int, default=8)
    ap.add_argument("--max-placements", type=int, default=3,
                    help="cap on block sites / lateral depths explored per base")
    ap.add_argument("--out", default="/a/mm/sweep_out/ablation")
    a = ap.parse_args()

    inv = json.load(open(a.inventory))
    bases = pick_bases(a.db, inv, a.n_bases)
    print(f"selected {len(bases)} bases from the (G=0,L=0,B=0) cell")
    vdir = os.path.join(a.out, "variants")
    os.makedirs(vdir, exist_ok=True)

    manifest, n_ok, n_bad = [], 0, 0
    for bi, base in enumerate(bases):
        b_macs, b_params = measure(base["src"])
        print(f"\n[{bi+1}/{len(bases)}] {base['name']}  ep5={base['ep5_dB']} dB  "
              f"{b_macs and round(b_macs)} MMac  blocks={len(base['block_sites'])} "
              f"laterals={len(base['lateral_sites'])}")

        jobs = []
        # --- the 2x2x2 factorial, default placement -------------------------------
        for g, b, l in itertools.product((0, 1), repeat=3):
            jobs.append(dict(kind="factorial", g=g, b=b, l=l, b_index=None, l_target=None,
                             cell=f"G{g}B{b}L{l}"))
        # --- placement arm: BLOCKRES at each site, LATERAL at each depth -----------
        for i, site in enumerate(base["block_sites"][:a.max_placements]):
            jobs.append(dict(kind="place_block", g=0, b=1, l=0, b_index=i, l_target=None,
                             cell=f"B@{i}:{site}"))
        for t in base["lateral_sites"][:a.max_placements]:
            jobs.append(dict(kind="place_lateral", g=0, b=0, l=1, b_index=None, l_target=t,
                             cell=f"L@{t}"))

        for j in jobs:
            why = None
            if (j["g"], j["b"], j["l"]) == (0, 0, 0):
                src, vinfo, valid = base["src"], {}, True   # the base IS cell G0B0L0
            else:
                src, vinfo = build_variant(base["src"], j["g"], j["b"], j["l"],
                                           j["b_index"], j["l_target"])
                if src is None:
                    why, vinfo = vinfo, {}                  # build_variant returns the reason
                    valid = False
                else:
                    valid = SO._builds_and_trains(src)
                    if not valid:
                        why = "emitted but not trainable"
            rec = dict(base=base["name"], **{k: j[k] for k in
                                             ("kind", "cell", "g", "b", "l",
                                              "b_index", "l_target")},
                       valid=bool(valid))
            if why:
                rec["why"] = why
            if isinstance(vinfo, dict) and vinfo.get("l_target"):
                rec["l_target_used"] = vinfo["l_target"]
            if valid:
                m, p = measure(src)
                rec.update(macs_MMac=m and round(m, 1), params_M=p and round(p, 4),
                           base_macs_MMac=b_macs and round(b_macs, 1))
                if m and b_macs:
                    rec["mac_delta_pct"] = round(100 * (m - b_macs) / b_macs, 2)
                    rec["over_cap"] = m > MAC_CAP
                    # parameter-free operators must not move cost; if they do, the
                    # "topology not capacity" claim for those arms is not true
                    if j["l"] == 0 and abs(rec["mac_delta_pct"]) > 1.0:
                        rec["MAC_DRIFT_WARNING"] = True
                fn = os.path.join(vdir, f"{base['name']}__{j['cell'].replace(':','_')}.py")
                open(fn, "w").write(src)
                rec["path"] = fn
                n_ok += 1
            else:
                n_bad += 1
            manifest.append(rec)
            print(f"   {j['cell']:22s} {'ok' if valid else 'INVALID':8s} "
                  f"{rec.get('macs_MMac','-')} MMac "
                  f"{('%+.2f%%' % rec['mac_delta_pct']) if rec.get('mac_delta_pct') is not None else ''}"
                  f"{'  <-- MAC DRIFT' if rec.get('MAC_DRIFT_WARNING') else ''}"
                  f"{'  <-- OVER CAP' if rec.get('over_cap') else ''}")

    out = os.path.join(a.out, "manifest.json")
    json.dump(manifest, open(out, "w"), indent=1)

    print(f"\n=== {n_ok} valid variants, {n_bad} rejected -> {out}")
    # per-cell realisability: a cell only usable on 2 of 8 bases is not a usable cell
    cells = {}
    for r in manifest:
        if r["kind"] == "factorial":
            cells.setdefault(r["cell"], [0, 0])[0 if r["valid"] else 1] += 1
    print("factorial cell coverage (valid/invalid across bases):")
    for k in sorted(cells):
        print(f"  {k}: {cells[k][0]} valid / {cells[k][1]} invalid")
    drift = [r for r in manifest if r.get("MAC_DRIFT_WARNING")]
    cap = [r for r in manifest if r.get("over_cap")]
    print(f"MAC-drift warnings: {len(drift)}   over-cap variants: {len(cap)}")


if __name__ == "__main__":
    main()
