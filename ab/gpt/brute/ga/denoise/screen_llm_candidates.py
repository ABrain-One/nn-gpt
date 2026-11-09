"""Screen LLM-generated denoise candidates through the SAME gates the GA applies.

"9 of 10 parsed" is a parser statistic, not a yield. A candidate only counts if it also
(a) builds and runs a forward pass, (b) fits the 9053 MMac edge budget, and (c) is not a
graph-level clone of something already in the corpus. This applies all three and reports
where each candidate dies, so cold and seeded can be compared on survivors rather than on
raw text output.

Run in-container: torch, thop and the mutator package are all required.

Usage: python screen_llm_candidates.py [run_dir] [--cap 9053]
"""
import argparse
import glob
import json
import os
import re
import sys

CAP = 9053.0
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast",
           os.path.abspath(os.path.join(HERE, "..", "..", "ast"))):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _make_gate(graded=True, cos_max=0.95):
    """Use the GA's own DiversityGate so 'distinct' means exactly what it means there.

    Two mistakes this replaces, both mine:
      * I probed arch_embed for wl_hash/graph_hash/embed_source and landed on
        `embed_source`, which returns a continuous EMBEDDING VECTOR, not a hash --
        `TypeError: unhashable type: 'numpy.ndarray'` when used as a dict key. The exact
        hash entry point is `arch_embed.exact_source()`; `both_source()` returns (hash,
        embed) from a single trace.
      * DiversityGate's constructor calls
        `functional_similarity._disable_db_prm_lookup()` to "skip the ~2-min per-model DB
        prm scan". Calling arch_embed directly skips that, so hashing the 50-model
        reference corpus was heading for ~100 minutes. It ran 10 minutes looking merely
        slow rather than misconfigured.

    THE VERDICT USES THE EXACT HASH ONLY -- the same standard every GA run has used.
    No launcher has ever passed --diversity-graded, and --cka-max (in
    launch_production_v2.sh) is inert: DiversityGate's own signature comment says the
    CKA-era args are "accepted but unused by the WL gate". So the GA accepts anything that
    is not a literal graph clone, and screening LLM candidates at cosine 0.95 would hold
    them to a STRICTER bar than the GA's own output -- the two would not be comparable.

    graded=True is still passed, but only so `both_source()` returns (hash, embed) from a
    single trace: the cosine is reported as ADVISORY context (how close the candidate sits
    to the existing corpus) and never decides accept/reject. Exact-hash matching cannot see
    "same architecture, rescaled widths" -- those sit at cosine >=0.985 and pass -- so the
    advisory column is what reveals that, without changing the standard.
    """
    from diversity_gate import DiversityGate
    return DiversityGate(store_path=None, log=lambda *a: None,
                         graded=graded, cos_max=cos_max)


def fingerprint(gate, code):
    """(exact_hash, embed) or (None, err)."""
    try:
        h, e = gate._fingerprint(code)
        return (h, e), None
    except Exception as ex:
        return (None, None), f"{type(ex).__name__}: {ex}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?",
                    default="/a/mm/sweep_out/llm_gen/run")
    ap.add_argument("--cap", type=float, default=CAP)
    a = ap.parse_args()

    import edge_cost as EC
    import numpy as np
    gate = _make_gate(graded=True, cos_max=0.95)

    def cos(a, b):
        a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
        d = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / d) if d else 0.0

    # Reference corpus: everything already in ab/nn/nn, so "novel" means novel against the
    # models we already have, not merely against the other 8 candidates in this batch.
    nd = next((d for d in ("/a/mm/nn-dataset/ab/nn/nn",
                           "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/ab/nn/nn")
               if os.path.isdir(d)), None)
    known, ref_embeds = {}, []
    if nd:
        files = sorted(glob.glob(os.path.join(nd, "Denoise*.py")) +
                       glob.glob(os.path.join(nd, "LiteDenoise*.py")))
        files = [f for f in files
                 if not re.search(r"-[0-9a-f]{16,}\.py$", os.path.basename(f))]
        for fp in files:
            nm = os.path.basename(fp)[:-3]
            (h, e), err = fingerprint(gate, open(fp).read())
            if h is None:
                continue
            known.setdefault(h, nm)
            if e is not None:
                ref_embeds.append((nm, e))
    print(f"reference corpus: {len(known)} distinct graph hashes "
          f"({len(ref_embeds)} embeds) from {nd}\n")

    rows = []
    for cdir in sorted(glob.glob(os.path.join(a.run_dir, "cand", "*"))):
        cid = os.path.basename(cdir)
        arm = "seeded" if cid.endswith("seeded") else "cold"
        src = os.path.join(cdir, "candidate.py")
        if not os.path.exists(src):
            rows.append((cid, arm, None, None, "NO CODE (parse)"))
            continue
        code = open(src).read()
        try:
            c = EC.measure_cost(code)
        except Exception as e:
            rows.append((cid, arm, None, None, f"BUILD FAIL: {type(e).__name__}"))
            continue
        if not c.get("ok"):
            rows.append((cid, arm, None, None, f"BUILD FAIL: {c.get('err')}"))
            continue
        macs = c["macs"] / 1e6
        (h, e), herr = fingerprint(gate, code)
        near = ""
        if e is not None and ref_embeds:
            nm, cs = max(((n, cos(e, v)) for n, v in ref_embeds), key=lambda t: t[1])
            # ADVISORY ONLY -- does not gate. The GA accepts on exact-hash novelty, so
            # rejecting here at cosine 0.95 would judge LLM output more harshly than the
            # GA's own. Flagged, not rejected, so the near-duplicate rate is still visible.
            near = f"  [nearest {nm} cos={cs:.3f}]"
            if cs >= 0.95:
                near += " NEAR-DUP(advisory)"
        if h is None:
            verdict = f"MAC OK, hash unavailable ({herr})"
        elif macs > a.cap:
            verdict = f"OVER CAP ({macs / a.cap:.1f}x)"
        elif h in known:
            verdict = f"CLONE of {known[h]}"
        else:
            verdict = "ACCEPT"
            known[h] = cid
            if e is not None:
                ref_embeds.append((cid, e))
        rows.append((cid, arm, macs, h, verdict + near))

    print(f"{'candidate':<16}{'arm':<8}{'MMac':>9}  verdict")
    print("-" * 68)
    for cid, arm, macs, _h, v in rows:
        m = f"{macs:.0f}" if macs is not None else "-"
        print(f"{cid:<16}{arm:<8}{m:>9}  {v}")

    print()
    for arm in ("cold", "seeded"):
        sub = [r for r in rows if r[1] == arm]
        acc = [r for r in sub if r[4].startswith("ACCEPT")]
        print(f"{arm:<8} {len(acc)}/{len(sub)} accepted   "
              + "  ".join(f"{k}={sum(1 for r in sub if r[4].startswith(k))}"
                          for k in ("NO CODE", "BUILD FAIL", "OVER CAP", "CLONE", "ACCEPT")))
    out = os.path.join(a.run_dir, "screen.json")
    with open(out, "w") as f:
        json.dump([{"id": r[0], "arm": r[1], "macs": r[2], "verdict": r[4]} for r in rows],
                  f, indent=2)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    raise SystemExit(main())
