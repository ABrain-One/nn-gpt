"""Pool the sharded 1-epoch screen results and report what the 516 LLM candidates actually did.

Three questions, in order of how much they matter:

1. WHAT SURVIVES AT ALL. Every candidate is one of: screen-fail (raised), trained but scored
   low, or trained and scored well. "Parseable code" was 94% at generation time; the smoke
   test showed that number is a large overestimate of "runs".

2. WHERE TO PUT THE GATE. The 32.7 default is the measured floor that retains all six
   canonical families finishing >= 35 dB after 50 epochs -- but that was calibrated on 18
   HAND-WRITTEN models spanning 0.6 dB. LLM output spans far more, and the one smoke
   survivor screened at 31.5, i.e. below it. This prints the survivor count at several
   candidate gates so the production gate is chosen from 516 measurements instead of 18.

   Keep in mind what the screen can and cannot do: ep1 -> ep50 rank correlation on the
   canonical corpus is weak (rho=+0.263). It is a FLOOR for "not broken", never a ranking.

3. COLD vs SEEDED, AND STRATEGY vs STRATEGY. This is the professor's comparison. It is only
   meaningful because both arms share the same TASK/CONSTRAINTS/INTERFACE/OUTPUT blocks by
   construction. Candidates repaired by repair_device_attr.py are counted separately: the
   missing self.device line was a defect in MY prompt, not a property of the arm, and the
   cold arm carried 122 of the 125. Quoting a raw cold-vs-seeded yield that includes them
   measures my bug.

  python analyse_screen.py --eval-dir /a/mm/sweep_out/llm_eval2
"""
import argparse
import glob
import json
import os
import re
import statistics as st

GATES = [30.0, 31.0, 31.5, 32.0, 32.7, 33.0, 33.5, 34.0]


def load(eval_dir):
    recs = []
    # Both names: a sharded screen writes results_shard<N>.json, an unsharded one (the
    # evolve loop, which must own the DB and so cannot shard) writes plain results.json.
    # Globbing only the sharded name made analyse_screen exit "no results" on every
    # generation of the evolve loop while the file sat right there.
    paths = (sorted(glob.glob(os.path.join(eval_dir, "results_shard*.json")))
             + sorted(glob.glob(os.path.join(eval_dir, "results.json"))))
    for p in paths:
        try:
            recs += json.load(open(p))
        except Exception as e:
            print(f"  WARN unreadable {p}: {e}")
    return recs


def arm_of(cid):
    return "cold" if "_cold_" in cid else ("seeded" if "_seeded_" in cid else "?")


def strat_of(cid):
    m = re.match(r"^\d+_(cold|seeded)_(.+)$", cid)
    return m.group(2) if m else "?"


def repaired_ids(gen_dir):
    """ids whose candidate.py we patched -- meta.json carries the marker."""
    out = set()
    for p in glob.glob(os.path.join(gen_dir, "*", "run", "cand", "*", "meta.json")):
        try:
            if "_repaired" in json.load(open(p)):
                out.add(os.path.basename(os.path.dirname(p)))
        except Exception:
            pass
    return out


def block(recs, title, rep=frozenset()):
    n = len(recs)
    if not n:
        print(f"\n=== {title} === (none)")
        return
    ok = [r for r in recs if r.get("ep1") is not None]
    fail = [r for r in recs if r.get("ep1") is None]
    print(f"\n=== {title} | n={n} ===")
    print(f"  ran            {len(ok):4d}  ({len(ok)/n:6.1%})")
    print(f"  screen-fail    {len(fail):4d}  ({len(fail)/n:6.1%})")
    if rep:
        r_in = [r for r in recs if r["id"] in rep]
        if r_in:
            r_ok = sum(1 for r in r_in if r.get("ep1") is not None)
            print(f"  (of which device-repaired: {len(r_in)}, {r_ok} now run)")
    if not ok:
        return
    d = sorted(r["ep1"] for r in ok)
    print(f"  ep1 dB         min {d[0]:6.2f}  p25 {d[len(d)//4]:6.2f}  "
          f"med {st.median(d):6.2f}  p75 {d[3*len(d)//4]:6.2f}  max {d[-1]:6.2f}")
    for g in GATES:
        k = sum(1 for v in d if v >= g)
        print(f"    >= {g:4.1f} dB   {k:4d}  ({k/n:6.1%} of all, {k/len(ok):6.1%} of ran)")


def fail_reasons(recs, topn=8):
    c = {}
    for r in recs:
        if r.get("ep1") is not None:
            continue
        e = (r.get("error") or "unknown")
        e = re.sub(r"^screen: ", "", e)
        # collapse the varying tensor sizes / channel counts into one bucket per error class
        key = re.sub(r"\d+", "N", e)[:90]
        c[key] = c.get(key, 0) + 1
    print("\n=== top screen-fail causes ===")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:topn]:
        print(f"  {v:4d}  {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="/a/mm/sweep_out/llm_eval2")
    ap.add_argument("--gen-dir", default="/a/mm/sweep_out/llm_gen2")
    ap.add_argument("--write-survivors", default="",
                    help="write ids >= --gate to this file, for the stage-2 full train")
    ap.add_argument("--gate", type=float, default=32.7)
    a = ap.parse_args()

    recs = load(a.eval_dir)
    if not recs:
        raise SystemExit(f"no results.json / results_shard*.json under {a.eval_dir}")
    rep = repaired_ids(a.gen_dir)
    print(f"pooled {len(recs)} records from {a.eval_dir}   "
          f"(device-repaired candidates: {len(rep)})")

    block(recs, "ALL", rep)
    for arm in ("cold", "seeded"):
        sub = [r for r in recs if arm_of(r["id"]) == arm]
        block(sub, f"arm = {arm}", rep)
        # arm figures excluding my prompt bug, which was 122/125 cold
        clean = [r for r in sub if r["id"] not in rep]
        if len(clean) != len(sub):
            block(clean, f"arm = {arm}  [excluding device-repaired]", rep)

    print("\n=== by strategy (ran / total, median ep1) ===")
    strats = {}
    for r in recs:
        strats.setdefault(strat_of(r["id"]), []).append(r)
    for s, rs in sorted(strats.items()):
        ok = [r["ep1"] for r in rs if r.get("ep1") is not None]
        med = f"{st.median(ok):6.2f}" if ok else "     -"
        pas = sum(1 for v in ok if v >= a.gate)
        print(f"  {s:<22} {len(ok):4d}/{len(rs):<4d}  med {med}   >= {a.gate}: {pas}")

    fail_reasons(recs)

    if a.write_survivors:
        ids = sorted(r["id"] for r in recs
                     if r.get("ep1") is not None and r["ep1"] >= a.gate)
        with open(a.write_survivors, "w") as f:
            f.write("\n".join(ids) + ("\n" if ids else ""))
        print(f"\nwrote {len(ids)} survivor ids (>= {a.gate} dB) to {a.write_survivors}")


if __name__ == "__main__":
    raise SystemExit(main())
