"""Separate the models that actually denoise from the ones that do not.

THE NUMBER THIS TURNS ON. Returning the noisy validation image UNCHANGED scores 33.02 dB
under the project's own metric (Y-channel PSNR, all 50 centre crops) -- measured with a
harness verified to reproduce a stored DB value exactly (34.54 = 34.54, 0 missing keys).
Nothing in the corpus had ever recorded that floor, so a stored 33.7 dB read as a
respectable result when it is in fact +0.69 dB over doing nothing, and a stored 32.8 dB read
as "slightly below average" when it is a model that DEGRADES the image it is given.

WHY THIS MARKS RATHER THAN DELETES. Two reasons, and the second is the one that matters.

  1. A failing run is EVIDENCE. "This architecture was searched and it does not work" is a
     real result; deleting it makes the corpus look like every architecture we tried
     succeeded, and destroys the negative examples that make a search corpus honest.
  2. The threshold is a MEASUREMENT, and measurements get revised. 33.02 dB is specific to
     this validation set and this metric. Deleting rows acts on it irreversibly; a column
     lets the cut be re-drawn when the number is re-measured, which today has already
     happened twice.

So: a `beats_noop` column plus exported manifests. Consumers filter; nothing is lost.

FAILURE IS NOT ONE CATEGORY, and lumping it together hides the more interesting half. The
excluded set splits into models that crashed or diverged into noise (single-digit dB) and
models that trained fine and simply learned to do slightly less than nothing. Only the second
group is a statement about the architecture; the first is a statement about the run.

  python3 filter_noop.py --noop 33.02            # report + write manifests
  python3 filter_noop.py --noop 33.02 --mark     # also add/refresh the DB column
"""
import argparse
import collections
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import naming  # noqa: E402  (needs HERE on sys.path first)

TASK = "img-denoising"


def arm_of(name):
    # matches_family accepts 'LLM-' and the legacy 'LLM_' alike; arm_of picks up
    # the bare `g1_0000_seeded_*` candidate-id names imported before prefixes
    # existed. Both are needed -- the DB spans three naming vintages.
    if naming.matches_family(name, "LLM-") or naming.arm_of(name):
        return "LLM"
    if name.startswith(("DenoiseGA-", "DnSeedGA-", "DenoiseSkip-", "DnSkip-",
                        "DnNewFam-", "DnSearch-")):
        return "GA"
    return "hand-written"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/db/ab.nn.db")
    ap.add_argument("--noop", type=float, default=33.02,
                    help="no-op baseline in dB; a model at or below this denoises nothing")
    ap.add_argument("--broken-below", type=float, default=25.0,
                    help="below this the run failed, it is not an architecture verdict")
    ap.add_argument("--mark", action="store_true", help="write the beats_noop column")
    ap.add_argument("--outdir",
                    default="/shared/ssd/home/b-b-ahmed/thesis/sweep_out/manifest")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT nn, MAX(accuracy)*48 FROM stat WHERE task=? AND accuracy IS NOT NULL "
        "GROUP BY nn", (TASK,)).fetchall()
    con.close()

    keep = [(n, d) for n, d in rows if d > a.noop]
    drop = [(n, d) for n, d in rows if d <= a.noop]
    broken = [(n, d) for n, d in drop if d < a.broken_below]
    useless = [(n, d) for n, d in drop if d >= a.broken_below]

    print(f"=== no-op filter | baseline {a.noop:.2f} dB | {len(rows)} models ===")
    print(f"  KEEP  beats the baseline          {len(keep):5d}  ({len(keep)/len(rows):5.1%})")
    print(f"  DROP  at or below it              {len(drop):5d}  ({len(drop)/len(rows):5.1%})")
    print(f"    of which FAILED RUNS (<{a.broken_below:.0f} dB) {len(broken):5d}"
          f"   -- crashed/diverged, says nothing about the architecture")
    print(f"    of which trained-but-useless    {len(useless):5d}"
          f"   -- ran fine, learned to do less than nothing")

    print(f"\n=== dropped, by arm ===")
    da = collections.Counter(arm_of(n) for n, _ in drop)
    ta = collections.Counter(arm_of(n) for n, _ in rows)
    for arm in sorted(ta):
        print(f"  {arm:<14} {da.get(arm,0):4d} of {ta[arm]:5d}  ({da.get(arm,0)/ta[arm]:5.1%})")

    if useless:
        print(f"\n=== trained-but-useless, worst 10 (these are real architecture results) ===")
        for n, d in sorted(useless, key=lambda r: r[1])[:10]:
            print(f"  {d:6.2f} dB  ({d - a.noop:+.2f})  {n[:52]}")

    os.makedirs(a.outdir, exist_ok=True)
    kp = os.path.join(a.outdir, "models_beating_noop.txt")
    dp = os.path.join(a.outdir, "models_below_noop.txt")
    with open(kp, "w") as f:
        f.write("\n".join(n for n, _ in sorted(keep, key=lambda r: -r[1])) + "\n")
    with open(dp, "w") as f:
        for n, d in sorted(drop, key=lambda r: r[1]):
            f.write(f"{n}\t{d:.3f}\t"
                    f"{'failed-run' if d < a.broken_below else 'below-noop'}\n")
    js = os.path.join(a.outdir, "noop_filter.json")
    with open(js, "w") as f:
        json.dump({"noop_db": a.noop, "broken_below": a.broken_below,
                   "total": len(rows), "keep": len(keep), "drop": len(drop),
                   "failed_runs": len(broken), "trained_but_useless": len(useless),
                   "dropped": {n: d for n, d in sorted(drop, key=lambda r: r[1])}},
                  f, indent=2)
    print(f"\nwrote {kp}  ({len(keep)} names -- use this as the fine-tuning allowlist)")
    print(f"wrote {dp}  ({len(drop)} names + dB + reason)")
    print(f"wrote {js}")

    if not a.mark:
        print("\n(no DB write; pass --mark to add the beats_noop column)")
        return 0

    # Under LEMUR's own writer lock: GA processes write this NFS-hosted file continuously.
    from merge_shard_dbs import lemur_write_lock
    lock = lemur_write_lock(a.db)
    lock.__enter__()
    conn = sqlite3.connect(a.db, timeout=180)
    cur = conn.cursor()
    try:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(nn)")]
        if "beats_noop" not in cols:
            cur.execute("ALTER TABLE nn ADD COLUMN beats_noop INTEGER")
            print("  added column nn.beats_noop")
        cur.execute("BEGIN IMMEDIATE")
        for n, d in rows:
            cur.execute("UPDATE nn SET beats_noop=? WHERE name=?",
                        (1 if d > a.noop else 0, n))
        integrity = cur.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            conn.rollback()
            print(f"ROLLBACK: integrity_check {integrity}")
            return 1
        conn.commit()
        k = cur.execute("SELECT COUNT(*) FROM nn WHERE beats_noop=1").fetchone()[0]
        z = cur.execute("SELECT COUNT(*) FROM nn WHERE beats_noop=0").fetchone()[0]
        u = cur.execute("SELECT COUNT(*) FROM nn WHERE beats_noop IS NULL").fetchone()[0]
        print(f"  marked: {k} beat the baseline, {z} do not, {u} unscored")
    except Exception as e:
        conn.rollback()
        print(f"ROLLBACK on {type(e).__name__}: {e}")
        return 1
    finally:
        conn.close()
        lock.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
