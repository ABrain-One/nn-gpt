"""Add an ARCHITECTURAL id to the nn table, alongside the existing lexical one.

THE PROPOSAL THIS IMPLEMENTS (phase 1 of 2). LEMUR keys `nn` on uuid4(code) = md5 of the
source with whitespace stripped -- a LEXICAL identity, so the same network written differently
becomes two dataset rows. arch_embed.exact() already computes the right thing: a torch.fx
trace hashed with Weisfeiler-Lehman over behavioural node labels, vocabulary-free, hence
invariant to naming, comments, statement order and helper-class structure.

WHY A COLUMN AND NOT A RE-KEY. `nn.name` is a primary key that `stat.nn` references across
~1800 trial rows, and 10 of 829 models are UNTRACEABLE (torch.fx cannot follow data-dependent
control flow), so a re-key needs a decided fallback rather than a silent one. Adding a column
is additive, reversible, and lets the collapse rate be observed on live data before anything
depends on it. Phase 2 -- if the numbers justify it -- changes the key.

WHAT THE MEASUREMENT SAID (829 models, img-denoising): 819 traceable, 753 distinct
architectures, 66 isomorphic duplicates in 38 groups. But the breakdown matters more than the
headline:
    6 groups  F/non-F seed pairs      -- SAME architecture, different training recipe. By
                                         design, not a defect.
   19 groups  generator rediscovers a hand-written seed -- arguably validation of the search.
    3 groups  hand-written seeds isomorphic to each other.
   16 groups  generator duplicating a generator -- the genuine defect, and mostly the LLM,
              whose largest group is SIX models that are one architecture. The GA's
              diversity_gate already hashes on exactly this WL id, which is why so few of
              these are within a single GA run.

WRITES ARE TAKEN UNDER LEMUR'S OWN WRITER LOCK (see merge_shard_dbs.lemur_write_lock), because
GA processes write this same NFS-hosted file continuously.

  python add_arch_uid.py --db /a/mm/nn-dataset/db/ab.nn.db --uids .../arch_uids.json --dry-run
"""
import argparse
import datetime
import json
import os
import shutil
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from merge_shard_dbs import lemur_write_lock          # noqa: E402

COL = "arch_uid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--uids", default="/a/mm/sweep_out/manifest/arch_uids.json",
                    help="output of arch_uid_audit.py; {arch_hash: [model names]}")
    ap.add_argument("--backup-dir", default="/a/mm/sweep_out/db_backup")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    d = json.load(open(a.uids))
    name_to_arch = {}
    for h, names in d["arch_to_names"].items():
        for n in names:
            name_to_arch[n] = h
    untraceable = set(d.get("untraceable", {}))
    print(f"loaded {len(name_to_arch)} name->architecture mappings "
          f"({len(untraceable)} untraceable)")

    if not a.dry_run:
        os.makedirs(a.backup_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = os.path.join(a.backup_dir, f"ab.nn.pre-archuid.{ts}.db")
        shutil.copy2(a.db, bak)
        if os.path.getsize(bak) != os.path.getsize(a.db):
            print("FATAL: backup size mismatch; not touching the DB")
            return 2
        print(f"backup: {bak}  ({os.path.getsize(bak)/1e6:.1f} MB)")

    lock = lemur_write_lock(a.db)
    lock.__enter__()
    conn = sqlite3.connect(a.db, timeout=180)
    cur = conn.cursor()
    try:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(nn)")]
        print(f"nn columns before: {cols}")
        cur.execute("BEGIN IMMEDIATE")
        if COL not in cols:
            # ADD COLUMN is metadata-only in SQLite -- fast, and it cannot rewrite or drop
            # any existing row. Existing readers that SELECT by name are unaffected.
            cur.execute(f"ALTER TABLE nn ADD COLUMN {COL} TEXT")
            print(f"added column nn.{COL}")
        else:
            print(f"column nn.{COL} already present -- refreshing values")

        have = [r[0] for r in cur.execute("SELECT name FROM nn")]
        n_set = n_miss = 0
        for nm in have:
            h = name_to_arch.get(nm)
            if h is None:
                n_miss += 1
                continue
            cur.execute(f"UPDATE nn SET {COL} = ? WHERE name = ?", (h, nm))
            n_set += 1
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_nn_{COL} ON nn({COL})")

        # Report the collapse rate from the DB itself, not from the audit file, so what is
        # quoted is what was actually written.
        tot = cur.execute(f"SELECT COUNT(*) FROM nn WHERE {COL} IS NOT NULL").fetchone()[0]
        dis = cur.execute(f"SELECT COUNT(DISTINCT {COL}) FROM nn "
                          f"WHERE {COL} IS NOT NULL").fetchone()[0]
        integrity = cur.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"\n  rows with {COL}      {n_set}")
        print(f"  rows left NULL       {n_miss}  (untraceable, or not in the audit's task)")
        print(f"  distinct architectures {dis} of {tot} rows"
              + (f"  -> {tot-dis} isomorphic duplicates ({(tot-dis)/tot:.1%})" if tot else ""))
        print(f"  integrity_check: {integrity}")

        if integrity != "ok":
            conn.rollback()
            print("\nROLLBACK: integrity check failed")
            return 1
        if a.dry_run:
            conn.rollback()
            print("\nDRY RUN -- rolled back, schema unchanged")
            return 0
        conn.commit()
        print("\nCOMMITTED")
        print(f"\nFind duplicates any time with:\n"
              f"  SELECT {COL}, COUNT(*) c, GROUP_CONCAT(name) FROM nn\n"
              f"  WHERE {COL} IS NOT NULL GROUP BY {COL} HAVING c > 1 ORDER BY c DESC;")
    except Exception as e:
        conn.rollback()
        print(f"\nROLLBACK on {type(e).__name__}: {e}")
        return 1
    finally:
        conn.close()
        lock.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
