"""Fill nn.arch_uid for rows that do not have one yet. Runs after every merge.

WHY THIS IS NOT OPTIONAL. arch_uid is written by a batch audit, so it is a SNAPSHOT. Every
generation's merge inserts new models with arch_uid NULL, and preflight's --arch-dedupe-db
builds its "already known" set from that column. Without a backfill the set goes stale
immediately: generation 3 would be deduped against architectures known at generation 1, and
every duplicate of a generation-2 model would pass straight through while the summary line
still read "N already in dataset" and looked like it was working.

Incremental by design -- only NULL rows are computed -- so it costs seconds per generation
instead of re-tracing the whole corpus.

Writes under LEMUR's own writer lock, because GA processes write this same NFS-hosted file
continuously.

  python backfill_arch_uid.py --db /a/mm/nn-dataset/db/ab.nn.db
"""
import argparse
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from merge_shard_dbs import lemur_write_lock          # noqa: E402
# SWITCHED (2026-08-03) to arch_uid2: the source-only AST identity (v2, with per-class
# attribute scoping and slice ports). No model is built, so no torch is needed and the
# 24.6% of models fx cannot trace are identified rather than recorded as untraceable.
from arch_uid2 import arch_uid as _uid2               # noqa: E402


def of_source(src):
    """-> (uid, error), preserving the old interface's explicit-failure contract."""
    try:
        return _uid2(src), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recompute-all", action="store_true",
                    help="rehash EVERY row, not just NULL ones -- required once when the "
                         "identity function changes, or old and new hash spaces mix and "
                         "the dedup set silently behaves as empty")
    a = ap.parse_args()

    # REFUSE TO RUN WITHOUT TORCH, rather than write false negatives.
    #
    # arch_uid.of_source() catches every exception and returns (None, reason). That is
    # deliberate on its side -- a silent None is indistinguishable from "no duplicates", which
    # is how an earlier broken check looked correct. But this caller treated `h is None` as
    # "untraceable" regardless of the reason, so running on the login node (no torch) marked
    # 157 perfectly healthy models as having no architectural identity and reported it as a
    # property of the models. Re-run in a pod, 127 of the same 157 traced fine.
    #
    # The verdict has to depend on the model, never on where the script happens to run.
    # (The old torch guard is gone with the switch to arch_uid2: nothing is instantiated,
    # so the login-node/pod distinction no longer changes any verdict.)
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    cols = [r[1] for r in c.execute("PRAGMA table_info(nn)")]
    if "arch_uid" not in cols:
        print("FATAL: nn.arch_uid does not exist; run add_arch_uid.py first")
        return 2
    q = ("SELECT name, code FROM nn" if a.recompute_all
         else "SELECT name, code FROM nn WHERE arch_uid IS NULL")
    rows = c.execute(q + (f" LIMIT {a.limit}" if a.limit else "")).fetchall()
    c.close()
    print(f"{len(rows)} rows need an architectural id")
    if not rows:
        return 0

    done, failed = [], []
    for i, (name, code) in enumerate(rows, 1):
        h, err = of_source(code)
        (done if h else failed).append((name, h or err))
        if i % 25 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] ok={len(done)} untraceable={len(failed)}", flush=True)

    lock = lemur_write_lock(a.db)
    lock.__enter__()
    conn = sqlite3.connect(a.db, timeout=180)
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        for name, h in done:
            cur.execute("UPDATE nn SET arch_uid = ? WHERE name = ?", (h, name))
        if a.recompute_all:
            # a row that fails under the NEW identity must not keep its OLD hash --
            # that would leave two hash spaces mixed in one column
            for name, _err in failed:
                cur.execute("UPDATE nn SET arch_uid = NULL WHERE name = ?", (name,))
        integrity = cur.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            conn.rollback()
            print(f"ROLLBACK: integrity_check {integrity}")
            return 1
        conn.commit()
        tot = cur.execute("SELECT COUNT(*) FROM nn WHERE arch_uid IS NOT NULL").fetchone()[0]
        dis = cur.execute("SELECT COUNT(DISTINCT arch_uid) FROM nn "
                          "WHERE arch_uid IS NOT NULL").fetchone()[0]
        print(f"\n  filled {len(done)}, untraceable {len(failed)}")
        print(f"  corpus: {dis} distinct architectures across {tot} rows"
              + (f"  ({tot - dis} isomorphic duplicates)" if tot else ""))
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
