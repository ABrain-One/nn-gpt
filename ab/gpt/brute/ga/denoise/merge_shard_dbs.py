"""Merge per-shard LEMUR DBs back into the real one. SINGLE WRITER, INSERT-ONLY.

WHY SHARD DBS EXIST AT ALL. The 5-epoch train is 41% of an evolve generation and is pure
GPU, so it is the obvious thing to parallelise -- but it writes the LEMUR SQLite file, which
lives on NFSv3. SQLite's locking depends on POSIX advisory locks that NFSv3 does not
implement reliably, so concurrent writers do not block each other, they interleave and
corrupt. That is why screen_and_train.py refuses --num-shards>1 unless --screen-only.

The way out is not to make concurrent writes safe -- it is to not have them. Each shard
trains against a pod-local COPY of the DB, and this script folds the copies back in
afterwards from ONE process.

WHAT MAKES THIS SAFE TO POINT AT THE PRODUCTION DB

  INSERT-ONLY.        No UPDATE, no DELETE, no schema change. The worst failure mode is
                      surplus rows, never a lost model. In particular a model whose name
                      already exists with DIFFERENT code is reported as a collision and
                      skipped -- never overwritten -- because that would silently rewrite an
                      architecture someone already has results for.
  BACKUP FIRST.       A timestamped copy is taken and its size verified before the
                      transaction opens.
  ONE TRANSACTION.    BEGIN IMMEDIATE takes the write lock up front; everything commits or
                      nothing does.
  VERIFIED AFTER.     integrity_check plus a row-count reconciliation run BEFORE commit, and
                      any failure rolls back.
  --dry-run           Does the entire merge and then rolls back, printing exactly what would
                      have been inserted. Run it first.

THE prm TABLE NEEDS SPECIAL HANDLING. `nn` is keyed by name and `stat` by id, so INSERT OR
IGNORE dedupes both. `prm` has NO unique constraint -- only a non-unique index on
(uid, name, value) -- so OR IGNORE silently does nothing there, which is why the production
DB already carries 70 duplicate (uid, name) pairs. Deduping has to be explicit, or every
generation would multiply them by the shard count.

  python merge_shard_dbs.py --main /a/mm/nn-dataset/db/ab.nn.db \
      --shards /a/mm/sweep_out/shard_db/*.db --dry-run
"""
import argparse
import contextlib
import datetime
import glob
import os
import shutil
import sqlite3
import sys

try:
    import fcntl
except ImportError:
    fcntl = None


@contextlib.contextmanager
def lemur_write_lock(main_db):
    """Hold the SAME lock LEMUR's own writers hold, for as long as we write.

    ab/nn/util/db/Write.py wraps save_nn (and every other writer) in @_serialized_db_write,
    which takes an exclusive flock on <db_dir>/ab.nn.db.lock. This tool talked straight to
    sqlite3 and took none of it -- so a merge could land while a GA process was mid-write on
    the same NFS-hosted file. BEGIN IMMEDIATE alone does not save us: SQLite's own locking
    relies on POSIX advisory locks that NFSv3 implements unreliably, which is the whole
    reason the shard DBs exist. Taking LEMUR's lock makes this cooperate with every other
    writer in the project instead of racing them.

    Not optional and not best-effort: if the lock cannot be taken, we do not write.
    """
    lock_path = os.path.join(os.path.dirname(os.path.abspath(main_db)), "ab.nn.db.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    f = open(lock_path, "a", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()

# Only these three carry a trained model. Everything else in the schema (run, tflite, prun,
# nn_stat, nn_minhash, ...) is empty and written by other tooling; copying them blindly would
# be inventing data movement nobody asked for.
TABLES = ("nn", "prm", "stat")


def _counts(cur, prefix="main"):
    return {t: cur.execute(f"SELECT COUNT(*) FROM {prefix}.{t}").fetchone()[0] for t in TABLES}


def _check_readable(p):
    """A shard DB that is corrupt or truncated must be found HERE, not half way through the
    transaction against production."""
    if not os.path.exists(p):
        return f"missing"
    try:
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        r = c.execute("PRAGMA integrity_check").fetchone()[0]
        if r != "ok":
            return f"integrity_check: {r[:80]}"
        for t in TABLES:
            c.execute(f"SELECT COUNT(*) FROM {t}")
        c.close()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def merge_one(cur, sh, dry):
    """-> dict of rows inserted per table, plus any nn-name collisions.

    `sh` is the ATTACH alias, not a path: SQLite refuses ATTACH inside a transaction, so
    every shard is attached before BEGIN IMMEDIATE and all of them are folded in under one
    transaction. That is also what we want semantically -- a generation's shards land
    together or not at all, rather than leaving the DB holding two of four.
    """
    added, collisions = {}, []
    if True:
        # --- nn: new architectures only, by primary key `name` -----------------
        # A name present in both with different code is a genuine conflict. Report it; do
        # not resolve it. Silently keeping either version would make the DB disagree with
        # whichever run is quoted in the write-up.
        for (nm,) in cur.execute(
                f"SELECT s.name FROM {sh}.nn s JOIN main.nn m ON m.name = s.name "
                "WHERE s.code <> m.code"):
            collisions.append(nm)
        n = cur.execute(
            f"INSERT INTO main.nn (name, code, id) "
            f"SELECT s.name, s.code, s.id FROM {sh}.nn s "
            "WHERE s.name NOT IN (SELECT name FROM main.nn)").rowcount
        added["nn"] = max(n, 0)

        # --- prm: dedupe on (uid, name); the table has no unique constraint ----
        n = cur.execute(
            f"INSERT INTO main.prm (uid, name, value) "
            f"SELECT s.uid, s.name, s.value FROM {sh}.prm s "
            "WHERE NOT EXISTS (SELECT 1 FROM main.prm m "
            "                  WHERE m.uid = s.uid AND m.name = s.name)").rowcount
        added["prm"] = max(n, 0)

        # --- stat: new trial rows only, by primary key `id` --------------------
        n = cur.execute(
            "INSERT INTO main.stat (id, accuracy, epoch, duration, task, dataset, nn, "
            "                       transform, metric, prm) "
            "SELECT s.id, s.accuracy, s.epoch, s.duration, s.task, s.dataset, s.nn, "
            f"       s.transform, s.metric, s.prm FROM {sh}.stat s "
            "WHERE s.id NOT IN (SELECT id FROM main.stat)").rowcount
        added["stat"] = max(n, 0)
    return added, collisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--shards", nargs="+", required=True,
                    help="shard DB paths; globs are expanded")
    ap.add_argument("--backup-dir", default="/a/mm/sweep_out/db_backup")
    ap.add_argument("--dry-run", action="store_true",
                    help="do the whole merge, report it, then ROLL BACK")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the pre-merge copy (only for a throwaway --main)")
    a = ap.parse_args()

    shards = []
    for s in a.shards:
        shards += sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s]
    shards = [s for s in shards if os.path.abspath(s) != os.path.abspath(a.main)]
    if not shards:
        print("FATAL: no shard DBs matched")
        return 2

    print(f"main   : {a.main}")
    print(f"shards : {len(shards)}")
    bad = {s: _check_readable(s) for s in shards}
    bad = {s: v for s, v in bad.items() if v}
    if bad:
        # Merging a subset would produce a DB that silently disagrees with the run logs.
        for s, v in bad.items():
            print(f"  UNUSABLE {s}: {v}")
        print("FATAL: refusing to merge while any shard is unreadable")
        return 2

    if not os.path.exists(a.main):
        print(f"FATAL: main DB not found: {a.main}")
        return 2

    if not a.no_backup and not a.dry_run:
        os.makedirs(a.backup_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = os.path.join(a.backup_dir, f"ab.nn.{ts}.db")
        shutil.copy2(a.main, bak)
        if os.path.getsize(bak) != os.path.getsize(a.main):
            print("FATAL: backup size mismatch; aborting before touching the DB")
            return 2
        print(f"backup : {bak}  ({os.path.getsize(bak)/1e6:.1f} MB)")

    if len(shards) > 8:
        # SQLITE_MAX_ATTACHED defaults to 10 and atomicity across shards requires them all
        # attached at once. Merge in batches rather than silently dropping any.
        print(f"FATAL: {len(shards)} shards; merge at most 8 at a time")
        return 2

    # Everything from here until commit happens under LEMUR's writer lock.
    lock = lemur_write_lock(a.main)
    lock.__enter__()
    conn = sqlite3.connect(a.main, timeout=120)
    cur = conn.cursor()
    aliases = []
    for i, s in enumerate(shards):
        al = f"sh{i}"
        cur.execute(f"ATTACH DATABASE ? AS {al}", (f"file:{s}?mode=ro",))
        aliases.append(al)
    before = _counts(cur)
    print(f"\nbefore : " + "  ".join(f"{t}={before[t]}" for t in TABLES))

    total, all_coll = {t: 0 for t in TABLES}, []
    try:
        # IMMEDIATE, not deferred: take the write lock now so a second writer fails here
        # rather than half way through.
        cur.execute("BEGIN IMMEDIATE")
        for al, s in zip(aliases, shards):
            added, coll = merge_one(cur, al, a.dry_run)
            all_coll += coll
            for t in TABLES:
                total[t] += added[t]
            print(f"  {os.path.basename(s):<34} " +
                  "  ".join(f"+{t}={added[t]}" for t in TABLES) +
                  (f"  COLLISIONS={len(coll)}" if coll else ""))

        after = _counts(cur)
        # Reconcile BEFORE committing: if the deltas do not match what we inserted, some
        # other writer touched the file and the whole premise of single-writer is void.
        mismatch = [t for t in TABLES if after[t] - before[t] != total[t]]
        integrity = cur.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"\nafter  : " + "  ".join(f"{t}={after[t]}" for t in TABLES))
        print(f"delta  : " + "  ".join(f"+{total[t]}" for t in TABLES))
        print(f"integrity_check: {integrity}")
        if all_coll:
            print(f"\nname collisions (same name, DIFFERENT code) -- skipped, not overwritten:")
            for nm in all_coll[:10]:
                print(f"  {nm}")
            if len(all_coll) > 10:
                print(f"  ... and {len(all_coll)-10} more")

        if mismatch or integrity != "ok":
            print(f"\nROLLBACK: " +
                  (f"row-count mismatch on {mismatch}" if mismatch else "integrity failed"))
            conn.rollback()
            return 1
        if a.dry_run:
            conn.rollback()
            print("\nDRY RUN -- rolled back, nothing written")
            return 0
        conn.commit()
        print("\nCOMMITTED")
    except Exception as e:
        conn.rollback()
        print(f"\nROLLBACK on {type(e).__name__}: {e}")
        return 1
    finally:
        for al in aliases:
            try:
                cur.execute(f"DETACH DATABASE {al}")
            except Exception:
                pass
        conn.close()
        lock.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
