"""
Rebuild the LEMUR DB to a clean baseline for the layerwise-LR re-evaluation.

Steps:
  1. Delete the existing DB file.
  2. init_population() — repopulates vanilla architecture baselines (and the
     bundled llr* rows) from the packaged dataset.
  3. Wipe every llr/llr2/llr3 row so the subsequent brute-force evaluation pass
     produces all results under identical conditions (no stale, dedup-skewed,
     or mixed-epoch entries left behind).

Vanilla baselines (plain arch names) are preserved — they are needed as the
"before" side of the vanilla->llr training pairs.

Usage:
    python -m ab.gpt.brute.llr.rebuild_db
"""

import os
import sqlite3
from pathlib import Path


def _db_path() -> str:
    for p in (Path('/a/mm/db/ab.nn.db'),
              Path(__file__).resolve().parents[4] / 'db' / 'ab.nn.db'):
        if p.parent.exists():
            return str(p)
    raise RuntimeError("Cannot resolve DB path")


def main():
    db = _db_path()
    if os.path.exists(db):
        os.remove(db)
        print(f"removed existing DB: {db}")

    from ab.nn.util.db.Write import init_population
    init_population()
    print("init_population() complete")

    con = sqlite3.connect(db)
    cur = con.cursor()
    before = cur.execute("SELECT COUNT(*) FROM stat WHERE nn LIKE 'llr%'").fetchone()[0]
    cur.execute("DELETE FROM stat WHERE nn LIKE 'llr%'")
    cur.execute("DELETE FROM nn   WHERE name LIKE 'llr%'")
    con.commit()
    vanilla = cur.execute("SELECT COUNT(DISTINCT nn) FROM stat").fetchone()[0]
    con.close()
    print(f"wiped {before} bundled llr stat rows; {vanilla} distinct non-llr models remain")


if __name__ == '__main__':
    main()
