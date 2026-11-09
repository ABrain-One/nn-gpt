"""Import one run-3 cycle's screened models into the LEMUR DB as 1-epoch rows.

The screen stage is zero-DB-write by design, and run 3 no longer has a training stage --
without this, screened models exist only as files and, worse, the NEXT cycle's preflight
cannot dedupe against them (the gate reads nn.arch_uid). This importer runs after each
cycle: every screened candidate with a measured score becomes an nn row plus a stat row
under the standard recipe prm at epoch 1, via the official Write.py code path (additive,
same guarantees as import_canonical_stats.py). Weights are not kept -- the corpus never
stored weights for any model.

Run in-container (needs the ab.nn environment):
  python run3_db_import.py --eval-dir .../cycle1/eval --run-dir .../cycle1/run --cycle 1
"""
import argparse
import glob
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import naming  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import ab.nn.util.db.Write as W
    from ab.nn.util.db.Init import sql_conn, close_conn
    from ab.nn.util.Util import uuid4

    recs = []
    for p in (sorted(glob.glob(os.path.join(a.eval_dir, "results_shard*.json")))
              + sorted(glob.glob(os.path.join(a.eval_dir, "results.json")))):
        try:
            recs += json.load(open(p))
        except Exception as e:
            print(f"  WARN unreadable {p}: {e}")
    scored = [r for r in recs if r.get("ep1") is not None]
    print(f"{len(scored)} screened models with a measured score")
    if not scored:
        return 0

    # the corpus-standard recipe these were screened under
    base_prm = {"lr": 1e-4, "batch": 16, "num_workers": 6, "transform": "denoise"}
    prm_uid = uuid4(base_prm)

    conn, cursor = sql_conn()
    already = {r[0] for r in cursor.execute("SELECT name FROM nn").fetchall()}
    sub_config = ("img-denoising", "denoise", "psnr")

    imported = skipped = 0
    for r in scored:
        code_path = os.path.join(a.run_dir, "cand", r["id"], "candidate.py")
        if not os.path.isfile(code_path):
            print(f"  WARN {r['id']}: no candidate.py; skipped"); skipped += 1
            continue
        code = open(code_path).read()
        # Hyphen after LLM: '_' is LEMUR's stat-dir separator, so an underscored
        # model name breaks the 4-way split in Util.py:61. See naming.py.
        name = naming.check(
            f"LLM-R3c{a.cycle}-{hashlib.sha256(code.encode()).hexdigest()[:24]}")
        if name in already:
            skipped += 1
            continue
        if a.dry_run:
            print(f"  would import {name}  ep1={r['ep1']:.2f} dB"); imported += 1
            continue
        acc = r["ep1"] / 48.0
        trial = dict(base_prm, uid=prm_uid, accuracy=acc,
                     best_accuracy=acc, best_epoch=1, duration=None)
        W.code_to_db(cursor, "nn", code=code, force_name=name)
        W.save_stat(sub_config + (name, 1), trial, cursor)
        already.add(name)
        imported += 1

    if not a.dry_run:
        close_conn(conn)
    print(f"imported {imported} models as 1-epoch rows, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
