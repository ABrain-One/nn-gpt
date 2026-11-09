"""Import the screened refab / llmprod survivors into the LEMUR DB as 1-epoch rows.

WHY THIS IS A SEPARATE SCRIPT FROM run3_db_import.py. That one walks a single cycle's eval
directory and imports every scored candidate. This pool spans three run directories with two
different on-disk layouts, and it is pre-filtered on two conditions that matter more than
convenience:

    ep1 > 33.021         beats the no-op (returning the noisy input unchanged)
    arch_uid2 unseen     not structurally identical to anything already in the corpus

So it consumes an explicit manifest (built by the caller, one entry per model) rather than a
directory, and it does no filtering of its own beyond a name collision check. The manifest is
the record of what was selected and why; regenerate it rather than editing it.

WHAT THESE ROWS ARE, STATED PLAINLY SO NOBODY MISREADS THEM LATER. They are 1-epoch screens.
That is a real limitation and it is also existing practice: 91 models in the corpus already have
epoch_max=1 as their only measurement, 89 of them the LLM_R3c* set imported by run3_db_import.py
under exactly this protocol. It is NOT the 5-epoch protocol the 2,049 DenoiseGA models use, so
these rows are not directly rank-comparable with those.

AND THE HONEST CAVEAT ON THE SELECTION BAR. "Beats the no-op" is a weaker filter than it sounds:
the measured per-run noise is 0.207 dB (the confirmation run; the 0.14 dB used earlier was an
assumption), and only 6 of these 96 clear 33.021 + 0.207. The rest beat doing nothing by less
than one standard deviation, which means a re-run could put them below it. They are imported for
corpus breadth and for the preflight deduper -- which reads nn.arch_uid and cannot skip what it
has never seen -- not as quality claims. Anyone ranking models should filter on dB, not on
membership.

Weights are not kept; the corpus has never stored weights for any model.

Run in-container, or with an interpreter that can import ab.nn:
  python refab_db_import.py --manifest .../import_manifest.json --dry-run
  python refab_db_import.py --manifest .../import_manifest.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import naming  # noqa: E402

NOOP_DB = 33.021
SD_DB = 0.207
PSNR_SCALE = 48.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    print(f"manifest: {len(man)} models")

    # Fail loudly on a manifest that violates either invariant, rather than importing a mess and
    # discovering it in the corpus. Both are cheap to check and expensive to undo.
    bad = [m["name"] for m in man if not naming.is_legal(m["name"])]
    if bad:
        sys.exit(f"FATAL: {len(bad)} name(s) contain '_' and would break the stat-dir split: {bad[:5]}")
    if len({m["arch_uid"] for m in man}) != len(man):
        sys.exit("FATAL: manifest contains duplicate arch_uid2 values")
    missing = [m["name"] for m in man if not os.path.isfile(m["code_path"])]
    if missing:
        sys.exit(f"FATAL: {len(missing)} source file(s) missing: {missing[:5]}")

    import ab.nn.util.db.Write as W
    from ab.nn.util.db.Init import sql_conn, close_conn
    from ab.nn.util.Util import uuid4

    # The recipe these were screened under. Must match the corpus standard or the rows are not
    # joinable with anything; it is the same dict run3_db_import.py uses.
    base_prm = {"lr": 1e-4, "batch": 16, "num_workers": 6, "transform": "denoise"}
    prm_uid = uuid4(base_prm)
    sub_config = ("img-denoising", "denoise", "psnr")

    conn, cursor = sql_conn()
    already = {r[0] for r in cursor.execute("SELECT name FROM nn").fetchall()}
    known_uid = {r[0] for r in cursor.execute(
        "SELECT DISTINCT arch_uid FROM nn WHERE arch_uid IS NOT NULL").fetchall()}

    imported = skipped = 0
    for m in man:
        if m["name"] in already:
            skipped += 1
            continue
        # Re-check against the live DB, not just the manifest: the manifest may have been built
        # against an older snapshot, and importing a structural duplicate would corrupt the one
        # invariant the corpus actually maintains.
        if m["arch_uid"] in known_uid:
            print(f"  SKIP {m['name']}: arch_uid2 already in corpus")
            skipped += 1
            continue
        if a.dry_run:
            print(f"  would import {m['name']}  ep1={m['ep1']:.3f} dB  [{m['source']}]")
            imported += 1
            continue
        acc = m["ep1"] / PSNR_SCALE
        trial = dict(base_prm, uid=prm_uid, accuracy=acc,
                     best_accuracy=acc, best_epoch=1, duration=None)
        W.code_to_db(cursor, "nn", code=open(m["code_path"]).read(), force_name=m["name"])
        W.save_stat(sub_config + (m["name"], 1), trial, cursor)
        already.add(m["name"])
        known_uid.add(m["arch_uid"])
        imported += 1

    if not a.dry_run:
        close_conn(conn)

    strong = sum(1 for m in man if m["ep1"] > NOOP_DB + SD_DB)
    print(f"\n{'would import' if a.dry_run else 'imported'} {imported}, skipped {skipped}")
    print(f"  of these, clearing the {SD_DB} dB noise floor: {strong}/{len(man)} "
          f"-- the rest are within noise of the no-op and are breadth, not quality")
    if not a.dry_run:
        print("  NEXT: backfill_arch_uid.py to populate nn.arch_uid for the new rows, then "
              "export_db_stats.py to write their stat dirs for upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
