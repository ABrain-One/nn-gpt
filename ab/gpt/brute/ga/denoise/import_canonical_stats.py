"""Additive import of canonical per-epoch stat JSONs into the LEMUR DB, for an ALLOW-LIST
of models only.

Why not json_train_to_db(): that reloads EVERY sub-config directory and is the documented
way to lose uncommitted models. This runs the exact same official code path -- Write.py's
populate_code_table + save_stat, trial for trial -- but only for models named on the
command line that are NOT already in the nn table. Nothing is deleted, nothing existing is
touched, and a model already present is skipped whole.

Run in-container (needs the ab.nn environment and the live DB):

  python import_canonical_stats.py --models-file names.txt          # one name per line
  python import_canonical_stats.py --glob 'DenoiseGA-*'             # by nn/ file glob
"""
import argparse
import glob as globmod
import json
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-file", default=None, help="file with one model name per line")
    ap.add_argument("--glob", default=None, help="glob over ab/nn/nn to pick models")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from ab.nn.util.Const import stat_train_dir, nn_dir
    import ab.nn.util.db.Write as W
    from ab.nn.util.db.Init import sql_conn, close_conn
    from ab.nn.util.Util import conf_to_names

    names = set()
    if a.models_file:
        with open(a.models_file) as f:
            names |= {ln.strip() for ln in f if ln.strip()}
    if a.glob:
        names |= {os.path.basename(p)[:-3]
                  for p in globmod.glob(str(nn_dir / (a.glob + ".py")))}
    if not names:
        print("no models named"); return 1

    conn, cursor = sql_conn()
    already = {r[0] for r in cursor.execute("SELECT name FROM nn").fetchall()}
    todo = sorted(n for n in names if n not in already)
    print(f"{len(names)} named, {len(names) - len(todo)} already in DB, {len(todo)} to import")

    imported = trials = skipped = 0
    for nn in todo:
        sub_dirs = [d for d in globmod.glob(str(stat_train_dir / ("*_" + nn)))
                    if os.path.isdir(d)]
        # a name suffix must match exactly, not another model with this name as prefix
        sub_dirs = [d for d in sub_dirs if conf_to_names(os.path.basename(d))[3] == nn]
        if not sub_dirs:
            skipped += 1
            continue
        if not (nn_dir / (nn + ".py")).exists():
            print(f"  SKIP {nn}: no code file"); skipped += 1
            continue
        if a.dry_run:
            n_ep = sum(len(os.listdir(d)) for d in sub_dirs)
            print(f"  would import {nn}: {len(sub_dirs)} config(s), {n_ep} epoch files")
            imported += 1
            continue
        for d in sub_dirs:
            sub_config = conf_to_names(os.path.basename(d))
            _, _, metric, _ = sub_config
            for epoch_file in sorted(os.listdir(d)):
                if not epoch_file.endswith(".json"):
                    continue
                epoch = int(epoch_file[:-5])
                try:
                    with open(os.path.join(d, epoch_file), encoding="utf-8", errors="replace") as f:
                        for trial in json.load(f):
                            W.populate_code_table("nn", cursor, name=nn)
                            for m in metric.split(","):
                                W.populate_code_table("metric", cursor, name=m.strip())
                            if trial.get("transform"):
                                W.populate_code_table("transform", cursor, name=trial["transform"])
                            W.save_stat(sub_config + (epoch,), trial, cursor)
                            trials += 1
                except Exception as e:
                    print(f"  WARN {nn} {os.path.basename(d)}/{epoch_file}: {e}", file=sys.stderr)
        imported += 1
    if not a.dry_run:
        close_conn(conn)
    print(f"imported {imported} models ({trials} stat rows), {skipped} skipped (no stats/code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
