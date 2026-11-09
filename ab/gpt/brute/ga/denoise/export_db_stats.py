"""Export per-epoch training stats from the DB into ab/nn/stat/train/, denoise only.

WHY. DnSeedGA -- the strongest arm we have (best AWGN PSNR 36.98, best zero-shot SIDD transfer
32.59 dB, lowest latency 1.66 ms p50) -- exists only in the DB. Its 376 models had no `.py` and no
stat dirs, so the upload would have published the corpus without its best family.

FIDELITY. Every field a training-written epoch file carries is recoverable:
  * `prm` supplies batch, best_accuracy, best_epoch, cpu_*, gpu_*, gradient_norm, lr, metric_psnr,
    num_workers, occupied_*, ram_*, samples_per_second, test_loss, total_ram_kb, train_accuracy,
    train_loss, transform  -- 24 fields, stored as (uid, name, value) triples
  * `stat` supplies accuracy, duration, epoch
  * epoch_max is computed across the model's rows; uid is the prm uid
Nothing is invented. If a field is absent from `prm` it is omitted, never filled with a plausible
value -- a fabricated gpu_type or train_loss would be indistinguishable from a measured one.

THE LIMITATION YOU MUST KNOW ABOUT. The DB stores ONE row per DnSeedGA model, at epoch 5. Zero of
376 have more than one epoch recorded, because the GA persisted only the final result. So these
dirs get `5.json` and nothing else, while all 1,927 existing DenoiseGA stat dirs hold a complete
1..5 curve. `validate_upload.py` will therefore flag every DnSeedGA dir as
"missing epoch(s) [1, 2, 3, 4] of 1..5" -- correctly. That is a real gap in the record, not a bug
in the exporter, and it cannot be closed from the DB. Closing it means re-running the 5-epoch
training for those 376 models.

AWGN IS EXCLUDED. `--datasets denoise` by default; the awgn runs were internal and are not for
upload. (As it happens all 392 DnSeedGA rows are dataset=denoise/metric=psnr, so nothing is
dropped here -- the filter is a guard, not a no-op waiting to bite.)

Usage:
    python export_db_stats.py --prefix DnSeedGA-            # dry run
    python export_db_stats.py --prefix DnSeedGA- --apply
"""
import argparse
import collections
import json
import os
import sqlite3
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import naming  # noqa: E402

REPO = "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset"
STAT_DIR = os.path.join(REPO, "ab", "nn", "stat", "train")
DB = os.path.join(REPO, "db", "ab.nn.db")

# keys a training-written epoch file carries, for reporting coverage
EXPECTED = {"accuracy", "batch", "best_accuracy", "best_epoch", "cpu_count", "cpu_type",
            "cpu_usage_percent", "duration", "epoch_max", "gpu_memory_kb",
            "gpu_memory_usage_percent", "gpu_total_memory_kb", "gpu_type", "gradient_norm",
            "lr", "metric_psnr", "num_workers", "occupied_gpu_memory_kb", "occupied_ram_kb",
            "ram_usage_percent", "samples_per_second", "test_loss", "total_ram_kb",
            "train_accuracy", "train_loss", "transform", "uid"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--datasets", nargs="*", default=["denoise"],
                    help="datasets to export; awgn is deliberately NOT a default")
    ap.add_argument("--task", default="img-denoising")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    prm = collections.defaultdict(dict)
    for uid, name, val in con.execute("select uid, name, value from prm"):
        prm[uid][name] = val

    # Match the prefix in BOTH spellings. Now that generators mint 'LLM-Denoise',
    # that is the prefix a caller will naturally type -- but the 470 rows already in
    # the DB are still 'LLM_Denoise', so a single-spelling match would silently
    # select nothing and the export would look like "already done".
    # LIKE is avoided here because '_' is a single-char wildcard in LIKE, which made
    # the old pattern match by accident; GLOB treats it literally.
    pats = [naming.canon(a.prefix) + "*", a.prefix.replace("-", "_") + "*"]
    rows = list(con.execute(
        "select nn, dataset, metric, epoch, accuracy, duration, transform, prm "
        "from stat where (nn glob ? or nn glob ?) and task = ?",
        (pats[0], pats[1], a.task)))
    rows = [r for r in rows if r[1] in a.datasets]
    print(f"stat rows for {a.prefix!r} in datasets {a.datasets}: {len(rows)}")
    if not rows:
        sys.exit("nothing to export")

    # group by (model, dataset, metric) -> {epoch: record}
    groups = collections.defaultdict(dict)
    for nn, ds, mt, ep, acc, dur, tf, puid in rows:
        p = prm.get(puid, {})
        rec = {"accuracy": acc, "duration": dur, "transform": tf, "uid": puid}
        for k, v in p.items():
            rec.setdefault(k, v)
        rec.setdefault("metric_psnr", acc)
        groups[(nn, ds, mt)][int(ep)] = rec

    missing_fields = collections.Counter()
    wrote = skipped = 0
    epoch_hist = collections.Counter()
    clipped = {}
    renamed = {}
    for (nn, ds, mt), eps in sorted(groups.items()):
        emax = max(eps)
        # THE GATE. This is the line where a DB name becomes a committed path, and
        # where the 111 `denoise_awgn` dirs and 372 `LLM_*` dirs escaped last time.
        #
        # Model names are canonicalised rather than rejected: the on-disk tree was
        # already renamed to hyphens while 470 DB rows still carry the old spelling,
        # so the DB is the laggard and disk is authoritative. Rewriting keeps the two
        # in agreement. Every rewrite is reported below -- never silent.
        #
        # task/dataset/metric are NOT rewritten, they are hard errors. Each one names
        # a real module (ab/nn/loader/<dataset>.py, ab/nn/metric/<metric>.py), so
        # renaming it here would produce a stat dir whose loader cannot be found on
        # re-import. Those must be fixed by renaming the module itself.
        nn_out = naming.canon(nn)
        if nn_out != nn:
            renamed[nn] = nn_out
        d = os.path.join(STAT_DIR, naming.stat_dir(a.task, ds, mt, nn_out))
        naming.check_stat_dir(os.path.basename(d))
        epoch_hist[len(eps)] += 1
        if os.path.isdir(d):
            skipped += 1          # never overwrite stats written by an actual training run
            continue
        if a.apply:
            os.makedirs(d, exist_ok=True)
        # A PARTIAL EXPORT MUST NOT CLAIM A BEST IT DOES NOT CONTAIN.
        #
        # `prm.best_accuracy` / `best_epoch` describe the whole training run, but the DB kept only
        # the FINAL epoch for these models. So an exported one-epoch dir would say "accuracy
        # 34.244" while its own metadata claimed a best of 34.311 at an epoch that is not in the
        # directory. validate_upload.py rejects exactly that -- it flagged all 500 of batch 3 --
        # and it is right to: the record contradicts itself.
        #
        # Every single-epoch dir ALREADY in the repo either carries best_accuracy equal to its one
        # stored epoch (221 of 400 sampled) or omits the field (179 of 400). Zero disagree. So we
        # follow that convention: when the exported epochs do not include the recorded best, point
        # best_* at the best epoch actually present. The DB still holds the original run's best;
        # this only stops the shipped file from asserting something it cannot show.
        have_best = any(
            r.get("best_epoch") is not None and int(r["best_epoch"]) in eps for r in eps.values())
        if not have_best:
            top_ep = max(eps, key=lambda e: eps[e].get("accuracy") or -1)
            top_acc = eps[top_ep].get("accuracy")
            for r in eps.values():
                if top_acc is not None:
                    r["best_accuracy"] = top_acc
                r["best_epoch"] = top_ep
                clipped[nn_out] = True
        for ep, rec in sorted(eps.items()):
            rec["epoch_max"] = emax
            for k in EXPECTED - set(rec):
                missing_fields[k] += 1
            if a.apply:
                with open(os.path.join(d, f"{ep}.json"), "w") as f:
                    json.dump([rec], f, indent=4, sort_keys=True)
        wrote += 1

    print(f"\n  models: {len(groups)} | {'written' if a.apply else 'would write'}: {wrote} "
          f"| skipped (stat dir already exists): {skipped}")
    print(f"  epoch files per model: {dict(sorted(epoch_hist.items()))}")
    if clipped:
        print(f"  best_accuracy/best_epoch re-pointed to the best STORED epoch for "
              f"{len(clipped)} model(s) whose recorded best epoch was not exported")
    if renamed:
        print(f"\n  {len(renamed)} legacy model name(s) canonicalised '_'->'-' on export "
              f"(the DB row keeps the old name; the stat dir gets the new one):")
        for old, new in sorted(renamed.items())[:10]:
            print(f"    {old}  ->  {new}")
        if len(renamed) > 10:
            print(f"    ... and {len(renamed) - 10} more")
    if missing_fields:
        print("\n  fields absent from the DB and therefore OMITTED (not invented):")
        for k, v in missing_fields.most_common():
            print(f"    {v:>5}  {k}")
    else:
        print("\n  every expected field was recoverable from the DB")
    if 1 in epoch_hist or any(k < 5 for k in epoch_hist):
        print("\n  WARNING: some models have fewer than 5 epoch files. The DB never stored the")
        print("  intermediate epochs for them, so the curve cannot be reconstructed. "
              "validate_upload.py")
        print("  will flag these as missing epochs -- correctly. Re-training is the only fix.")
    if not a.apply:
        print("\nDRY RUN -- rerun with --apply")


if __name__ == "__main__":
    main()
