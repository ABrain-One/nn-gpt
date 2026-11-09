"""Plan denoise model uploads in batches of 500, deduplicated by arch_uid, with a ledger.

THE PROBLEM THIS SOLVES. There are 2,201 denoise models on disk but only 2,822 distinct arch_uid
across the whole 2,958-row DB: 46 arch_uid groups contain 182 models, and the worst single group
holds 49 models that are the SAME architecture under different names. Uploading by filename would
put ~136 redundant architectures into the shared DB. arch_uid is the project's definition of
distinctness, so it is the key used here -- not the filename, and not the code hash (which misses
structurally identical models that differ by whitespace or ordering).

WHAT A BATCH CONTAINS. A model and its training stats always travel together. A commit with a model
but no stats publishes something LEMUR cannot rank; stats without a model publishes a record nobody
can load. Every batch list therefore pairs `ab/nn/nn/<name>.py` with
`ab/nn/stat/train/img-denoising_*_<name>/`.

WHAT COUNTS AS ALREADY UPLOADED. The union of
  (a) files tracked in git -- the authority, since that is what was actually pushed, and
  (b) names recorded in the ledger by `--confirm`.
(a) alone is enough in normal use; the ledger exists so that a batch you have pushed but not yet
pulled back, or a push made from another machine, does not get re-offered.

Usage:
    python upload_batches.py --status              # what is left, and why things are excluded
    python upload_batches.py --plan                # write the next batch list (does not touch git)
    python upload_batches.py --confirm 1           # after YOU have committed+pushed batch 1

This never runs a git command that changes state. It only reads `git ls-files`.
"""
import argparse
import collections
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

REPO = "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset"
NN_DIR = os.path.join(REPO, "ab", "nn", "nn")
STAT_DIR = os.path.join(REPO, "ab", "nn", "stat", "train")
DB = os.path.join(REPO, "db", "ab.nn.db")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import naming  # noqa: E402

LEDGER = os.path.join(HERE, "docs", "upload_ledger.json")

# PSNR of the identity function on this validation set (256 crop, Y channel). Measured, not
# assumed: see docs/noop_floor.md. A model scoring below it is worse than no denoising at all.
NOOP_DB = 33.021
BATCH_DIR = os.path.join(HERE, "docs", "upload_batches")
SIZE = 500

# "LLM-" AND "LLM_" are both listed on purpose. The 372 LLM models were renamed on disk from
# LLM_Pg23_seeded-<hash> to LLM-Pg23-seeded-<hash> because config_splitter is '_' and Util.py:61
# parses a stat dir with a plain 4-way split, so an underscore in the model name breaks import.
# The DB was NOT rewritten (json_train_to_db would drop uncommitted models), so it still holds the
# old names -- hence db_aliased() below. Dropping "LLM_" here would silently exclude every one of
# them from the upload; keeping only "LLM_" excluded all 372, which is how this was found.
# Every denoise-producing arm. This list is the ONLY thing that decides whether a model on disk
# is even considered for upload, and it silently ignores anything it does not list -- a model with
# a valid source and a valid stat dir simply never appears in --status. It predated the dnseed2
# family, so on 2026-10-01 the 1,434 recovered DnSeed2*/DnSeed3/DnSeedGA2 models were exported
# correctly and then counted as zero. Add new prefixes here when an arm is added.
DN_PREFIXES = ("Denoise", "DnSeedGA-", "DnSeedGA2-", "DnSeed2Act-", "DnSeed2Fast-",
               "DnSeed2Wide-", "DnSeed3-", "DnSearch-", "DnSkip-", "DnNewFam-",
               "LLM_", "LLM-", "LiteDenoise")


def is_denoise(name):
    return name.startswith(DN_PREFIXES)


def db_aliased(d):
    """Add '-'-for-'_' aliases so DB rows keyed by pre-rename names resolve for on-disk names.

    Only fills gaps: a real entry for the new name always wins. The rename was a full
    str.replace('_', '-'), and it was collision-checked at the time (372 unique targets, none
    already taken), so an alias cannot overwrite a different model's value.

    Delegates to naming.alias_both_ways so there is exactly one implementation of the
    old-name/new-name mapping in the tree.
    """
    return naming.alias_both_ways(d)


def load_ledger():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"uploaded": {}, "batches": {}, "skipped_duplicates": {}}


def save_ledger(led):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    json.dump(led, open(LEDGER, "w"), indent=1, sort_keys=True)


def git_tracked_models():
    out = subprocess.run(["git", "-C", REPO, "ls-files", "ab/nn/nn"],
                         capture_output=True, text=True).stdout.split()
    return {os.path.basename(p)[:-3] for p in out
            if p.endswith(".py") and is_denoise(os.path.basename(p)[:-3])}


def stat_dirs_by_model():
    """model -> [stat dir names]. Anchored on the metric segment because dataset names can
    contain underscores (awgn_color), which makes splitting on '_' unsafe."""
    out = collections.defaultdict(list)
    for d in os.listdir(STAT_DIR):
        if not d.startswith("img-denoising"):
            continue
        m = re.search(r"_(psnr|psnrrgb|ssimrgb)_(.+)$", d)
        if m:
            out[m.group(2)].append(d)
    return out


def fs_psnr(stats):
    """model -> best PSNR read from the stat JSON ON DISK, which is what actually gets uploaded.

    Preferred over the DB because the DB can be stale: the 376 DnSeedGA models were retrained to
    5 epochs and merged in from their save_path, and that never went through json_train_to_db
    (which would drop uncommitted models), so the DB still holds their old single-epoch numbers.
    Ranking on the DB there would order the batches by figures we are not shipping.
    """
    out = {}
    for model, dirs in stats.items():
        best = None
        for d in dirs:
            if "_psnr_" not in d:           # accuracy*48 is only PSNR for the psnr metric
                continue
            for f in os.listdir(os.path.join(STAT_DIR, d)):
                if not f.endswith(".json"):
                    continue
                try:
                    j = json.load(open(os.path.join(STAT_DIR, d, f)))
                except Exception:
                    continue
                j = j[0] if isinstance(j, list) else j
                a = j.get("accuracy")
                if a is not None and (best is None or a > best):
                    best = a
        if best is not None:
            out[model] = best * 48
    return out


ARCH_UID_CACHE = os.path.join(HERE, "docs", "arch_uid2_computed.json")


def arch_uids():
    """name -> arch_uid2, the project's definition of architectural distinctness.

    VERIFIED: the DB's `arch_uid` column IS arch_uid2 -- recomputing
    `arch_uid2.arch_uid(code)` for a 25-model sample reproduced the stored value 25/25. So the
    column is used directly where present, which avoids re-deriving 2,900 WL certificates.

    Models on disk but absent from the DB have no stored uid. They are NOT dropped: their uid is
    read from ARCH_UID_CACHE, computed from source by arch_uid2 (which works on source text and
    so applies even to models that will not import). All 79 such DenoiseGA models turned out to be
    genuinely new -- 79 distinct uids, zero collisions with the DB or with each other -- so
    excluding them for want of a uid would have discarded 79 distinct architectures.
    """
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    uids = {n: u for n, u in con.execute("select name, arch_uid from nn") if u}
    if os.path.exists(ARCH_UID_CACHE):
        for n, u in json.load(open(ARCH_UID_CACHE)).items():
            uids.setdefault(n, u)
    return db_aliased(uids)


def best_psnr():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    return db_aliased({n: p for n, p in con.execute(
        """select nn, max(accuracy)*48 from stat
           where task='img-denoising' and metric='psnr' group by nn""")})


def build():
    """-> (ready, excluded) where ready is an ordered list of model names safe to upload."""
    led = load_ledger()
    tracked = git_tracked_models()
    uploaded = set(tracked) | set(led["uploaded"])
    stats = stat_dirs_by_model()
    uids = arch_uids()
    psnr = best_psnr()
    psnr.update(fs_psnr(stats))     # on-disk stats win: they are what we ship

    on_disk = {f[:-3] for f in os.listdir(NN_DIR)
               if f.endswith(".py") and is_denoise(f[:-3])}

    excluded = collections.defaultdict(list)
    cands = []
    for n in sorted(on_disk):
        if n in uploaded:
            excluded["already uploaded"].append(n)
        elif n not in stats:
            excluded["no training stats -- inert in LEMUR"].append(n)
        elif n not in uids:
            excluded["no arch_uid2 (not in DB, not in cache)"].append(n)
        elif (psnr.get(n) or -1e9) < NOOP_DB:
            # BELOW THE NO-OP FLOOR = WORSE THAN NOT DENOISING AT ALL.
            #
            # 33.021 dB is what the identity function scores on this validation set at the
            # 256-crop Y metric: copy the noisy input to the output and you get that. A model
            # under it has learned something actively harmful, and 19 of the 303 such models
            # score below 20 dB. Shipping them in a denoising dataset would mean a reviewer who
            # samples the tail finds entries that degrade images, so they are held out by a
            # stated criterion rather than quietly dropped -- `--include-below-noop` ships them.
            excluded["below the %.3f dB no-op floor" % NOOP_DB].append(n)
        else:
            cands.append(n)

    # arch_uid already represented by something uploaded -> would duplicate the shared DB
    uploaded_uids = {uids[n] for n in uploaded if n in uids}
    cands2 = []
    for n in cands:
        if uids[n] in uploaded_uids:
            excluded["duplicate arch_uid of an already-uploaded model"].append(n)
        else:
            cands2.append(n)

    # within what is left, keep ONE representative per arch_uid: highest measured PSNR, then name
    groups = collections.defaultdict(list)
    for n in cands2:
        groups[uids[n]].append(n)
    ready = []
    for u, names in groups.items():
        names.sort(key=lambda x: (-(psnr.get(x) or -1e9), x))
        ready.append(names[0])
        for dup in names[1:]:
            excluded["duplicate arch_uid within this upload"].append(dup)
    # deterministic order, best first, so early batches carry the most useful models
    ready.sort(key=lambda x: (-(psnr.get(x) or -1e9), x))
    return ready, excluded, stats, uids, psnr, led


def cmd_status():
    ready, excluded, stats, uids, psnr, led = build()
    print(f"models on disk (denoise):        "
          f"{len([f for f in os.listdir(NN_DIR) if f.endswith('.py') and is_denoise(f[:-3])])}")
    print(f"already uploaded (git + ledger): {len(git_tracked_models()) + 0} tracked, "
          f"{len(led['uploaded'])} in ledger")
    print(f"\nREADY TO UPLOAD (distinct architectures): {len(ready)}")
    print(f"  -> {(len(ready) + SIZE - 1) // SIZE} batch(es) of {SIZE}")
    print("\nexcluded:")
    for k, v in sorted(excluded.items(), key=lambda x: -len(x[1])):
        print(f"  {len(v):>5}  {k}")
    if led["batches"]:
        print("\nbatches recorded:")
        for b, meta in sorted(led["batches"].items(), key=lambda x: int(x[0])):
            print(f"  batch {b}: {meta['count']} models, confirmed {meta.get('confirmed_at','NO')}")


def cmd_plan(a):
    ready, excluded, stats, uids, psnr, led = build()
    if not ready:
        print("nothing ready to upload")
        return
    done = {n for b in led["batches"].values() for n in b.get("models", [])}
    pend = [n for n in ready if n not in done]
    if not pend:
        print("every ready model is already assigned to a batch; confirm the open ones first")
        return
    bn = (max((int(x) for x in led["batches"]), default=0) + 1)
    batch = pend[:a.size]
    # LAST GATE BEFORE THE USER'S `git add`. Everything past this point is a commit,
    # and a single unparseable stat dir aborts the import of the entire batch -- that
    # is what the review caught in batch 1. Refuse to write the list at all rather
    # than emit one that will be rejected downstream; a plan that fails here is a
    # generator bug to fix at source, not something to paper over. See naming.py.
    bad = []
    for n in batch:
        if not naming.is_legal(n):
            bad.append((f"ab/nn/nn/{n}.py", f"model name contains '_'"))
        for d in sorted(stats[n]):
            try:
                naming.check_stat_dir(d)
            except naming.NameError_ as e:
                bad.append((f"ab/nn/stat/train/{d}", str(e)))
    if bad:
        print(f"REFUSING to write batch {bn}: {len(bad)} path(s) would break the "
              f"<task>_<dataset>_<metric>_<model> split in Util.py:61")
        for p, why in bad[:15]:
            print(f"    {p}\n      {why}")
        if len(bad) > 15:
            print(f"    ... and {len(bad) - 15} more")
        sys.exit(1)

    os.makedirs(BATCH_DIR, exist_ok=True)
    lst = os.path.join(BATCH_DIR, f"batch_{bn:03d}.txt")
    with open(lst, "w") as f:
        for n in batch:
            f.write(f"ab/nn/nn/{n}.py\n")
            for d in sorted(stats[n]):
                f.write(f"ab/nn/stat/train/{d}\n")
    led["batches"][str(bn)] = {"count": len(batch), "models": batch,
                               "planned_at": datetime.now(timezone.utc).isoformat(),
                               "list": os.path.relpath(lst, HERE)}
    save_ledger(led)
    nfiles = sum(1 for _ in open(lst))
    print(f"batch {bn}: {len(batch)} models, {nfiles} paths -> {lst}")
    print(f"  remaining after this batch: {len(pend) - len(batch)}")
    print(f"\n  PSNR range in this batch: "
          f"{min((psnr.get(n) or 0) for n in batch):.3f} .. {max((psnr.get(n) or 0) for n in batch):.3f} dB")
    print(f"\n  YOU run, from {REPO}:")
    print(f"    git add $(cat {lst})")
    print(f"    git diff --cached --name-status | awk '$1==\"D\"' | wc -l   # MUST print 0")
    print(f"    git commit -m 'denoise: add {len(batch)} distinct models + stats (batch {bn})'")
    print(f"  then: python upload_batches.py --confirm {bn}")


def cmd_confirm(a):
    led = load_ledger()
    b = str(a.confirm)
    if b not in led["batches"]:
        sys.exit(f"batch {b} was never planned")
    meta = led["batches"][b]
    if meta.get("confirmed_at"):
        print(f"batch {b} already confirmed at {meta['confirmed_at']}")
        return
    uids = arch_uids()
    for n in meta["models"]:
        led["uploaded"][n] = {"batch": int(b), "arch_uid": uids.get(n)}
    meta["confirmed_at"] = datetime.now(timezone.utc).isoformat()
    save_ledger(led)
    print(f"batch {b} confirmed: {len(meta['models'])} models recorded as uploaded")
    print(f"  ledger now holds {len(led['uploaded'])} uploaded model(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--confirm", type=int, default=None)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--include-below-noop", action="store_true",
                    help="also ship models scoring below the no-op floor; they are excluded "
                         "by default because their output is worse than their input")
    a = ap.parse_args()
    if a.include_below_noop:
        globals()["NOOP_DB"] = -1e9
    if a.confirm is not None:
        cmd_confirm(a)
    elif a.plan:
        cmd_plan(a)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
