"""Export model sources from the DB into ab/nn/nn/, for models whose stats are already on disk.

WHY THIS IS NEEDED. 372 LLM models have published-ready training stats in
`ab/nn/stat/train/img-denoising_denoise_psnr_LLM_*` but no `.py` in `ab/nn/nn/`. Committing the
stats alone would publish 372 records describing models nobody can load. All 372 sources are
present in `db/ab.nn.db`, so this recovers them rather than regenerating anything.

SAFETY RULES, because this writes into the repo's model directory:
  * never overwrite an existing file -- an on-disk model is the authority, the DB is the backup
  * every source must parse AND define `Net` before it is written; a model that cannot be imported
    is worse than a missing one, because it breaks any tool that walks the directory
  * refuse to write outside ab/nn/nn
  * report exactly what was written, skipped and rejected, and why

This does NOT touch git and does NOT run json_train_to_db (which would wipe uncommitted models).
"""
import argparse
import ast
import os
import re
import sqlite3
import sys
from collections import Counter

REPO = "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset"
NN_DIR = os.path.join(REPO, "ab", "nn", "nn")
STAT_DIR = os.path.join(REPO, "ab", "nn", "stat", "train")
DB = os.path.join(REPO, "db", "ab.nn.db")


def models_with_stats(prefix):
    """Model names that have a denoise train-stat dir and match `prefix`."""
    out = set()
    for d in os.listdir(STAT_DIR):
        if not d.startswith("img-denoising"):
            continue
        # the model name is the tail; dataset names can contain underscores (awgn_color), so
        # splitting on '_' is unsafe -- anchor on the metric segment instead.
        m = re.search(r"_(psnr|psnrrgb|ssimrgb)_(.+)$", d)
        if m and m.group(2).startswith(prefix):
            out.add(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="LLM_", help="only export models with this name prefix")
    ap.add_argument("--apply", action="store_true",
                    help="actually write files; without it this is a dry run")
    a = ap.parse_args()

    assert os.path.isdir(NN_DIR), NN_DIR
    want = models_with_stats(a.prefix)
    print(f"models matching {a.prefix!r} with a denoise train-stat dir: {len(want)}")

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    tally = Counter()
    wrote = []
    for name in sorted(want):
        dest = os.path.join(NN_DIR, f"{name}.py")
        if os.path.exists(dest):
            tally["already on disk (left alone)"] += 1
            continue
        row = con.execute("select code from nn where name=?", (name,)).fetchone()
        if not row or not row[0] or not row[0].strip():
            tally["NOT in DB / empty code"] += 1
            continue
        code = row[0]
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            tally[f"rejected: syntax error"] += 1
            print(f"  REJECT {name[:52]}: SyntaxError line {e.lineno}")
            continue
        if not any(isinstance(n, ast.ClassDef) and n.name == "Net" for n in tree.body):
            tally["rejected: no class Net"] += 1
            print(f"  REJECT {name[:52]}: no top-level class Net")
            continue
        # keep the write inside NN_DIR no matter what the name contains
        if os.path.dirname(os.path.abspath(dest)) != os.path.abspath(NN_DIR):
            tally["rejected: path escape"] += 1
            continue
        if a.apply:
            with open(dest, "w") as f:
                f.write(code if code.endswith("\n") else code + "\n")
        wrote.append(name)
        tally["exported" if a.apply else "would export"] += 1

    print()
    for k, v in tally.most_common():
        print(f"  {v:>5}  {k}")
    if not a.apply:
        print("\nDRY RUN -- rerun with --apply to write the files")
    else:
        print(f"\nwrote {len(wrote)} file(s) into {NN_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
