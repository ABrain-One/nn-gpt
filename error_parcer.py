#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def pick_dest_dir(dest: Path, use_existing: bool) -> Path:
    if use_existing:
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    # Create a fresh directory; if it exists, add suffixes: _01, _02, ...
    try:
        dest.mkdir(parents=True, exist_ok=False)  # raise if exists [web:140]
        return dest
    except FileExistsError:
        for i in range(1, 1000):
            candidate = dest.with_name(f"{dest.name}_{i:02d}")
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise SystemExit("Could not find a free destination directory name (too many suffixes).")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="out/nngpt/llm/epoch/A0/synth_nn", help="Where to search recursively")
    p.add_argument("--dest", default="out/nngpt/llm/epoch/A0/collected_errors", help="Destination directory")
    p.add_argument("--use-existing", action="store_true", help="If set, reuse --dest if it exists")
    p.add_argument("--overwrite", action="store_true", help="Overwrite if destination file already exists")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    dest = pick_dest_dir(Path(args.dest).resolve(), args.use_existing)

    error_files = sorted(root.rglob("error.txt"))
    copied = skipped = 0

    for src in error_files:
        model_dir_name = src.parent.name  # e.g. B712
        dst = dest / f"{model_dir_name}_error.txt"

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        shutil.copy2(src, dst)  # copy + preserve metadata [web:129]
        copied += 1

    print(f"Root: {root}")
    print(f"Dest: {dest}")
    print(f"Found: {len(error_files)} error.txt files")
    print(f"Copied: {copied}, skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
