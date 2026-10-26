"""Preserve the networks that are ready for the LEMUR dataset across runs.

``out/nngpt`` holds one run and is removed when the next one starts. The
networks under ``out/nngpt/ab_LEMUR_submission`` are not run scratch: they passed
training and carry the statistics that a submission to the NN Dataset needs,
and they were being destroyed together with everything else, so only the last
run's contribution ever survived.

They are therefore moved out of the run directory before it is deleted, into a
directory of their own that no run touches. The two subdirectories are kept as
they are -- ``nn`` for the model sources and ``train`` for the per-model
statistics -- so the archive stays in the layout the dataset expects and can be
submitted as it stands, however many runs contributed to it.
"""
import json
import shutil
from pathlib import Path

from ab.gpt.util.Const import submission_dir_name

#: The run subdirectory that holds submission-ready work, and the archive it is
#: merged into. Both are relative so a test can redirect them.
ARCHIVE_NAME = "ab_lemur_submission"

#: Kept verbatim: a submission is these two directories and nothing else.
SUBMISSION_PARTS = ("nn", "train")


def archive_dir_path():
    from ab.gpt.util.Const import out_dir
    return Path(out_dir) / ARCHIVE_NAME


def _same_file(src: Path, dst: Path) -> bool:
    try:
        if src.stat().st_size != dst.stat().st_size:
            return False
        return src.read_bytes() == dst.read_bytes()
    except Exception:
        return False


def _free_name(dst_dir: Path, name: str) -> Path:
    """A name next to the existing files rather than on top of one.

    Statistics are numbered per model (``1.json``, ``2.json``), so two runs
    that both produced a model of the same name collide by construction. The
    earlier result is the one that must not be lost -- it has already been
    measured -- so the incoming file is given the next free number instead.
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    if stem.isdigit():
        used = {int(p.stem) for p in dst_dir.glob(f"*{suffix}") if p.stem.isdigit()}
        n = max(used) + 1 if used else 1
        return dst_dir / f"{n}{suffix}"
    n = 2
    while (dst_dir / f"{stem}_{n}{suffix}").exists():
        n += 1
    return dst_dir / f"{stem}_{n}{suffix}"


def _merge_tree(src: Path, dst: Path, tally: dict):
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        target = dst / item.name
        if item.is_dir():
            _merge_tree(item, target, tally)
            continue
        if target.exists():
            if _same_file(item, target):
                tally["identical"] += 1
                continue
            target = _free_name(dst, item.name)
            tally["renamed"] += 1
        shutil.move(str(item), str(target))
        tally["moved"] += 1


def archive_submission(base_dir, archive=None):
    """Move submission-ready networks and statistics out of the run directory.

    Call this before the run directory is deleted. Returns the archive path when
    anything was moved, else None. Never raises: losing a run reset to a file
    permission problem would be a worse failure than an unarchived model.
    """
    source = Path(base_dir) / submission_dir_name
    if not source.is_dir():
        return None
    present = [p for p in SUBMISSION_PARTS if (source / p).is_dir()]
    if not present:
        return None

    archive = Path(archive) if archive is not None else archive_dir_path()
    tally = {"moved": 0, "identical": 0, "renamed": 0}
    try:
        for part in present:
            _merge_tree(source / part, archive / part, tally)
        shutil.rmtree(source, ignore_errors=True)
    except Exception as exc:
        print(f"[LEMUR] Could not archive {source}: {exc}")
        return None

    models = len(list((archive / "nn").glob("*.py"))) if (archive / "nn").is_dir() else 0
    records = len([p for p in (archive / "train").iterdir()
                   if p.is_dir()]) if (archive / "train").is_dir() else 0
    note = f", {tally['identical']} already present" if tally["identical"] else ""
    note += f", {tally['renamed']} renamed to avoid overwriting" if tally["renamed"] else ""
    print(f"[LEMUR] Moved {tally['moved']} file(s) to {archive}{note}; "
          f"it now holds {models} network(s) and {records} statistics record(s)")
    _write_manifest(archive, models, records)
    return archive


def _write_manifest(archive: Path, models: int, records: int):
    """A short note on what the archive holds, refreshed on every run."""
    from datetime import datetime, timezone
    try:
        (archive / "manifest.json").write_text(json.dumps({
            "networks": models,
            "statistics_records": records,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2))
    except Exception:
        pass
