"""Keep each run's statistics and plot after the run directory is deleted.

``out/nngpt`` holds one run and is removed when the next one starts, taking
``epoch_tracker.json`` and ``epoch_statistics.png`` with it. Those two files are
the result of the run -- every epoch's rates, accuracies and intervals, and the
figure drawn from them -- so only the most recent run could ever be looked at,
and two configurations could not be compared after the fact.

They are therefore moved out of the run directory before it is deleted, into an
archive that no run touches, under a number that increases with every run. The
pair keeps the SAME number, so a plot and the statistics it was drawn from stay
identified with each other however many runs accumulate.
"""
import shutil
from pathlib import Path

ARCHIVE_NAME = "ab_results"

#: What is archived, and where under the archive it goes. The figure sits at the
#: top so a reader can flip through the plots; the records it was drawn from go
#: in their own subdirectory rather than interleaved with them.
ARCHIVED_FILES = (
    ("epoch_statistics.png", ""),
    ("epoch_tracker.json", "ab_stat"),
)

INDEX_DIGITS = 4


def archive_dir_path():
    from ab.gpt.util.Const import out_dir
    return Path(out_dir) / ARCHIVE_NAME


def _destination(archive: Path, sub: str) -> Path:
    return archive / sub if sub else archive


def next_index(archive: Path) -> int:
    """One past the highest number already archived, across every kind of file.

    Counted over all of them together rather than per directory: the plot and
    its statistics must come back under the same number, and a run that wrote
    only one of the two would otherwise let the counters drift apart and start
    pairing a figure with another run's records.
    """
    highest = 0
    for name, sub in ARCHIVED_FILES:
        stem, suffix = Path(name).stem, Path(name).suffix
        directory = _destination(archive, sub)
        if not directory.is_dir():
            continue
        for path in directory.glob(f"{stem}_*{suffix}"):
            tail = path.stem[len(stem) + 1:]
            if tail.isdigit():
                highest = max(highest, int(tail))
    return highest + 1


def archive_run_results(base_dir, archive=None):
    """Move this run's statistics and plot into the numbered archive.

    Call this before the run directory is deleted. Returns the number they were
    archived under, or None when there was nothing to archive. Never raises:
    losing a run reset to a file permission problem would be a worse failure
    than an unarchived plot.
    """
    base_dir = Path(base_dir)
    present = [(n, sub) for n, sub in ARCHIVED_FILES if (base_dir / n).is_file()]
    if not present:
        return None

    try:
        # Resolving the archive directory is inside the try as well: it reaches
        # into the project configuration and can fail on its own, and a reset
        # that dies here would leave the run unable to start.
        archive = Path(archive) if archive is not None else archive_dir_path()
        index = next_index(archive)
        moved = []
        for name, sub in present:
            directory = _destination(archive, sub)
            directory.mkdir(parents=True, exist_ok=True)
            source = Path(name)
            target = directory / f"{source.stem}_{index:0{INDEX_DIGITS}d}{source.suffix}"
            shutil.move(str(base_dir / name), str(target))
            moved.append(target)
    except Exception as exc:
        print(f"[RESULTS] Could not archive the run's results: {exc}")
        return None

    print(f"[RESULTS] Run #{index} archived to {archive}: "
          + ", ".join(p.name for p in moved))
    return index
