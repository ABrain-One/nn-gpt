"""Each run's statistics and plot must outlive the run directory.

out/nngpt is deleted when the next run starts, so epoch_tracker.json and
epoch_statistics.png were lost with it and only the most recent run could ever
be inspected.
"""
import json
from pathlib import Path

from _unittest_bridge import load_tests_from, pytest  # stdlib stand-in without pytest

try:
    from ab.gpt.util.data.RunArchive import archive_run_results, next_index
except ImportError as exc:
    from _unittest_bridge import skip_module

    skip_module(f"test_run_archive needs ab.gpt (and so ab.nn): {exc}")


def _run(base: Path, marker="A", plot=True, stats=True):
    base.mkdir(parents=True, exist_ok=True)
    if plot:
        (base / "epoch_statistics.png").write_bytes(b"\x89PNG" + marker.encode())
    if stats:
        (base / "epoch_tracker.json").write_text(json.dumps([{"epoch": 0, "run": marker}]))
    (base / "run_config.json").write_text("{}")
    return base


def test_the_plot_and_the_statistics_land_in_their_own_places(tmp_path):
    archive = tmp_path / "ab_results"
    assert archive_run_results(_run(tmp_path / "nngpt"), archive) == 1

    assert (archive / "epoch_statistics_0001.png").is_file()
    assert (archive / "ab_stat" / "epoch_tracker_0001.json").is_file()
    # moved, not copied: the run directory is about to be deleted either way
    assert not (tmp_path / "nngpt" / "epoch_statistics.png").exists()
    assert not (tmp_path / "nngpt" / "epoch_tracker.json").exists()
    # and nothing else was taken with them
    assert (tmp_path / "nngpt" / "run_config.json").is_file()


def test_every_run_gets_its_own_number(tmp_path):
    archive = tmp_path / "ab_results"
    for i, marker in enumerate("ABC", start=1):
        assert archive_run_results(_run(tmp_path / f"run{marker}", marker), archive) == i

    assert sorted(p.name for p in archive.glob("*.png")) == [
        "epoch_statistics_0001.png", "epoch_statistics_0002.png",
        "epoch_statistics_0003.png"]
    assert sorted(p.name for p in (archive / "ab_stat").glob("*.json")) == [
        "epoch_tracker_0001.json", "epoch_tracker_0002.json",
        "epoch_tracker_0003.json"]


def test_a_plot_keeps_the_number_of_the_statistics_it_was_drawn_from(tmp_path):
    """The pair is useless if a figure can be matched to another run's records."""
    archive = tmp_path / "ab_results"
    archive_run_results(_run(tmp_path / "r1", "first"), archive)
    archive_run_results(_run(tmp_path / "r2", "second"), archive)

    for n, marker in ((1, "first"), (2, "second")):
        png = (archive / f"epoch_statistics_{n:04d}.png").read_bytes()
        rows = json.loads((archive / "ab_stat" / f"epoch_tracker_{n:04d}.json").read_text())
        assert png.endswith(marker.encode())
        assert rows[0]["run"] == marker


def test_a_run_that_wrote_only_one_of_the_two_does_not_desynchronise_them(tmp_path):
    """A crash before plotting must not make the next pair disagree."""
    archive = tmp_path / "ab_results"
    archive_run_results(_run(tmp_path / "r1", "first"), archive)
    archive_run_results(_run(tmp_path / "r2", "crashed", plot=False), archive)
    archive_run_results(_run(tmp_path / "r3", "third"), archive)

    assert not (archive / "epoch_statistics_0002.png").exists()
    assert (archive / "ab_stat" / "epoch_tracker_0002.json").is_file()
    # the third run's two files still carry one and the same number
    assert (archive / "epoch_statistics_0003.png").is_file()
    assert (archive / "ab_stat" / "epoch_tracker_0003.json").is_file()


def test_existing_numbers_are_continued_not_restarted(tmp_path):
    """A fresh session must not overwrite what earlier sessions archived."""
    archive = tmp_path / "ab_results"
    (archive / "ab_stat").mkdir(parents=True)
    (archive / "epoch_statistics_0007.png").write_bytes(b"old")
    (archive / "ab_stat" / "epoch_tracker_0007.json").write_text("[]")

    assert next_index(archive) == 8
    assert archive_run_results(_run(tmp_path / "nngpt"), archive) == 8
    assert (archive / "epoch_statistics_0007.png").read_bytes() == b"old"


def test_a_stray_unnumbered_file_is_ignored_when_counting(tmp_path):
    archive = tmp_path / "ab_results"
    archive.mkdir(parents=True)
    (archive / "epoch_statistics_notes.png").write_bytes(b"x")

    assert next_index(archive) == 1


def test_nothing_to_archive_is_not_an_error(tmp_path):
    base = tmp_path / "nngpt"
    base.mkdir()
    (base / "run_config.json").write_text("{}")

    assert archive_run_results(base, tmp_path / "ab_results") is None
    assert not (tmp_path / "ab_results").exists()


def test_run_reset_archives_the_results_before_deleting(tmp_path, monkeypatch):
    """The two steps are one operation: reset must never run without the move."""
    from ab.gpt.util.data import EpochStats as ES

    base = _run(tmp_path / "nngpt")
    archive = tmp_path / "ab_results"
    monkeypatch.setattr("ab.gpt.util.data.RunArchive.archive_dir_path", lambda: archive)
    monkeypatch.setattr("ab.gpt.util.data.LemurSubmission.archive_dir_path",
                        lambda: tmp_path / "ab_lemur_submission")

    assert ES.reset_run_dir(base) == base

    assert list(base.iterdir()) == [], "the run directory was not cleared"
    assert (archive / "epoch_statistics_0001.png").is_file()
    assert (archive / "ab_stat" / "epoch_tracker_0001.json").is_file()


load_tests = load_tests_from(globals())
