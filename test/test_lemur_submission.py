"""The submission archive: a run reset must not destroy measured networks."""
import json
from pathlib import Path

from _unittest_bridge import load_tests_from, pytest  # stdlib stand-in without pytest

try:
    from ab.gpt.util.data.LemurSubmission import archive_submission

    from ab.gpt.util.Const import submission_dir_name
except ImportError as exc:
    from _unittest_bridge import skip_module

    skip_module(f"test_lemur_submission needs ab.gpt (and so ab.nn): {exc}")


def _run(base: Path, models):
    """A run directory holding `models` as {name: {stat_dir: {file: text}}}.

    Built under `base`, which is what archive_submission reads; the dirs in
    Const point at out/nngpt, the one run directory a test must not write to.
    """
    nn_dir = base / submission_dir_name / "nn"
    stat_root = base / submission_dir_name / "train"
    nn_dir.mkdir(parents=True, exist_ok=True)
    stat_root.mkdir(parents=True, exist_ok=True)
    for name, (source, stats) in models.items():
        (nn_dir / f"{name}.py").write_text(source)
        for stat_dir, files in stats.items():
            d = stat_root / stat_dir
            d.mkdir(parents=True, exist_ok=True)
            for fname, text in files.items():
                (d / fname).write_text(text)
    (base / "epoch_tracker.json").write_text("[]")
    return base


def test_moves_both_subdirectories_out_of_the_run(tmp_path):
    base = _run(tmp_path / "nngpt", {
        "chn_a": ("class A: pass", {"img-classification_cifar-10_acc_chn_a": {"1.json": '{"accuracy": 0.6}'}}),
    })
    archive = tmp_path / "ab_lemur_submission"

    assert archive_submission(base, archive) == archive
    assert (archive / "nn" / "chn_a.py").read_text() == "class A: pass"
    assert (archive / "train" / "img-classification_cifar-10_acc_chn_a" / "1.json").exists()
    # moved, not copied: the run directory is about to be deleted either way
    assert not (base / submission_dir_name).exists()
    # and nothing else was taken with it
    assert (base / "epoch_tracker.json").exists()


def test_a_second_run_adds_to_what_is_already_there(tmp_path):
    archive = tmp_path / "archive"
    first = _run(tmp_path / "run1", {
        "chn_a": ("A", {"img-classification_cifar-10_acc_chn_a": {"1.json": '{"a": 1}'}}),
    })
    archive_submission(first, archive)

    second = _run(tmp_path / "run2", {
        "chn_b": ("B", {"img-classification_cifar-10_acc_chn_b": {"1.json": '{"b": 1}'}}),
    })
    archive_submission(second, archive)

    assert sorted(p.name for p in (archive / "nn").glob("*.py")) == ["chn_a.py", "chn_b.py"]
    assert sorted(p.name for p in (archive / "train").iterdir()) == [
        "img-classification_cifar-10_acc_chn_a",
        "img-classification_cifar-10_acc_chn_b",
    ]


def test_an_identical_file_is_not_duplicated(tmp_path):
    archive = tmp_path / "archive"
    model = {"chn_a": ("A", {"img-classification_cifar-10_acc_chn_a": {"1.json": '{"a": 1}'}})}
    archive_submission(_run(tmp_path / "run1", dict(model)), archive)
    archive_submission(_run(tmp_path / "run2", dict(model)), archive)

    assert [p.name for p in (archive / "nn").glob("*.py")] == ["chn_a.py"]
    stats = sorted(p.name for p in
                   (archive / "train" / "img-classification_cifar-10_acc_chn_a").iterdir())
    assert stats == ["1.json"]


def test_a_differing_statistic_is_kept_beside_the_earlier_one(tmp_path):
    """Two runs measured the same model. Neither measurement may be overwritten."""
    archive = tmp_path / "archive"
    stat = "img-classification_cifar-10_acc_chn_a"
    archive_submission(_run(tmp_path / "run1", {
        "chn_a": ("A", {stat: {"1.json": '{"accuracy": 0.61}'}})}), archive)
    archive_submission(_run(tmp_path / "run2", {
        "chn_a": ("A", {stat: {"1.json": '{"accuracy": 0.67}'}})}), archive)

    files = sorted(p.name for p in (archive / "train" / stat).iterdir())
    assert files == ["1.json", "2.json"], "a measurement was lost"
    kept = {json.loads((archive / "train" / stat / f).read_text())["accuracy"] for f in files}
    assert kept == {0.61, 0.67}


def test_a_differing_source_under_the_same_name_is_kept_too(tmp_path):
    archive = tmp_path / "archive"
    archive_submission(_run(tmp_path / "run1", {"chn_a": ("class A: pass", {})}), archive)
    archive_submission(_run(tmp_path / "run2", {"chn_a": ("class B: pass", {})}), archive)

    sources = {p.read_text() for p in (archive / "nn").glob("*.py")}
    assert sources == {"class A: pass", "class B: pass"}


def test_nothing_to_archive_is_not_an_error(tmp_path):
    base = tmp_path / "nngpt"
    base.mkdir()
    assert archive_submission(base, tmp_path / "archive") is None
    assert not (tmp_path / "archive").exists()


def test_an_empty_submission_directory_is_left_alone(tmp_path):
    base = tmp_path / "nngpt"
    (base / submission_dir_name).mkdir(parents=True)
    assert archive_submission(base, tmp_path / "archive") is None


def test_the_manifest_counts_what_the_archive_holds(tmp_path):
    archive = tmp_path / "archive"
    archive_submission(_run(tmp_path / "run1", {
        "chn_a": ("A", {"s_a": {"1.json": "{}"}}),
        "chn_b": ("B", {"s_b": {"1.json": "{}"}}),
    }), archive)

    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["networks"] == 2
    assert manifest["statistics_records"] == 2
    assert manifest["updated"]


def test_run_reset_archives_before_deleting(tmp_path, monkeypatch):
    """The two steps are one operation: reset must never run without the move."""
    from ab.gpt.util.data import EpochStats as ES

    base = _run(tmp_path / "nngpt", {
        "chn_a": ("A", {"img-classification_cifar-10_acc_chn_a": {"1.json": "{}"}}),
    })
    archive = tmp_path / "ab_lemur_submission"
    monkeypatch.setattr("ab.gpt.util.data.LemurSubmission.archive_dir_path",
                        lambda: archive)

    assert ES.reset_run_dir(base) == base
    assert list(base.iterdir()) == [], "the run directory was not cleared"
    assert (archive / "nn" / "chn_a.py").exists()
    assert (archive / "train" / "img-classification_cifar-10_acc_chn_a" / "1.json").exists()


load_tests = load_tests_from(globals())
