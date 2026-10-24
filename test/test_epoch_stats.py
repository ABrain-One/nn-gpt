"""Statistical correctness of the per-epoch report (ab/gpt/util/data/EpochStats.py).

Both cases here are regressions of bugs that shipped: a Wilson interval that
divided the wrong term by n, and a baseline that averaged the seed accuracy of
every generation attempt instead of only the ones the LLM modified successfully.
"""
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    cs = importlib.util.spec_from_file_location(
        "ab.gpt.util.data.CycleResults", ROOT / "ab/gpt/util/data/CycleResults.py")
    cr = importlib.util.module_from_spec(cs)
    cs.loader.exec_module(cr)
    sys.modules.setdefault("ab.gpt.util.data.CycleResults", cr)
    es = importlib.util.spec_from_file_location(
        "_epoch_stats_under_test", ROOT / "ab/gpt/util/data/EpochStats.py")
    mod = importlib.util.module_from_spec(es)
    es.loader.exec_module(mod)
    return mod


ES = _load()


@pytest.mark.parametrize("k, n, lo, hi", [
    (0, 10, 0.0000, 0.2775),
    (2, 10, 0.0566, 0.5098),
    (4, 10, 0.1682, 0.6873),
    (10, 10, 0.7225, 1.0000),
])
def test_wilson_interval_matches_reference(k, n, lo, hi):
    _, got_lo, got_hi = ES.wilson_ci(k, n)
    assert got_lo == pytest.approx(lo, abs=1e-4)
    assert got_hi == pytest.approx(hi, abs=1e-4)


def test_wilson_interval_is_degenerate_only_for_no_trials():
    assert ES.wilson_ci(0, 0) == (0.0, 0.0, 0.0)
    # zero successes out of n must still bound the rate above zero
    assert ES.wilson_ci(0, 10)[2] > 0.0


def test_mean_ci_uses_t_not_z_for_small_samples():
    vals = [0.5895, 0.41, 0.33, 0.25]
    out = ES.mean_ci(vals)
    assert out["n"] == 4
    half_t = out["ci_high"] - out["mean"]
    half_z = 1.96 * out["std"] / 2.0          # sqrt(n) == 2
    assert half_t > half_z                     # t(3)=3.182 is wider than 1.96


def _make_epoch(tmp_path, valid, failed_seeds):
    d = tmp_path / "A0" / "synth_nn"
    d.mkdir(parents=True)
    pd = pytest.importorskip("pandas")
    for i, (seed, acc) in valid.items():
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": seed}).to_pickle(md / "dataframe.df")
        (md / "eval_info.json").write_text(json.dumps({"eval_results": {"accuracy": acc}}))
    for i, seed in failed_seeds.items():
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": seed}).to_pickle(md / "dataframe.df")
        (md / "error.txt").write_text("compile error")
    return d


def test_baseline_covers_only_successfully_modified_models(tmp_path):
    # Seeds of the failing models are deliberately far from the succeeding ones,
    # so averaging all of them gives a visibly different answer.
    valid = {0: (0.30, 0.5895), 1: (0.32, 0.41), 2: (0.28, 0.33), 3: (0.34, 0.25)}
    failed = {i: 0.90 for i in range(4, 10)}
    stats = ES.collect_epoch_stats(_make_epoch(tmp_path, valid, failed), 0)

    assert stats["models_generated"] == 10
    assert stats["models_valid"] == 4
    assert stats["baseline_accuracy_n"] == 4          # not 10
    assert stats["baseline_accuracy_mean"] == pytest.approx(0.31)
    assert stats["baseline_models"] == ["seed-0", "seed-1", "seed-2", "seed-3"]


def test_improvement_is_paired_per_model(tmp_path):
    valid = {0: (0.30, 0.5895), 1: (0.32, 0.41), 2: (0.28, 0.33), 3: (0.34, 0.25)}
    stats = ES.collect_epoch_stats(_make_epoch(tmp_path, valid, {}), 0)
    expected = sum(a - s for s, a in valid.values()) / len(valid)
    assert stats["paired_n"] == 4
    assert stats["mean_improvement"] == pytest.approx(expected)
    assert stats["improvement_ci_low"] < stats["mean_improvement"] < stats["improvement_ci_high"]


def test_epoch_with_no_valid_models_reports_nulls_not_zeros(tmp_path):
    stats = ES.collect_epoch_stats(_make_epoch(tmp_path, {}, {i: 0.5 for i in range(10)}), 2)
    assert stats["models_valid"] == 0
    assert stats["accuracy_mean"] is None
    assert stats["accuracy_max"] is None
    assert stats["mean_improvement"] is None
    assert stats["valid_rate"] == 0.0
    assert stats["valid_rate_ci_high"] > 0.0


def test_tracker_replaces_rather_than_duplicates_a_rerun_epoch(tmp_path):
    path = tmp_path / "epoch_tracker.json"
    ES.append_epoch_stats(path, {"epoch": 0, "accuracy_max": 0.1})
    ES.append_epoch_stats(path, {"epoch": 1, "accuracy_max": 0.2})
    rows = ES.append_epoch_stats(path, {"epoch": 0, "accuracy_max": 0.9})
    assert [r["epoch"] for r in rows] == [0, 1]
    assert rows[0]["accuracy_max"] == 0.9


# --- finalisation of the per-epoch results file ---------------------------

def _stats(epoch, valid=2, total=10, best=0.45):
    return {"epoch": epoch, "models_valid": valid, "models_generated": total,
            "accuracy_max": best, "accuracy_mean": best, "valid_rate": valid / total}


def test_finalize_adopts_a_generic_file_left_by_the_last_cycle(tmp_path):
    # The run that "kept the name cycle_results.json": NNEval wrote only the
    # generic file, so the epoch has no record of its own.
    generic = tmp_path / "cycle_results.json"
    generic.write_text(json.dumps(
        {"cycle": 3, "evaluation": {"best_accuracy": 0.45}, "cycle_time_minutes": 12.0}))

    ES.finalize_epoch_results(tmp_path, 3, _stats(3))

    per_epoch = tmp_path / "cycle_results_3.json"
    assert per_epoch.is_file()
    payload = json.loads(per_epoch.read_text())
    assert payload["cycle"] == 3
    assert payload["cycle_time_minutes"] == 12.0      # original content preserved
    assert payload["statistics"]["models_valid"] == 2  # statistics merged in


def test_finalize_does_not_adopt_a_generic_file_from_another_epoch(tmp_path):
    (tmp_path / "cycle_results.json").write_text(json.dumps(
        {"cycle": 2, "evaluation": {"best_accuracy": 0.99}, "cycle_time_minutes": 5.0}))

    ES.finalize_epoch_results(tmp_path, 3, _stats(3, best=0.45))

    payload = json.loads((tmp_path / "cycle_results_3.json").read_text())
    assert payload["cycle"] == 3
    assert payload["evaluation"]["best_accuracy"] == 0.45   # rebuilt, not epoch 2's 0.99
    assert payload.get("cycle_time_minutes") is None


def test_finalize_writes_a_record_when_evaluation_never_ran(tmp_path):
    ES.finalize_epoch_results(tmp_path, 4, _stats(4, valid=0, total=10, best=None))
    payload = json.loads((tmp_path / "cycle_results_4.json").read_text())
    assert payload["cycle"] == 4
    assert payload["success"] is False
    assert payload["generation"]["total_generated"] == 10
    assert payload["evaluation"]["best_accuracy"] is None
    # the generic name is refreshed too, so both agree
    assert json.loads((tmp_path / "cycle_results.json").read_text())["cycle"] == 4


def test_update_epoch_report_finalises_and_tracks_a_collapsed_epoch(tmp_path):
    pd = pytest.importorskip("pandas")
    d = tmp_path / "A2" / "synth_nn"
    d.mkdir(parents=True)
    for i in range(10):                      # all generations failed
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": 0.5}).to_pickle(md / "dataframe.df")
        (md / "error.txt").write_text("compile error")

    ES.update_epoch_report(d, 2, tmp_path)

    rows = json.loads((tmp_path / "epoch_tracker.json").read_text())
    assert [r["epoch"] for r in rows] == [2]
    assert rows[0]["models_valid"] == 0 and rows[0]["models_generated"] == 10
    assert (tmp_path / "cycle_results_2.json").is_file()


# --- clearing the previous run's artifacts --------------------------------

def _seed_run_dir(tmp_path, epochs=range(20)):
    (tmp_path / "cycle_results.json").write_text("{}")
    (tmp_path / "epoch_tracker.json").write_text("[]")
    (tmp_path / "epoch_statistics.png").write_bytes(b"\x89PNG")
    for e in epochs:
        (tmp_path / f"cycle_results_{e}.json").write_text("{}")
    # files that belong to the model, not to the statistics, must survive
    (tmp_path / "run_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"w")
    keep = tmp_path / "checkpoint"
    keep.mkdir()
    (keep / "adapter_config.json").write_text("{}")


def test_reset_removes_every_statistics_file_of_a_previous_run(tmp_path):
    _seed_run_dir(tmp_path)
    removed = ES.reset_run_artifacts(tmp_path)

    assert len(removed) == 23            # 20 per-epoch + 3 named
    assert not list(tmp_path.glob("cycle_results*.json"))
    assert not (tmp_path / "epoch_tracker.json").exists()
    assert not (tmp_path / "epoch_statistics.png").exists()


def test_reset_leaves_model_and_config_files_alone(tmp_path):
    _seed_run_dir(tmp_path, epochs=range(3))
    ES.reset_run_artifacts(tmp_path)

    assert (tmp_path / "run_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / "checkpoint" / "adapter_config.json").is_file()


def test_reset_is_a_noop_on_a_clean_or_missing_directory(tmp_path):
    assert ES.reset_run_artifacts(tmp_path) == []
    assert ES.reset_run_artifacts(tmp_path / "does_not_exist") == []


def test_without_reset_a_short_run_inherits_a_longer_run_s_tail(tmp_path):
    # 20-epoch run, then a 4-epoch run writing over epochs 0..3 only.
    path = tmp_path / "epoch_tracker.json"
    for e in range(20):
        ES.append_epoch_stats(path, {"epoch": e, "accuracy_max": 0.1})
    for e in range(4):
        rows = ES.append_epoch_stats(path, {"epoch": e, "accuracy_max": 0.9})
    assert len(rows) == 20                      # epochs 4..19 are stale leftovers

    ES.reset_run_artifacts(tmp_path)
    for e in range(4):
        rows = ES.append_epoch_stats(path, {"epoch": e, "accuracy_max": 0.9})
    assert [r["epoch"] for r in rows] == [0, 1, 2, 3]


# --- the plot must carry the original networks on both accuracy panels ----

def _plot_rows():
    return [
        {"epoch": 0, "models_generated": 10, "models_valid": 4, "valid_rate": 0.4,
         "valid_rate_ci_low": 0.17, "valid_rate_ci_high": 0.69,
         "accuracy_mean": 0.46, "accuracy_ci_low": 0.28, "accuracy_ci_high": 0.65,
         "accuracy_max": 0.5895, "baseline_accuracy_mean": 0.54,
         "baseline_accuracy_ci_low": 0.44, "baseline_accuracy_ci_high": 0.63,
         "baseline_accuracy_max": 0.61},
        # n=2: the t-interval runs outside [0, 1] and must be clipped when drawn
        {"epoch": 1, "models_generated": 10, "models_valid": 2, "valid_rate": 0.2,
         "valid_rate_ci_low": 0.06, "valid_rate_ci_high": 0.51,
         "accuracy_mean": 0.44, "accuracy_ci_low": 0.29, "accuracy_ci_high": 0.59,
         "accuracy_max": 0.4501, "baseline_accuracy_mean": 0.535,
         "baseline_accuracy_ci_low": -0.04, "baseline_accuracy_ci_high": 1.11,
         "baseline_accuracy_max": 0.58},
        {"epoch": 2, "models_generated": 10, "models_valid": 0, "valid_rate": 0.0,
         "valid_rate_ci_low": 0.0, "valid_rate_ci_high": 0.28,
         "accuracy_mean": None, "accuracy_ci_low": None, "accuracy_ci_high": None,
         "accuracy_max": None, "baseline_accuracy_mean": None,
         "baseline_accuracy_ci_low": None, "baseline_accuracy_ci_high": None,
         "baseline_accuracy_max": None},
    ]


def _labels(ax):
    return ax.get_legend_handles_labels()[1]


def test_plot_series_cover_generated_and_original_on_both_panels(tmp_path, monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["labels"] = [_labels(ax) for ax in fig.axes]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")

    rate, accuracy = captured["labels"]
    assert any("Wilson" in l for l in rate)
    # accuracy panel: both means AND both intervals stay identifiable...
    assert any(l == "generated mean" for l in accuracy)
    assert any(l == "generated 95% CI" for l in accuracy)
    assert any("original" in l and "mean" in l for l in accuracy)
    assert any("original" in l and "95% CI" in l for l in accuracy)
    # ...and the maxima, folded in from the former third panel, are on it too
    assert any(l == "generated best" for l in accuracy)
    assert any(l == "original best" for l in accuracy)


def test_plot_draws_intervals_as_computed_and_frames_on_measured_values(tmp_path, monkeypatch):
    """Bounds are drawn as computed; the VIEW is framed on the measured values.

    A t-interval on n=2 can run past 0 and 1. Clamping it to [0, 1] turned every
    wide interval into a band filling the panel, so no series appeared to move
    between epochs. The band is therefore drawn where it was computed and simply
    leaves the frame, while the axis is framed on the means and best values that
    were actually observed -- which is what a reader is comparing.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            ys = []
            for coll in fig.axes[1].collections:          # the fill_between bands
                for path in coll.get_paths():
                    ys.extend(v[1] for v in path.vertices)
            captured["ys"] = ys
            captured["ylim"] = fig.axes[1].get_ylim()
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    rows = _plot_rows()
    ES.plot_epoch_stats(rows, tmp_path / "p.png")

    ys = captured["ys"]
    assert ys, "no shaded interval was drawn"
    assert min(ys) == pytest.approx(-0.04), "lower bound was altered, not drawn as computed"

    lo, hi = captured["ylim"]
    measured = [v for r in rows
                for v in (r.get("accuracy_mean"), r.get("baseline_accuracy_mean"),
                          r.get("accuracy_max"), r.get("baseline_accuracy_max"))
                if v is not None]
    assert lo > min(ys), "the view was framed on the interval, not the measurements"
    assert lo <= min(measured) and hi >= max(measured), "a measured value fell outside the view"
    # and the stored record keeps the exact, unclipped bounds
    assert rows[1]["baseline_accuracy_ci_low"] == -0.04


# --- a single observation supports no interval ---------------------------

def test_single_observation_yields_no_interval_not_a_zero_width_one():
    """One model cannot support a confidence interval.

    Returning the mean for both bounds draws a zero-width band, which reads as
    perfect certainty and makes two means look significantly different when
    nothing of the sort has been shown.
    """
    out = ES.mean_ci([0.1455])
    assert out["n"] == 1
    assert out["mean"] == pytest.approx(0.1455)
    assert out["ci_low"] is None and out["ci_high"] is None
    assert out["std"] is None
    # two observations are enough for a (wide) interval
    two = ES.mean_ci([0.45, 0.43])
    assert two["ci_low"] is not None and two["ci_high"] > two["ci_low"]


def test_plot_draws_the_full_range_when_the_interval_is_not_estimable(tmp_path, monkeypatch):
    """One model gives an unbounded interval; an accuracy lives in [0, 1].

    Drawing the whole feasible range states that plainly. The failure this
    guards against is the opposite: a band collapsed onto the mean, which
    would assert the mean is known exactly.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    rows = _plot_rows()
    rows[1].update(models_valid=1, accuracy_ci_low=None, accuracy_ci_high=None,
                   baseline_accuracy_ci_low=None, baseline_accuracy_ci_high=None)
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            spans = []
            for coll in fig.axes[1].collections:
                for path in coll.get_paths():
                    at_x1 = [v[1] for v in path.vertices if abs(v[0] - 1.0) < 1e-9]
                    if at_x1:
                        spans.append((min(at_x1), max(at_x1)))
            captured["spans"] = spans
            captured["labels"] = fig.axes[1].get_legend_handles_labels()[1]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / "p.png")

    assert captured["spans"], "no band was drawn at the single-model epoch"
    for low, high in captured["spans"]:
        assert low == pytest.approx(0.0)
        assert high == pytest.approx(1.0)
    # no extra legend entry is needed: the band speaks for itself
    assert not any("estimable" in l for l in captured["labels"])
    assert sum("95% CI" in l for l in captured["labels"]) == 2


# --- the confidence boundary must read as a band edge, not a second series ---


def test_ci_boundary_is_hairline_and_faint_next_to_the_mean_line(tmp_path, monkeypatch):
    """The CI edge used to be as heavy as the mean, so panel 2 showed six lines.

    It is context, not a measurement: thinner than the mean line and well below
    full opacity.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["lines"] = [(round(l.get_linewidth(), 3), l.get_alpha())
                                 for l in fig.axes[1].lines]
            # CI edges are the hairlines drawn at exactly CI_EDGE_WIDTH; the
            # accuracy panel also carries max/best-so-far lines that have an
            # alpha but are not interval boundaries.
            captured["edges"] = [(w, a) for w, a in captured["lines"]
                                 if w == round(ES.CI_EDGE_WIDTH, 3)]
            captured["means"] = [(w, a) for w, a in captured["lines"] if a is None]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")

    edges, means = captured["edges"], captured["means"]
    assert edges, "no CI boundary was drawn"
    assert means, "no mean line was drawn"
    mean_width = max(w for w, _ in means)
    for width, alpha in edges:
        assert width < mean_width, "boundary is as heavy as the mean line"
        assert width <= ES.CI_EDGE_WIDTH
        assert alpha <= ES.CI_EDGE_ALPHA, "boundary is too opaque to read as an edge"


def test_a_single_epoch_still_shows_its_interval(tmp_path, monkeypatch):
    """fill_between needs two x positions, so one cycle drew no interval at all.

    The first cycle of every run is exactly that case. One epoch is rendered as
    a capped whisker instead, with the band kept for two epochs or more.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["segments"] = [
                (l.get_xdata(), l.get_ydata()) for l in fig.axes[1].lines
                if l.get_alpha() is not None
            ]
            captured["labels"] = fig.axes[1].get_legend_handles_labels()[1]
            captured["wilson"] = [
                l for l in fig.axes[0].lines if l.get_alpha() is not None
            ]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows()[:1], tmp_path / "p.png")

    verticals = [(xs, ys) for xs, ys in captured["segments"]
                 if len(ys) == 2 and ys[0] != ys[1]]
    assert verticals, "a single epoch drew no visible interval"
    for _xs, ys in verticals:
        assert ys[1] > ys[0], "the whisker has no extent"
    # The legend is unchanged, so the panel reads the same as a longer run's.
    assert sum("95% CI" in l for l in captured["labels"]) == 2
    assert captured["wilson"], "the valid-rate panel lost its interval too"


def test_many_epochs_keep_the_shaded_band(tmp_path, monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["areas"] = [
                max(v[1] for v in p.vertices) - min(v[1] for v in p.vertices)
                for coll in fig.axes[1].collections for p in coll.get_paths()
            ]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")
    assert captured["areas"] and max(captured["areas"]) > 0


# --- seed models are reported together with the dataset they were trained on --


def test_seed_models_carry_their_dataset(tmp_path):
    pd = pytest.importorskip("pandas")
    d = tmp_path / "A0" / "synth_nn"
    d.mkdir(parents=True)
    for i, ds in enumerate(("cifar-10", "cifar-100")):
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": 0.3, "dataset": ds}).to_pickle(
            md / "dataframe.df")
        (md / "eval_info.json").write_text(
            json.dumps({"eval_results": {"accuracy": 0.4}}))

    stats = ES.collect_epoch_stats(d, 0)

    assert stats["baseline_models"] == ["seed-0", "seed-1"]
    assert stats["baseline_models_with_dataset"] == [
        "seed-0 (cifar-10)", "seed-1 (cifar-100)"]
    assert stats["datasets"] == ["cifar-10", "cifar-100"]
    assert {s["nn"]: s["dataset"] for s in stats["baseline_seeds"]} == {
        "seed-0": "cifar-10", "seed-1": "cifar-100"}


def test_a_seed_without_a_dataset_is_still_named(tmp_path):
    pd = pytest.importorskip("pandas")
    d = tmp_path / "A0" / "synth_nn"
    (d / "B0").mkdir(parents=True)
    pd.Series({"nn": "seed-0", "accuracy": 0.3}).to_pickle(d / "B0" / "dataframe.df")
    (d / "B0" / "eval_info.json").write_text(
        json.dumps({"eval_results": {"accuracy": 0.4}}))

    stats = ES.collect_epoch_stats(d, 0)
    assert stats["baseline_models_with_dataset"] == ["seed-0"]
    assert stats["datasets"] == []


# --- a new run starts from an empty directory -------------------------------


def test_reset_run_dir_removes_the_whole_previous_run(tmp_path):
    run = tmp_path / "nngpt"
    run.mkdir()
    _seed_run_dir(run, epochs=range(3))
    (run / "generation_seeds.json").write_text('{"nn": ["a"]}')
    (run / "llm" / "epoch" / "A0" / "synth_nn" / "B0").mkdir(parents=True)
    (run / "llm" / "epoch" / "A0" / "synth_nn" / "B0" / "new_nn.py").write_text("x = 1")

    assert ES.reset_run_dir(run) == run

    assert run.is_dir()                      # recreated, so the run can write
    assert list(run.iterdir()) == []         # and nothing of the old run is left


def test_reset_run_dir_is_a_noop_on_a_missing_directory(tmp_path):
    assert ES.reset_run_dir(tmp_path / "absent") is None


# --- the caption: what was run, stated once ------------------------------

_FULL_TRAINING = {
    "learning_rate": 1e-05, "lr_scheduler": "cosine", "warmup_steps": 5,
    "num_train_epochs": 1, "cycles": 4, "optimizer": "paged_adamw_8bit",
    "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.05,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
    "max_prompts": 512, "sft_max_length": 4096, "context_length": 8192,
    "max_new_tokens": 8192, "test_nn": 16, "load_in_4bit": True,
    "unsloth": True, "vllm_sleep": True,
}


#: The same run with the sampling settings recorded, as Tune.py writes them.
_SAMPLING_TRAINING = dict(_FULL_TRAINING, temperature=0.8, top_k=70, top_p=0.9)


def test_param_block_states_every_hyper_parameter_on_one_line():
    """All of them, once -- a setting left out cannot be recovered by a reader."""
    lines = ES._param_block(_FULL_TRAINING)
    assert len(lines) == 1, f"configuration spread over {len(lines)} lines"
    text = lines[0]
    for key, value in _FULL_TRAINING.items():
        if key in ES.PARAM_SKIP or isinstance(value, bool):
            continue
        assert f"{ES.PARAM_LABEL.get(key, key)}={ES._fmt_value(value)}" in text, key
    assert len(text) <= ES.PARAM_MAX_CHARS


def test_param_block_states_the_sampling_settings():
    """temperature and top_p decide what the LLM emits, so the curves move with
    them; a figure that leaves them out cannot be reproduced from its caption."""
    text = " ".join(ES._param_block(_SAMPLING_TRAINING))
    assert "temp=0.8" in text
    assert "top_k=70" in text
    assert "top_p=0.9" in text


def test_the_line_opens_on_what_is_tuned_first():
    """lr, dropout and batch size are the three a reader reaches for first, so
    they open the line instead of being hunted for in the middle of it."""
    cells = ES._param_block(_SAMPLING_TRAINING)[0].split()
    assert [c.split("=", 1)[0] for c in cells[:3]] == ["lr", "dropout", "batch"]


def test_the_token_budgets_stand_together_before_the_sampling_settings():
    """They are one quantity in three places and only mean anything read
    against each other, so they are not scattered through the line."""
    cells = ES._param_block(_SAMPLING_TRAINING)[0].split()
    names = [c.split("=", 1)[0] for c in cells]
    where = [names.index(ES.PARAM_LABEL.get(k, k)) for k in ES.PARAM_TOKENS]
    assert where == list(range(where[0], where[0] + len(ES.PARAM_TOKENS))), \
        f"the token budgets are split apart: {cells}"
    assert where[-1] < names.index("temp")


def test_the_sampling_settings_come_last():
    """They act at generation time, not during fine-tuning, so they sit after
    every training setting instead of among the optimiser's numbers."""
    lines = ES._param_block(_SAMPLING_TRAINING)
    assert len(lines) == 1, f"configuration spread over {len(lines)} lines"
    assert lines[0].endswith("temp=0.8 top_k=70 top_p=0.9")


def test_the_full_configuration_including_sampling_stays_on_one_line():
    """Two lines read as two unrelated statements; the configuration is one."""
    lines = ES._param_block(_SAMPLING_TRAINING)
    assert len(lines) == 1
    assert len(lines[0]) <= ES.PARAM_MAX_CHARS


def test_the_configuration_line_clears_the_panels():
    """PARAM_MAX_CHARS is a width budget, so it is only correct paired with the
    size the line is drawn at. A full-width line must stay inside the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 9))
    try:
        text = fig.text(0.5, 0.5, "x" * ES.PARAM_MAX_CHARS, ha="center",
                        family="monospace", fontsize=ES.PARAM_FONTSIZE)
        width = text.get_window_extent(
            renderer=fig.canvas.get_renderer()).width / fig.bbox.width
    finally:
        plt.close(fig)
    assert width <= 0.96, f"a full line spans {width:.0%} of the figure"


def test_param_block_omits_what_the_figure_already_shows():
    """test_nn is the denominator of every label on the upper panel."""
    text = " ".join(ES._param_block(_FULL_TRAINING))
    for key in ("test_nn", "unsloth", "vllm_sleep"):
        assert ES.PARAM_LABEL.get(key, key) not in text, f"{key} is still shown"


def test_the_plan_is_stated_even_though_the_x_axis_counts_the_cycles():
    """The axis counts the cycles that finished; only the plan says how many
    were meant to. Three points under cycles=10 is an unfinished run, and
    nothing else on the figure distinguishes that from a run of three."""
    cells = ES._param_block(dict(_SAMPLING_TRAINING, cycles=10))[0].split()
    assert "cycles=10" in cells
    assert cells[cells.index("cycles=10") + 1].startswith("ep/cyc="), \
        f"the epochs per cycle do not follow the plan: {cells}"


def test_the_plan_follows_the_settings_it_was_run_with():
    """It is the length of the run, not one more knob to compare with a rank,
    so it comes after every optimiser setting rather than among them."""
    cells = ES._param_block(_SAMPLING_TRAINING)[0].split()
    names = [c.split("=", 1)[0] for c in cells]
    assert names.index("cycles") > names.index("opt")
    assert names.index("ep/cyc") < names.index("sft_len")


def test_a_flag_is_named_when_set_and_absent_when_not():
    assert "4bit" in " ".join(ES._param_block(_FULL_TRAINING))
    off = dict(_FULL_TRAINING, load_in_4bit=False)
    assert "4bit" not in " ".join(ES._param_block(off))


def test_named_values_are_collected_after_the_numbers():
    """Numbers are compared against their neighbours; a name breaks the run.

    The names are contiguous rather than last: the trailing groups are numbers
    too, and they are set down after the names on purpose.
    """
    cells = ES._param_block(_FULL_TRAINING)[0].split()
    named = [i for i, c in enumerate(cells)
             if not ES._is_numeric(c.split("=", 1)[-1])]
    assert named, "no named setting in the fixture"
    assert named == list(range(named[0], named[-1] + 1)), \
        f"a named setting sits among the numbers: {cells}"
    trailing = {k for group in ES.PARAM_TRAILING for k in group}
    after = cells[named[-1] + 1:]
    assert after, "nothing follows the names"
    labels = {ES.PARAM_LABEL.get(k, k) for k in trailing}
    assert all(c.split("=", 1)[0] in labels for c in after), \
        f"a plain setting sits after the names: {after}"


def test_param_block_keeps_an_unrecognised_setting():
    """A setting added to run_config.json later still reaches the figure."""
    assert "rope_scaling=yarn" in " ".join(
        ES._param_block(dict(_FULL_TRAINING, rope_scaling="yarn")))


def test_param_block_wraps_rather_than_truncating():
    lines = ES._param_block(_FULL_TRAINING, max_chars=40)
    assert len(lines) == 2
    assert "opt=paged_adamw_8bit" in " ".join(lines)


# --- the improvement panel: what the LLM changed -------------------------

#: Per-model (original, generated) pairs, as collect_epoch_stats records them
#: in baseline_seeds. Cycle 2 improves on cycle 1 on every model.
_SEEDS = [
    [(0.55, 0.50), (0.60, 0.58), (0.58, 0.61), (0.62, 0.59)],
    [(0.55, 0.59), (0.60, 0.63), (0.58, 0.57), (0.62, 0.66)],
]


def _paired_rows(seeds=True, **overrides):
    """_plot_rows with the pairing collect_epoch_stats records.

    ``seeds=False`` is a tracker written before baseline_seeds existed: the
    summary difference is there, the per-model pairs are not.
    """
    deltas = [(-0.04, -0.09, 0.01), (0.03, -0.01, 0.07)]
    rows = []
    for r, (mean, lo, hi), pairs in zip(_plot_rows(), deltas, _SEEDS):
        row = dict(r, paired_n=len(pairs), mean_improvement=mean,
                   improvement_ci_low=lo, improvement_ci_high=hi, **overrides)
        if seeds:
            row["baseline_seeds"] = [
                {"nn": f"net{i}", "dataset": "cifar-10",
                 "baseline_accuracy": before, "accuracy": after}
                for i, (before, after) in enumerate(pairs)]
        rows.append(row)
    return rows


def _panel_titles(tmp_path, monkeypatch, rows, name="p.png"):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and fig.axes:
            captured["titles"] = [ax.get_title() for ax in fig.axes]
            captured["n"] = len(fig.axes)
            captured["labels"] = [
                ax.get_legend_handles_labels()[1] for ax in fig.axes]
            captured["ylim"] = fig.axes[-1].get_ylim()
            captured["xlabel"] = [ax.get_xlabel() for ax in fig.axes]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / name)
    return captured


def test_no_improvement_panel_without_an_initial_network(tmp_path, monkeypatch):
    """A run that generates from scratch has nothing to subtract, so it keeps
    the two panels rather than gaining a third one full of blanks."""
    out = _panel_titles(tmp_path, monkeypatch, _plot_rows())
    assert out["n"] == 2
    assert not any(t.startswith("\u0394") for t in out["titles"])


def test_the_improvement_panel_appears_when_the_pairing_exists(
        tmp_path, monkeypatch):
    """collect_epoch_stats pairs each generated model with the seed it was
    modified from; where that pairing exists, so does the question."""
    out = _panel_titles(tmp_path, monkeypatch, _paired_rows())
    assert out["n"] == 3
    assert out["titles"][2].startswith("\u0394 accuracy per cycle")


def test_the_improvement_panel_shows_the_mean_and_the_best(
        tmp_path, monkeypatch):
    """The paired mean with its interval, and the gap between the two best
    networks -- the two differences the run is judged on."""
    labels = _panel_titles(tmp_path, monkeypatch, _paired_rows())["labels"][2]
    joined = " | ".join(labels)
    assert "paired mean \u0394" in labels, joined
    assert "paired mean \u0394 95% CI" in labels, joined
    assert "best \u0394 (observed)" in labels, joined
    assert any(l.startswith("best-of-") for l in labels), joined
    assert not [l for l in labels if "before modification" in l], \
        "zero is where the original networks sit; naming it spends a row"


def test_an_older_tracker_keeps_what_it_can(tmp_path, monkeypatch):
    """Without the per-model pairs there is no win rate and no best-of-k --
    both are computed from them -- but the summary difference is still drawn
    rather than the whole panel being withheld."""
    out = _panel_titles(tmp_path, monkeypatch, _paired_rows(seeds=False),
                        "old.png")
    assert out["n"] == 3
    delta_labels = out["labels"][2]
    assert "paired mean \u0394" in delta_labels
    assert not [l for l in delta_labels if l.startswith("best-of-")]




def test_best_of_k_takes_the_sample_size_out():
    """The maximum of twelve models beats the maximum of six for no reason but
    the extra draws; the expected best of a fixed k does not."""
    small = [0.50, 0.55, 0.60]
    large = small + [0.52, 0.57, 0.59, 0.61, 0.48]
    assert max(large) > max(small)
    assert ES.expected_best_of_k(large, 3) == pytest.approx(
        ES.expected_best_of_k(small, 3), abs=0.04), \
        "best-of-k still tracks the sample size"


def test_expected_best_of_k_matches_enumerating_every_subset():
    """It is the mean of the subset maxima, computed from the order
    statistics instead of by listing C(n, k) subsets."""
    import itertools
    import math as _math
    values = [0.11, 0.37, 0.42, 0.58, 0.63, 0.64]
    for k in range(1, len(values) + 1):
        brute = sum(max(c) for c in itertools.combinations(values, k))
        brute /= _math.comb(len(values), k)
        assert ES.expected_best_of_k(values, k) == pytest.approx(brute)
    assert ES.expected_best_of_k(values, 1) == pytest.approx(
        sum(values) / len(values))                      # k=1 is the mean
    assert ES.expected_best_of_k(values, len(values)) == max(values)
    assert ES.expected_best_of_k([], 3) is None


def test_k_is_the_smallest_cycle_so_the_cycles_compare():
    """A best-of-5 cannot be formed where three models paired, and falling
    back to the maximum there would reintroduce the dependence on n."""
    rows = _paired_rows()
    rows[0]["baseline_seeds"] = rows[0]["baseline_seeds"][:2]
    assert ES._common_k(rows) == 2
    assert ES._common_k(_paired_rows(seeds=False)) is None




def _figure_note_text(tmp_path, monkeypatch, rows, name):
    """The note as drawn, its lines joined -- it is now one text per line."""
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None:
            captured["note"] = "\n".join(
                t.get_text() for t in fig.texts
                if t.get_color() == ES.NOTE_COLOR
                and t.get_fontsize() == ES.NOTE_FONTSIZE)
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / name)
    return captured.get("note", "")


def test_the_best_the_run_started_from_is_carried_forward(
        tmp_path, monkeypatch):
    """Each cycle draws different seeds, so the per-cycle original best moves
    around: a generated network clearing the grey square beside it has not
    necessarily cleared the best the run was ever given. The grey running
    line is the mark to beat."""
    labels = _panel_titles(tmp_path, monkeypatch, _plot_rows())["labels"][1]
    assert "original best so far" in labels, labels
    assert "generated best so far" in labels, labels


def _peak_labels(tmp_path, monkeypatch, rows, name):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and len(fig.axes) > 1:
            fig.canvas.draw()
            captured["marks"] = [
                (t.get_text(), t.get_color(), t.get_window_extent())
                for t in fig.axes[1].texts if t.get_text().startswith("best ")]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / name)
    return captured.get("marks", [])


def test_both_headline_bests_are_labelled(tmp_path, monkeypatch):
    """The red number alone says how good the run got without saying what it
    had to beat, which is the comparison the panel exists for."""
    marks = _peak_labels(tmp_path, monkeypatch, _plot_rows(), "peaks.png")
    colours = {colour for _, colour, _ in marks}
    assert ES.BEST in colours, marks
    assert ES.ORIGINAL in colours, marks
    best = {colour: text for text, colour, _ in marks}
    assert best[ES.BEST] == "best 0.5895"           # max accuracy_max
    assert best[ES.ORIGINAL] == "best 0.6100"       # max baseline_accuracy_max


def test_the_two_peak_labels_never_land_on_each_other(tmp_path, monkeypatch):
    """Whichever cohort is higher at that cycle takes the space above and the
    other takes the space below, so both cohorts peaking together is safe."""
    rows = [dict(r, accuracy_max=0.70, baseline_accuracy_max=0.66)
            for r in _plot_rows()]
    rows[0] = dict(rows[0], accuracy_max=0.40, baseline_accuracy_max=0.40)
    marks = _peak_labels(tmp_path, monkeypatch, rows, "together.png")
    assert len(marks) == 2, marks
    (_, _, first), (_, _, second) = marks
    assert not first.overlaps(second), f"{first} overlaps {second}"


def test_a_running_best_is_drawn_over_the_per_cycle_one(tmp_path, monkeypatch):
    """They coincide for most of a run -- for all of it on the originals when
    the seed pool does not change -- and underneath, the dotted line is hidden
    by the dashed one exactly when it agrees with it, which reads as missing.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and len(fig.axes) > 1:
            captured["z"] = {l.get_label(): l.get_zorder()
                             for l in fig.axes[1].lines}
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "z.png")
    z = captured["z"]
    assert z["original best so far"] > z["original best"]
    assert z["generated best so far"] > z["generated best"]


def test_both_running_bests_only_ever_climb(tmp_path, monkeypatch):
    """A running best is a record: it cannot fall when a later cycle is
    worse, which is the whole point of drawing it beside the per-cycle one."""
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and len(fig.axes) > 1:
            captured["runs"] = [
                list(line.get_ydata()) for line in fig.axes[1].lines
                if line.get_linestyle() in (":", (0, (1, 1.65)))]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    # cycle 2 is worse than cycle 1 on both cohorts
    rows = _plot_rows()[:2]
    rows[1] = dict(rows[1], accuracy_max=0.30, baseline_accuracy_max=0.30)
    ES.plot_epoch_stats(rows, tmp_path / "run.png")
    runs = captured.get("runs") or []
    assert len(runs) == 2, f"expected two running bests, got {len(runs)}"
    for series in runs:
        assert series == sorted(series), f"a running best fell: {series}"


def test_the_note_glosses_only_what_cannot_be_inferred(
        tmp_path, monkeypatch):
    """A mean, a maximum and a Wilson interval are terms a reader already has.
    Glossing them would bury the one line that is worth reading."""
    note = _figure_note_text(tmp_path, monkeypatch, _paired_rows(), "n.png")
    assert "est-of-" in note
    for standard in ("Wilson", "paired mean \u0394:", "confidence interval:"):
        assert standard not in note, f"{standard!r} needs no gloss: {note}"


def test_the_note_carries_the_formula_and_where_it_is_defined(
        tmp_path, monkeypatch):
    """A name alone cannot be checked. The note gives the expression, the
    formal name to search the reference for, and the reference."""
    note = _figure_note_text(tmp_path, monkeypatch, _paired_rows(), "f.png")
    assert r"\binom{i-1}{k-1}" in note, note       # the formula, as a formula
    assert r"\binom{n}{k}" in note, note
    assert "U-statistic with the maximum as its kernel" in note, note
    assert "Hoeffding, 1948" in note, note
    assert "293-325" in note, note


def test_no_note_where_nothing_is_unfamiliar(tmp_path, monkeypatch):
    assert _figure_note_text(
        tmp_path, monkeypatch, _plot_rows(), "n2.png") == ""


def test_the_prose_is_wrapped_and_the_formulas_are_not(tmp_path, monkeypatch):
    """Half of a sum is not a formula, so only the prose is broken -- and the
    prose is broken, since unwrapped it would run off both sides."""
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plt.figure(figsize=(10, 12.6))
    formula = r"$E_k = \sum_{i=k}^{n} x_{(i)}$"
    source = (formula, "a sentence that keeps going " * 12)
    try:
        lines = ES._wrap_note(fig, source)
        assert formula in lines, "the formula was broken up"
        assert len(lines) > 2, "the long prose line was not wrapped"
        renderer = fig.canvas.get_renderer()
        for line in lines:
            probe = fig.text(0, 0, line, fontsize=ES.NOTE_FONTSIZE)
            width = probe.get_window_extent(
                renderer=renderer).width / fig.bbox.width
            probe.remove()
            assert width <= ES.NOTE_WIDTH, f"{width:.0%}: {line}"
        # the note the figure actually carries also stays inside the width
        for line in ES._wrap_note(fig, ES._figure_note(True, 5)):
            probe = fig.text(0, 0, line, fontsize=ES.NOTE_FONTSIZE)
            width = probe.get_window_extent(
                renderer=renderer).width / fig.bbox.width
            probe.remove()
            assert width <= ES.NOTE_WIDTH, f"{width:.0%}: {line}"
    finally:
        plt.close(fig)


def test_the_note_lines_do_not_collide(tmp_path, monkeypatch):
    """A formula stands three times the height of a word, so the lines are
    stacked by measurement; on one leading the sum sat across the prose."""
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None:
            fig.canvas.draw()
            boxes = [t.get_window_extent() for t in fig.texts
                     if t.get_color() == ES.NOTE_COLOR
                     and t.get_fontsize() == ES.NOTE_FONTSIZE]
            captured["spans"] = sorted((b.y0, b.y1) for b in boxes)
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_paired_rows(), tmp_path / "collide.png")
    spans = captured["spans"]
    assert len(spans) > 1
    assert spans[0][0] >= 0, "the last line falls off the figure"
    for (_, lower_top), (upper_bottom, _) in zip(spans, spans[1:]):
        assert lower_top <= upper_bottom + 0.5, \
            f"two note lines overlap: {spans}"


def test_the_best_of_k_interval_is_a_bootstrap_over_the_pairs(tmp_path,
                                                              monkeypatch):
    """An interval is possible here and not for the observed maximum: this is
    an average over the order statistics, which the bootstrap handles."""
    labels = _panel_titles(tmp_path, monkeypatch, _paired_rows())["labels"][2]
    assert any(l.startswith("best-of-") and "CI" in l for l in labels), labels
    assert not any(l.startswith("best \u0394 (observed)") and "CI" in l
                   for l in labels), "the observed maximum must carry none"


def test_the_best_of_k_interval_is_the_same_on_every_redraw():
    """A bootstrap seeded from the clock would redraw a different interval
    each time the plot was refreshed, which reads as the run changing."""
    pairs = [(0.60, 0.63), (0.58, 0.57), (0.62, 0.66), (0.55, 0.59),
             (0.61, 0.60), (0.59, 0.64)]
    first = ES._best_of_k_ci(pairs, 3)
    assert first == ES._best_of_k_ci(pairs, 3)
    lo, hi = first
    assert lo < ES._best_of_k_from_pairs(pairs, 3) < hi


def test_the_bootstrap_resamples_pairs_not_cohorts():
    """A resample has to be a plausible redrawing of the same experiment, so
    a model keeps the original it was made from."""
    pairs = [(0.50, 0.90), (0.90, 0.50)]          # perfectly anti-correlated
    lo, hi = ES._best_of_k_ci(pairs, 1)
    # Resampling the two columns independently would allow (0.90, 0.90) and
    # (0.50, 0.50) draws, widening this well past the paired range.
    assert lo >= -0.4 and hi <= 0.4, (lo, hi)


def test_pooling_grows_the_sample_the_tail_rests_on():
    """One cycle's best-of-k rests on about ten models; the pooled figure
    rests on everything the run has produced up to that cycle."""
    rows = _paired_rows()
    pooled = ES._cumulative_pairs(rows)
    assert [len(p) for p in pooled] == [4, 8]
    assert pooled[1][:4] == pooled[0]


def test_the_improvement_panel_always_shows_zero(tmp_path, monkeypatch):
    """Which side of zero the curve falls on is the whole question, and a
    panel cropped above zero answers it without saying so."""
    for shift in (0.5, -0.5):          # entirely above, entirely below
        rows = [dict(r, mean_improvement=r["mean_improvement"] + shift,
                     improvement_ci_low=r["improvement_ci_low"] + shift,
                     improvement_ci_high=r["improvement_ci_high"] + shift)
                for r in _paired_rows()]
        lo, hi = _panel_titles(tmp_path, monkeypatch, rows,
                               f"z{shift}.png")["ylim"]
        assert lo <= 0.0 <= hi, f"zero is off the panel: {(lo, hi)}"


def test_the_improvement_panel_borrows_panel_twos_colours(
        tmp_path, monkeypatch):
    """Same quantity, same reading: green is what the LLM produced, the dark
    red is its best, grey is the networks it started from."""
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and len(fig.axes) == 3:
            captured["colours"] = [l.get_color() for l in fig.axes[2].lines]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_paired_rows(), tmp_path / "c.png")
    colours = captured.get("colours", [])
    for expected in ("#2ca02c", "#9d0208", "#7f7f7f"):
        assert expected in colours, f"{expected} missing from {colours}"


def test_the_header_band_is_the_same_size_on_both_figures(
        tmp_path, monkeypatch):
    """The header is positioned in figure fractions, which shrink as the
    figure grows: a third panel would have squeezed the caption. Measured in
    inches, the band is identical whether or not the panel is there."""
    plt = pytest.importorskip("matplotlib.pyplot")

    def _band(rows, name):
        captured, real_close = {}, plt.close

        def _capture(fig=None):
            if fig is not None and fig.axes:
                fig.canvas.draw()
                top = fig.bbox.y1
                captured["panel"] = round(
                    (top - fig.axes[0].get_window_extent().y1) / fig.dpi, 3)
            return real_close(fig)

        monkeypatch.setattr(plt, "close", _capture)
        ES.plot_epoch_stats(rows, tmp_path / name)
        return captured["panel"]

    assert _band(_plot_rows(), "two.png") == _band(_paired_rows(), "three.png")


def test_the_cycle_axis_stays_on_the_bottom_panel(tmp_path, monkeypatch):
    """The x axis belongs to whichever panel is last, not to panel 2."""
    out = _panel_titles(tmp_path, monkeypatch, _paired_rows())
    assert out["xlabel"][-1] == ES.AXIS_LABEL
    assert out["xlabel"][1] == "", "panel 2 kept the axis label"


# --- the prompts: which configs produced these curves --------------------

_STUDY = {"task": "img-classification", "dataset": ["cifar-10"]}


def test_both_prompt_configs_are_named():
    """Identical hyper-parameters on different prompts give different curves,
    and nothing else on the figure tells the two runs apart."""
    phrase = ES._prompt_phrase(dict(_STUDY, prompts={
        "train": "Transform_gen.json", "test": "NN_gen.json"}))
    assert phrase == "prompts train Transform_gen, test NN_gen"


def test_one_config_in_both_roles_is_stated_once():
    """The common case is a single file used for both; naming it twice spends
    width on a repetition and reads as two different things."""
    assert ES._prompt_phrase(dict(_STUDY, prompts={
        "train": "NN_gen.json", "test": "NN_gen.json"})) == "prompts NN_gen"


@pytest.mark.parametrize("prompts, expected", [
    ({"test": "NN_gen.json"}, "prompts test NN_gen"),
    ({"train": "NN_gen.json"}, "prompts train NN_gen"),
    ({}, ""),
    (None, ""),
])
def test_a_missing_prompt_config_is_left_out(prompts, expected):
    """An older run_config.json records neither; the line is as it was."""
    assert ES._prompt_phrase(dict(_STUDY, prompts=prompts)) == expected


def test_the_study_line_carries_the_prompts():
    line = ES._study_sentence(dict(_STUDY, prompts={
        "train": "Transform_gen.json", "test": "NN_gen.json"}), [])
    assert line.startswith("Automated generation")
    assert line.endswith("prompts train Transform_gen, test NN_gen")


def test_a_long_study_line_loses_point_size_not_a_second_row(tmp_path):
    """The header band is a fixed height and the configuration line below is
    already at the floor of its own size, so width is bought with points."""
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plt.figure(figsize=(10, 9))
    try:
        short = ES._study_sentence(_STUDY, [])
        long = ES._study_sentence(dict(
            _STUDY, dataset=["cifar-10", "cifar-100", "svhn", "imagenet"],
            prompts={"train": "Transform_gen_channel_alter_v2.json",
                     "test": "NN_gen_improve_classification_only_v3.json"}), [])
        assert ES._fit_fontsize(fig, short, style="italic") == ES.STUDY_FONTSIZES[0]
        chosen = ES._fit_fontsize(fig, long, style="italic")
        assert chosen < ES.STUDY_FONTSIZES[0], "a line that overflows kept its size"

        text = fig.text(0.5, 0.5, long, ha="center", fontsize=chosen,
                        style="italic")
        width = text.get_window_extent(
            renderer=fig.canvas.get_renderer()).width / fig.bbox.width
        assert width <= ES.STUDY_MAX_WIDTH, f"study line spans {width:.0%}"
        assert "\n" not in long
    finally:
        plt.close(fig)


def test_naming_the_prompts_costs_the_panels_no_room(tmp_path, monkeypatch):
    """"Without extra space" is the requirement: the panels must sit exactly
    where they sat before the prompts were added to the line."""
    plt = pytest.importorskip("matplotlib.pyplot")

    def _panel_box(config):
        captured, real_close = {}, plt.close

        def _capture(fig=None):
            if fig is not None and fig.axes:
                fig.canvas.draw()
                captured["box"] = fig.axes[0].get_position().bounds
            return real_close(fig)

        monkeypatch.setattr(plt, "close", _capture)
        base = tmp_path / str(abs(hash(str(config))))
        base.mkdir(parents=True, exist_ok=True)
        (base / "run_config.json").write_text(json.dumps(config))
        ES.plot_epoch_stats(_plot_rows(), base / "p.png", base_dir=base)
        return captured["box"]

    without = _panel_box(dict(_STUDY))
    with_prompts = _panel_box(dict(_STUDY, prompts={
        "train": "Transform_gen.json", "test": "NN_gen.json"}))
    assert without == pytest.approx(with_prompts), \
        f"the panels moved: {without} -> {with_prompts}"


def test_study_sentence_uses_the_names_the_dataset_records():
    """Verbatim, so the figure can be matched against the database."""
    assert ES._study_sentence({"task": "img-classification", "dataset": "cifar-10"}, []) == (
        "Automated generation of neural network architectures "
        "for img-classification on cifar-10")


def test_study_sentence_takes_the_dataset_from_the_rows_when_absent():
    sentence = ES._study_sentence({"task": "img-classification"},
                                  [{"datasets": ["cifar-10"]}, {"datasets": ["cifar-10"]}])
    assert sentence.endswith("for img-classification on cifar-10")


def test_study_sentence_survives_a_task_it_has_never_seen():
    sentence = ES._study_sentence({"task": "depth-estimation", "dataset": "nyu-v2"}, [])
    assert "depth-estimation" in sentence and "nyu-v2" in sentence


def test_study_sentence_without_a_configuration_still_reads_as_a_sentence():
    assert ES._study_sentence({}, []) == \
        "Automated generation of neural network architectures"


def test_legend_corners_are_square(tmp_path, monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["rounded"] = [
                ax.get_legend().legendPatch.get_boxstyle().__class__.__name__
                for ax in fig.axes if ax.get_legend() is not None]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")
    assert captured["rounded"], "no legend was drawn"
    assert all(name == "Square" for name in captured["rounded"]), captured["rounded"]


def test_brand_sits_in_the_top_right_corner(tmp_path, monkeypatch):
    """Top-right, so a crop that removes it also removes the run's identity.

    The title and the configuration occupy the same band; the mark cannot be
    cropped away without taking part of them with it.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "texts"):
            captured["marks"] = [(t.get_position(), t.get_ha(), t.get_va())
                                 for t in fig.texts if t.get_text() == ES.BRAND_TEXT]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")

    marks = captured.get("marks")
    assert marks and len(marks) == 1, "the project mark was not drawn exactly once"
    (x, y), ha, va = marks[0]
    assert x > 0.9 and y > 0.9, f"mark at {(x, y)}, not the top-right corner"
    assert ha == "right" and va == "top"


# --- the run time: what the run cost, stated once ------------------------

_T0 = datetime(2026, 10, 18, 9, 0)


def _timed_rows(*offsets):
    """Epoch rows stamped at the given hours after _T0."""
    return [{"epoch": i, "timestamp": (_T0 + timedelta(hours=h)).isoformat()}
            for i, h in enumerate(offsets, 1)]


@pytest.mark.parametrize("seconds, expected", [
    (0, "<1m"), (59, "<1m"),                       # below the resolution shown
    (60, "1m"), (20 * 60, "20m"),
    (3600, "1h.00m"), (3 * 3600 + 12 * 60, "3h.12m"),
    (86400, "1d.00h.00m"), (2 * 86400 + 7 * 3600 + 41 * 60, "2d.07h.41m"),
])
def test_a_duration_reads_as_days_hours_minutes(seconds, expected):
    """An empty leading unit is a zero to count past, so it is left off."""
    assert ES._fmt_duration(seconds) == expected


def test_the_run_time_counts_from_the_start_of_the_run():
    """The first epoch is generation, training and evaluation -- the most
    expensive one. Measured between epoch rows it would not be counted."""
    rows = _timed_rows(11, 22, 33)
    config = {"started_at": _T0.isoformat()}
    assert ES._fmt_duration(ES._run_duration(config, rows)) == "1d.09h.00m"
    # ...which is the span of the rows plus the 11 hours before the first one.
    assert ES._fmt_duration(ES._run_duration({}, rows)) == "22h.00m"


def test_the_run_time_ends_at_the_last_epoch_not_at_the_clock():
    """Re-drawing the figure later must report what the run cost, not how long
    ago it ran, so the end is a recorded stamp and never datetime.now()."""
    rows = _timed_rows(0, 5)
    config = {"started_at": _T0.isoformat()}
    assert ES._run_duration(config, rows) == 5 * 3600


def test_a_resumed_run_covers_every_epoch_it_plots():
    """--peft stamps a new start over the old one while the tracker keeps the
    earlier epochs. The duration spans what is drawn, and cannot go negative."""
    rows = _timed_rows(1, 2, 30)
    config = {"started_at": (_T0 + timedelta(hours=20)).isoformat()}
    seconds = ES._run_duration(config, rows)
    assert seconds == 29 * 3600


def test_the_run_time_is_absent_rather_than_wrong_without_stamps():
    """Nothing recorded means nothing is claimed."""
    assert ES._run_duration({}, [{"epoch": 1}]) is None
    assert ES._run_duration({"started_at": "whenever"},
                            [{"timestamp": "whenever"}]) is None
    assert ES._fmt_duration(None) == ""


@pytest.mark.parametrize("seconds, expected", [
    (30, "<1m"), (45 * 60, "45m"), (89 * 60, "89m"),
    (90 * 60, "1.5h"), (2.4 * 3600, "2.4h"),
    (6 * 3600, "6h"),                       # never a trailing ".0"
    (9.9 * 3600, "9.9h"), (11 * 3600, "11h"), (35 * 3600, "35h"),
    (36 * 3600, "1.5d"), (12 * 86400, "12d"),
    (None, ""), (-5, ""),
])
def test_a_cycle_span_is_one_number_wide(seconds, expected):
    """It is read in passing, under a tick, so it carries one unit."""
    assert ES._fmt_span(seconds) == expected


def test_the_first_cycle_is_measured_from_the_start_of_the_run():
    """Its own beginning is recorded nowhere else: measured off the row before
    it, which does not exist, the most expensive cycle would show nothing."""
    rows = _timed_rows(11, 23, 34)
    spans = ES._cycle_durations({"started_at": _T0.isoformat()}, rows)
    assert [ES._fmt_span(s) for s in spans] == ["11h", "12h", "11h"]
    assert ES._fmt_span(ES._cycle_durations({}, rows)[0]) == ""


def test_a_missing_stamp_costs_both_cycles_that_share_it():
    """Carrying the last good stamp forward would charge the whole gap to the
    next cycle: a stats run repeated days later as days of work."""
    rows = [{"timestamp": "unreadable"}] + _timed_rows(11, 23, 34)[1:]
    spans = ES._cycle_durations({"started_at": _T0.isoformat()}, rows)
    assert [ES._fmt_span(s) for s in spans] == ["", "", "11h"]


def test_a_cycle_span_is_drawn_under_each_tick(tmp_path, monkeypatch):
    """Under the cycle number it belongs to, below the axis, never over data."""
    plt = pytest.importorskip("matplotlib.pyplot")
    rows = [dict(r, timestamp=(_T0 + timedelta(hours=9 * (i + 1))).isoformat())
            for i, r in enumerate(_plot_rows())]
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and fig.axes:
            ax = fig.axes[-1]
            captured["spans"] = [(t.get_text(), t.get_position())
                                 for t in ax.texts
                                 if t.get_color() == ES.SPAN_COLOR]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / "p.png")
    spans = captured.get("spans", [])
    assert len(spans) == len(rows) - 1, f"one per stamped cycle: {spans}"
    assert all(y < 0 for _, (x, y) in spans), f"a span is inside the axes: {spans}"


def test_no_cycle_spans_are_drawn_when_the_rows_are_unstamped(
        tmp_path, monkeypatch):
    """An older tracker has no timestamps, and the axis is left as it was."""
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and fig.axes:
            captured["pad"] = fig.axes[-1].xaxis.labelpad
            captured["texts"] = [t.get_text() for t in fig.axes[-1].texts
                                 if t.get_color() == ES.SPAN_COLOR]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")
    assert not captured.get("texts"), captured.get("texts")
    assert captured["pad"] < ES.SPAN_RESERVE, "room was reserved for nothing"


def test_the_axis_label_clears_the_durations(tmp_path, monkeypatch):
    """Measured on the rendered figure, in millimetres, because the gap
    depends on font metrics no constant here can predict. Too close and the
    durations and the label read as one block."""
    plt = pytest.importorskip("matplotlib.pyplot")
    rows = [dict(r, timestamp=(_T0 + timedelta(hours=9 * (i + 1))).isoformat())
            for i, r in enumerate(_plot_rows())]
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and fig.axes:
            ax = fig.axes[-1]
            fig.canvas.draw()
            spans = [t for t in ax.texts if t.get_color() == ES.SPAN_COLOR]
            if spans:
                bottom = min(t.get_window_extent().y0 for t in spans)
                top = ax.xaxis.label.get_window_extent().y1
                captured["mm"] = (bottom - top) / fig.dpi * 25.4
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / "p.png")
    gap = captured.get("mm")
    assert gap is not None, "no durations were drawn"
    assert gap >= ES.SPAN_LABEL_CLEARANCE_MM, f"only {gap:.2f}mm of clearance"


def test_the_run_time_is_dark_enough_to_read():
    """It is a number to be read, not an imprint that may recede. 4.5:1
    against white is the point below which small text stops being legible
    off a screen -- the project mark's own grey sits at 2.2:1."""
    def _luminance(colour):
        parts = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        parts = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                 for c in parts]
        return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2]

    contrast = 1.05 / (_luminance(ES.RUNTIME_COLOR) + 0.05)
    assert contrast >= 4.5, f"run time at {contrast:.1f}:1 against white"
    assert contrast > 1.05 / (_luminance(ES.BRAND_COLOR) + 0.05), \
        "the run time is no darker than the mark it sits opposite"


def test_the_axis_says_cycle_not_epoch(tmp_path, monkeypatch):
    """A point is a cycle of generation, training and evaluation, holding
    ep/cyc training epochs. Called an epoch, a run of 4 cycles at 5 epochs
    each read off the figure as 4 epochs instead of 20."""
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and fig.axes:
            captured["labels"] = [fig.axes[-1].get_xlabel()]
            captured["titles"] = [ax.get_title() for ax in fig.axes]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")
    shown = captured["labels"] + captured["titles"]
    assert any("cycle" in t for t in shown), shown
    assert not any("epoch" in t.lower() for t in shown), shown


def test_the_recorded_key_is_still_epoch(tmp_path):
    """Only the figure is renamed. The tracker's schema is not: an existing
    epoch_tracker.json has to keep loading, and its rows are keyed by it."""
    assert all("epoch" in r for r in _plot_rows())
    assert ES.plot_epoch_stats(_plot_rows(), tmp_path / "keyed.png") is not None
    renamed = [{k.replace("epoch", "cycle"): v for k, v in r.items()}
               for r in _plot_rows()]
    assert ES.plot_epoch_stats(renamed, tmp_path / "renamed.png") is None, \
        "the plotter read a row keyed by cycle, so the schema moved too"


def _figure_texts(tmp_path, monkeypatch, rows):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured, real_close = {}, plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "texts"):
            captured["texts"] = [(t.get_text(), t.get_position(),
                                  t.get_ha(), t.get_va()) for t in fig.texts]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / "p.png")
    return captured.get("texts", [])


def test_the_run_time_sits_in_the_top_left_corner(tmp_path, monkeypatch):
    """Opposite the project mark, on the same band: both identify the run
    rather than reporting a result, and neither belongs over the data."""
    rows = [dict(r, timestamp=(_T0 + timedelta(hours=9 * i)).isoformat())
            for i, r in enumerate(_plot_rows())]
    found = [t for t in _figure_texts(tmp_path, monkeypatch, rows)
             if t[0].startswith(ES.RUNTIME_LABEL)]
    assert len(found) == 1, "the run time was not drawn exactly once"
    text, (x, y), ha, va = found[0]
    assert "d." in text or "h." in text or text.endswith("m"), text
    assert x < 0.1 and y > 0.9, f"run time at {(x, y)}, not the top-left corner"
    assert ha == "left" and va == "top"


def test_no_run_time_is_drawn_when_the_rows_are_unstamped(tmp_path, monkeypatch):
    """An older tracker has no timestamps; the figure simply says nothing."""
    texts = _figure_texts(tmp_path, monkeypatch, _plot_rows())
    assert not [t for t in texts if t[0].startswith(ES.RUNTIME_LABEL)]


@pytest.mark.parametrize("pair, suffix", [
    (("accuracy_mean", "baseline_accuracy_mean"), "mean"),
    (("accuracy_max", "baseline_accuracy_max"), "best"),
])
def test_both_series_stay_visible_where_their_values_coincide(
        pair, suffix, tmp_path, monkeypatch):
    """Equal values must not render as a single marker.

    Both series are accuracies on one axis, so they can land within a marker
    width of each other -- which is what a cycle that changed nothing produces,
    the case most worth seeing. At equal size the marker drawn second covered
    the first outright and one series vanished while its legend entry stayed.
    The generated value takes the larger marker and the original is stacked on
    top of it, so a coincidence reads as a small marker centred on a large one.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    generated_key, original_key = pair
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["drawn"] = {
                l.get_label(): (l.get_markersize(), l.get_zorder(),
                                tuple(l.get_ydata()))
                for l in fig.axes[1].lines if l.get_label().endswith(suffix)}
        return real_close(fig)

    rows = _plot_rows()[:1]
    rows[0][original_key] = rows[0][generated_key]

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / "p.png")

    drawn = captured["drawn"]
    original = next(v for k, v in drawn.items() if k.startswith("original"))
    generated = next(v for k, v in drawn.items() if k.startswith("generated"))
    assert original[2] == generated[2], "the fixture no longer makes them coincide"
    assert generated[0] > original[0], "the generated marker is not the larger one"
    assert original[1] > generated[1], \
        "the smaller marker is stacked underneath, so the larger one hides it"


def test_the_legend_still_reads_original_before_generated():
    """Stacking is settled by zorder, so the listed order is free to stay."""
    assert ES.MEAN_MARKER[0][0] < ES.MEAN_MARKER[1][0]
    assert ES.BEST_MARKER[0][0] < ES.BEST_MARKER[1][0]