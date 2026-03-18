from __future__ import annotations

import os
import time
import statistics
from typing import Sequence
import unittest
import functools

import ab.nn.api as lemur
from ab.nn.api import JoinConf
from ab.gpt.util.Util import extract_all_to_train


# ---------------------- Helpers ----------------------

def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return wrapper


def assert_columns(df, required: Sequence[str], label: str = ""):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"{label} missing columns: {missing}. Have={list(df.columns)}")


def bench(fn, *, repeats: int, warmup: int) -> tuple[float, list[float]]:
    for _ in range(max(0, warmup)):
        fn()
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def print_bench(name: str, median_s: float, runs: list[float], task: str, dataset: str, metric: str):
    runs_str = ", ".join(f"{t:.2f}" for t in runs)
    print(f"[BENCH] {name}: median={median_s:.2f}s runs=[{runs_str}] (task={task}, dataset={dataset}, metric={metric})")


SIM_BANDS = {
    "high": (0.95, 1.0000001),
    "medium": (0.85, 0.95),
    "low": (0.60, 0.85),
    "very_low": (0.0, 0.60),
}


def band_range(name: str) -> tuple[float, float]:
    if name not in SIM_BANDS:
        raise ValueError(f"Invalid band '{name}'. Must be one of {list(SIM_BANDS)}")
    return SIM_BANDS[name]


# ---------------------- Config ----------------------

TASK = os.getenv("LEMUR_TASK", "img-classification")
DATASET = os.getenv("LEMUR_DATASET", "celeba-gender")
METRIC = os.getenv("LEMUR_METRIC", "acc")

REPEATS = int(os.getenv("LEMUR_BENCH_REPEATS", "3"))
WARMUP = int(os.getenv("LEMUR_BENCH_WARMUP", "1"))

LEGACY_ROWS = int(os.getenv("LEMUR_LEGACY_ROWS", "20000"))
VAR_N_ROWS = int(os.getenv("LEMUR_VARN_ROWS", "500"))
VAR_N = int(os.getenv("LEMUR_VARN", "10"))

THRESH_LEGACY = float(os.getenv("LEMUR_THRESH_LEGACY", "60"))
THRESH_VARN = float(os.getenv("LEMUR_THRESH_VARN", "30"))

# for anchor-band tests
BAND_N = int(os.getenv("LEMUR_BAND_N", "5"))
# BANDS_TO_TEST = os.getenv("LEMUR_BANDS", "high,medium,low,very_low").split(",")
EXTENDED = os.getenv("LEMUR_EXTENDED", "0") == "1"
BANDS_TO_TEST = os.getenv("LEMUR_BANDS", "high,medium,low,very_low" if EXTENDED else "high").split(",")


class Test(unittest.TestCase):
    """
    Verification:
      python test.py
      (requires nn_minhash table populated via json_nn_to_db)
    """

    # ---------------------- Tests ----------------------

    @measure_time
    def test_legacy_pairwise_schema(self):
        """
        Legacy behavior: 2-model join, wide schema.
        MUST stay unchanged.
        """
        conf = JoinConf(
            num_joint_nns=2,
            same_columns=("epoch", "metric", "dataset", "task"),
            diff_columns=("nn",),
            enhance_nn=True,
            similarity_mode="none",
        )

        df = lemur.data(
            sql=conf,
            include_nn_stats=True,
            task=TASK,
            dataset=DATASET,
            metric=METRIC,
            max_rows=200,
        )

        required = [
            "nn", "epoch", "nn_code", "accuracy", "prm_id", "prm", "transform_code",
            "nn_2", "epoch_2", "nn_code_2", "accuracy_2", "prm_id_2", "prm_2", "transform_code_2"
        ]
        assert_columns(df, required, label="legacy schema")

        print("[PASS] legacy pairwise schema")
        print(df[["nn", "accuracy", "nn_2", "accuracy_2"]].head(5))

    @measure_time
    def test_sql_variable_n_correctness(self):
        """
        SQL-only variable-N selection (tall schema). One row per model.
        """
        N = 5
        conf = JoinConf(
            num_joint_nns=N,
            similarity_mode="none",
        )

        df = lemur.data(
            sql=conf,
            include_nn_stats=False,
            task=TASK,
            dataset=DATASET,
            metric=METRIC,
            max_rows=200,
        )

        if len(df) != N:
            raise AssertionError(f"Expected {N} rows, got {len(df)}")

        required = ["nn", "accuracy", "prm_id"]
        assert_columns(df, required, label="var-N correctness")

        if not df["nn"].is_unique:
            raise AssertionError("Duplicate nn entries detected")

        print("[PASS] SQL variable-N selection correctness")
        print(df[["nn", "accuracy"]].head(10))

    @measure_time
    def test_anchor_band_correctness_all_bands(self):
        """
        Anchor-band (DB-minhash + UDF) with auto anchor:
        - should return BAND_N rows for each band (if feasible)
        - returned anchor_jaccard must fall inside the band.
        """
        print("[TEST] Running anchor_band_correctness_all_bands (DB MinHash, auto-anchor, all bands)")
        for band in BANDS_TO_TEST:
            band = band.strip()
            mn, mx = band_range(band)

            conf = JoinConf(
                num_joint_nns=BAND_N,
                similarity_mode="anchor_band_db_minhash",
                similarity_band=band,
            )

            df = lemur.data(
                sql=conf,
                include_nn_stats=False,
                task=TASK,
                dataset=DATASET,
                metric=METRIC,
                max_rows=5000,
            )

            if len(df) != BAND_N:
                raise AssertionError(f"[{band}] Expected {BAND_N} rows, got {len(df)} (maybe no feasible anchor?)")

            assert_columns(df, ["nn", "accuracy", "anchor_jaccard"], label=f"[{band}] anchor-band")

            jmin = float(df["anchor_jaccard"].min())
            jmax = float(df["anchor_jaccard"].max())

            if not (jmin >= mn - 1e-12 and jmax < mx + 1e-12):
                raise AssertionError(
                    f"[{band}] anchor_jaccard out of band. "
                    f"band=[{mn},{mx}) got min={jmin} max={jmax}"
                )

            print(f"[PASS] anchor-band '{band}' correctness: min/max j = {jmin:.6f}/{jmax:.6f}")
            print(df[["nn", "accuracy", "anchor_jaccard"]].head(5))

    @measure_time
    def test_legacy_performance_smoke(self):
        conf = JoinConf(
            num_joint_nns=2,
            diff_columns=("nn",),
            similarity_mode="none",
        )

        def run():
            lemur.data(
                sql=conf,
                include_nn_stats=False,
                task=TASK,
                dataset=DATASET,
                metric=METRIC,
                max_rows=LEGACY_ROWS,
            )

        median_s, runs = bench(run, repeats=REPEATS, warmup=WARMUP)
        print_bench("legacy_perf", median_s, runs, TASK, DATASET, METRIC)

        if median_s > THRESH_LEGACY:
            raise AssertionError(f"Legacy query unexpectedly slow: median {median_s:.2f}s > {THRESH_LEGACY:.2f}s")

    @measure_time
    def test_sql_variable_n_performance_smoke(self):
        conf = JoinConf(
            num_joint_nns=VAR_N,
            similarity_mode="none",
        )

        def run():
            df = lemur.data(
                sql=conf,
                include_nn_stats=False,
                task=TASK,
                dataset=DATASET,
                metric=METRIC,
                max_rows=VAR_N_ROWS,
            )
            if len(df) != VAR_N:
                raise AssertionError(f"Expected {VAR_N} rows, got {len(df)}")
            if not df["nn"].is_unique:
                raise AssertionError("Duplicate nn entries detected")

        median_s, runs = bench(run, repeats=REPEATS, warmup=WARMUP)
        print_bench("sql_variable_n_perf", median_s, runs, TASK, DATASET, METRIC)

        if median_s > THRESH_VARN:
            raise AssertionError(f"SQL variable-N query unexpectedly slow: median {median_s:.2f}s > {THRESH_VARN:.2f}s")

    def test_extract_generated_info_to_train_nn(self):
        self.assertTupleEqual(
            extract_all_to_train('aaa <nn>nn code</nn> <hp> hyper-parameter code </hp> <tr>tr code</tr>'),
            ('nn code', 'hyper-parameter code', 'tr code')
        )
        self.assertTupleEqual(
            extract_all_to_train('aaa <nn>nn code</nn> </hp> <tr>tr code</tr>'),
            ('nn code', None, 'tr code')
        )
        self.assertTupleEqual(
            extract_all_to_train('aaa <nn>nn code </hp> tr code</tr>'),
            (None, None, None)
        )


if __name__ == '__main__':
    print("LEMUR / NNGPT integration tests")
    print(f"[SCOPE] task={TASK} dataset={DATASET} metric={METRIC}")
    print(f"[BENCH CFG] repeats={REPEATS} warmup={WARMUP} legacy_rows={LEGACY_ROWS} varN={VAR_N} varN_rows={VAR_N_ROWS}")
    print(f"[THRESH] legacy<{THRESH_LEGACY}s varN<{THRESH_VARN}s")
    print(f"[ANCHOR-BAND] N={BAND_N} bands={BANDS_TO_TEST}")

    unittest.main(verbosity=2)