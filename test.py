# test.py
from __future__ import annotations

import sys
import time

import ab.nn.api as lemur
from ab.nn.api import JoinConf


def assert_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_legacy_pairwise_schema():
    """
    Legacy behavior: 2-model join, wide schema.
    This MUST stay unchanged.
    """
    conf = JoinConf(
        num_joint_nns=2,
        diff_columns=("nn",),
        task="img-classification",
        dataset="cifar-10",
        metric="acc",
        similarity_mode="none",
    )

    df = lemur.data(sql=conf, max_rows=50, include_nn_stats=False)

    required = [
        "nn", "nn_code", "accuracy", "prm_id",
        "nn_2", "nn_code_2", "accuracy_2", "prm_id_2",
    ]

    assert_columns(df, required)

    print("[PASS] legacy pairwise schema")
    print(df[["nn", "accuracy", "nn_2", "accuracy_2"]].head())


def test_sql_variable_n():
    """
    NEW functionality:
    SQL-only variable-N model selection.
    One row per model, no similarity, no joins.
    """
    N = 5
    conf = JoinConf(
        num_joint_nns=N,
        task="img-classification",
        dataset="cifar-10",
        metric="acc",
        similarity_mode="none",
    )

    df = lemur.data(sql=conf, max_rows=100, include_nn_stats=False)

    assert len(df) == N, f"Expected {N} rows, got {len(df)}"

    required = ["nn", "accuracy", "prm_id"]
    assert_columns(df, required)

    # Ensure uniqueness (no duplicate architectures)
    assert df["nn"].is_unique, "Duplicate nn entries detected"

    print("[PASS] SQL variable-N selection")
    print(df[["nn", "accuracy"]])



def test_otf_anchor_band():
    conf = JoinConf(
        num_joint_nns=5,
        task="img-classification",
        dataset="cifar-10",
        metric="acc",
        similarity_mode="anchor_band_otf",
        anchor_nn="rl-back-init-0b0d0b7728c47eb6cc9fe040ebb9d239",
        similarity_band="high",
    )

    df = lemur.data(sql=conf, max_rows=200, include_nn_stats=False)

    assert len(df) > 0
    assert "anchor_jaccard" in df.columns

    print("[PASS] OTF anchor band")
    print(df[["nn", "accuracy", "anchor_jaccard"]])

def test_legacy_performance_smoke():
    """
    Performance smoke test for legacy path.
    No strict SLA, just ensures no accidental blow-up.
    """
    conf = JoinConf(
        num_joint_nns=2,
        diff_columns=("nn",),
        task="img-classification",
        dataset="cifar-10",
        metric="acc",
        similarity_mode="none",
    )

    t0 = time.time()
    df = lemur.data(sql=conf, max_rows=20000, include_nn_stats=False)
    dt = time.time() - t0

    print(f"[INFO] legacy perf: rows={len(df)} time={dt:.2f}s")

    # Soft sanity guard 
    assert dt < 120, "Legacy query unexpectedly slow"

def test_sql_variable_n_performance_smoke():
    """
    Performance smoke test for SQL-only variable-N path.
    Ensures query remains bounded and does not regress into
    a full-table join or accidental cross-product.
    """
    N = 10

    conf = JoinConf(
        num_joint_nns=N,
        task="img-classification",
        dataset="cifar-10",
        metric="acc",
        similarity_mode="none",
    )

    t0 = time.time()
    df = lemur.data(sql=conf, max_rows=500, include_nn_stats=False)
    dt = time.time() - t0

    print(f"[INFO] SQL variable-N perf: rows={len(df)} time={dt:.2f}s")

    # Hard correctness guarantees
    assert len(df) == N, f"Expected {N} rows, got {len(df)}"
    assert df["nn"].is_unique, "Duplicate nn entries detected"

    assert dt < 30, "SQL variable-N query unexpectedly slow"
def main():
    print("LEMUR / NNGPT integration tests")

    test_legacy_pairwise_schema()
    test_sql_variable_n()
    test_legacy_performance_smoke()
    test_otf_anchor_band()
    test_sql_variable_n_performance_smoke()

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

