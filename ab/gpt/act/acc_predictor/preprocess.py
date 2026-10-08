"""Stage 1: build the supervised fine-tuning table from the nn_dataset API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from . import config




# --- data_preprocessing  ---

_DP_GROUP_KEYS = ("nn_id", "prm_id", "transform_id", "dataset")
_DP_LAYER_GROUP_KEYS = (
    "nn_id",
    "_stable_prm_key",
    "transform_id",
    "dataset",
)


def _dp_active_group_keys() -> tuple[str, ...]:
    return _DP_LAYER_GROUP_KEYS if config.USE_LAYER_STATS else _DP_GROUP_KEYS
_DP_STABLE_PRM_KEYS = (
    "batch",
    "dropout",
    "lr",
    "momentum",
    "transform",
    "model",
    "task",
    "dataset",
    "metric",
)

def _dp_stable_prm_key(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

    if not isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    stable_values = {
        key: value.get(key)
        for key in _DP_STABLE_PRM_KEYS
    }

    return json.dumps(
        stable_values,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )

_DP_REQUIRED_COLS = ("id", "nn_id", "prm_id", "transform_id", "dataset", "epoch", "accuracy")
_DP_EARLY_EPOCHS = (1, 2, 3)
_DP_MIN_EPOCHS = 50
_DP_LAYER_COLUMNS = (
    ("ww_alpha", "layer_ww_alpha"),
    ("grad_norm", "layer_grad_norm"),
    ("dead_frac", "layer_dead_frac"),
    ("taylor_imp", "layer_taylor_imp"),
    ("cka_redund", "layer_cka_redund"),
    ("rank_ratio", "layer_rank_ratio"),
    ("sensitivity", "layer_sensitivity"),
)

_DP_LAYER_REQUIRED_COLS = tuple(
    db_column
    for _, db_column in _DP_LAYER_COLUMNS
)


_DP_OUTPUT_COLUMNS = [
    "id",
    "task",
    "dataset",
    "metric",
    "nn",
    "nn_code",
    "transform_code",
    "prm",
    "accuracy_epoch_1",
    "accuracy_epoch_2",
    "accuracy_epoch_3",
    "best_accuracy",
    "epochs_to_best",
    "total_epochs",
    "layer_stats",
]



def _dp_load_nn_data() -> pd.DataFrame:
    try:
        from ab.nn.api import data as nn_data
    except ImportError as exc:
        raise ImportError(
            "Could not import ab.nn.api.data. Install the nn_dataset package."
        ) from exc

    df = pd.DataFrame(nn_data(only_best_accuracy=False,include_layer_stats=config.USE_LAYER_STATS,))
    required_columns = _DP_REQUIRED_COLS

    if config.USE_LAYER_STATS:
        required_columns += _DP_LAYER_REQUIRED_COLS
    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _dp_normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["accuracy"].max() <= 1.0:
        df["accuracy"] = df["accuracy"] * 100
    df["epoch"] = df["epoch"].astype(int)
    if "prm" not in df.columns:
        raise ValueError("Missing required column: prm")

    df["_stable_prm_key"] = df["prm"].map(_dp_stable_prm_key)
    return df


def _dp_filter_long_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    group_cols = list(_DP_GROUP_KEYS)
    if config.USE_LAYER_STATS:
        group_cols = list(_dp_active_group_keys())
    total_runs = df.groupby(group_cols, observed=True).ngroups
    run_max_epoch = df.groupby(group_cols, observed=True)["epoch"].transform("max")
    filtered = df[run_max_epoch >= _DP_MIN_EPOCHS].copy()
    valid_runs = filtered.groupby(group_cols, observed=True).ngroups
    return filtered, total_runs, valid_runs

def _dp_format_layer_stats(group_df: pd.DataFrame) -> str:
    lines = [
        "epoch\t" + "\t".join(
            prompt_column
            for prompt_column, _ in _DP_LAYER_COLUMNS
        )
    ]

    if config.USE_LAYER_STATS:
        group_df = group_df[
            group_df["epoch"].isin(_DP_EARLY_EPOCHS)
        ]

    for _, row in group_df.sort_values("epoch").iterrows():
        values = []

        for _, db_column in _DP_LAYER_COLUMNS:
            value = row.get(db_column)

            if pd.isna(value):
                values.append("NA")
            else:
                values.append(f"{float(value):.6g}")

        if all(value == "NA" for value in values):
            continue

        lines.append(
            f"{int(row['epoch'])}\t" + "\t".join(values)
        )

    return "\n".join(lines) if len(lines) > 1 else ""

def _dp_parse_prm(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    if pd.isna(value):
        return {}
    return value


def _dp_process_run(
    group_df: pd.DataFrame,
    dataset: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (record, drop_reason). drop_reason is None on success."""
    group_df = group_df.sort_values("epoch").drop_duplicates("epoch", keep="first")
    acc_by_epoch = group_df.set_index("epoch")["accuracy"]

    for epoch in _DP_EARLY_EPOCHS:
        if epoch not in acc_by_epoch.index:
            return None, f"missing_epoch_{epoch}"
        if pd.isna(acc_by_epoch.at[epoch]):
            return None, f"missing_epoch_{epoch}"

    best_accuracy = group_df["accuracy"].max()
    if pd.isna(best_accuracy):
        return None, "missing_best_accuracy"

    epochs_to_best = int(
        group_df.loc[group_df["accuracy"] == best_accuracy, "epoch"].min()
    )
    layer_stats = _dp_format_layer_stats(group_df)
    first_row = group_df.iloc[0]

    record = {
        "id": str(first_row["id"]),
        "task": first_row["task"],
        "dataset": dataset,
        "metric": first_row["metric"],
        "nn": first_row["nn"],
        "nn_code": first_row["nn_code"],
        "transform_code": first_row["transform_code"],
        "prm": _dp_parse_prm(first_row["prm"]),
        "accuracy_epoch_1": float(acc_by_epoch.at[1]),
        "accuracy_epoch_2": float(acc_by_epoch.at[2]),
        "accuracy_epoch_3": float(acc_by_epoch.at[3]),
        "best_accuracy": float(best_accuracy),
        "epochs_to_best": epochs_to_best,
        "total_epochs": int(group_df["epoch"].max()),
        "layer_stats": layer_stats,
    }
    return record, None


def _dp_build_records(df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    drop_counts: dict[str, int] = {}

    grouped = df.groupby(list(_DP_GROUP_KEYS), observed=True)
    if config.USE_LAYER_STATS:
        grouped = df.groupby(
            list(_dp_active_group_keys()),
            observed=True,
        )
    for (_, _, _, dataset), group_df in tqdm(grouped, desc="Processing runs"):
        record, drop_reason = _dp_process_run(group_df, dataset)
        if record is None:
            drop_counts[drop_reason] = drop_counts.get(drop_reason, 0) + 1
            continue
        records.append(record)

    return records, drop_counts


def _dp_serialize_prm(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, dict) else value


def _dp_save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            row = {**record, "prm": _dp_serialize_prm(record["prm"])}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dp_save_csv(df_final: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df_final.empty:
        pd.DataFrame(columns=_DP_OUTPUT_COLUMNS).to_csv(path, index=False)
        return

    df_out = df_final.copy()
    df_out["prm"] = df_out["prm"].map(_dp_serialize_prm)
    df_out.to_csv(path, index=False)


def data_preprocessing(
    output_jsonl_path: Path | str | None = None,
    output_csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Prepare data for supervised fine-tuning from nn_dataset.

    Loads per-epoch training stats, keeps runs with >= 50 epochs, extracts
    accuracy at epochs 1 and 2 as input features, and computes best_accuracy
    and epochs_to_best from the full run.

    Args:
        output_jsonl_path: Optional path to save JSONL output.
        output_csv_path: Optional path to save CSV output.

    Returns:
        DataFrame with one row per valid training run.
    """
    print("Loading data from nn_dataset API...")
    raw_df = _dp_load_nn_data()
    rows_loaded = len(raw_df)
    print(f"Loaded {rows_loaded:,} rows")

    df = _dp_normalize_dataframe(raw_df)
    df, total_runs, valid_runs = _dp_filter_long_runs(df)
    kept_pct = (100 * valid_runs / total_runs) if total_runs else 0.0
    print(
        f"Runs with >= {_DP_MIN_EPOCHS} epochs: {valid_runs:,} / {total_runs:,} "
        f"({kept_pct:.1f}% kept)"
    )

    records, drop_counts = _dp_build_records(df)
    dropped = sum(drop_counts.values())
    df_final = pd.DataFrame(records)

    print(f"Final records: {len(df_final):,} (dropped {dropped:,})")
    for reason, count in sorted(drop_counts.items()):
        print(f"  - {reason}: {count:,}")

    if output_jsonl_path:
        jsonl_path = Path(output_jsonl_path)
        _dp_save_jsonl(records, jsonl_path)
        print(f"Wrote JSONL: {jsonl_path}")

    if output_csv_path:
        csv_path = Path(output_csv_path)
        _dp_save_csv(df_final, csv_path)
        print(f"Wrote CSV: {csv_path}")

    if valid_runs:
        success_rate = 100 * len(df_final) / valid_runs
        print(f"Success rate: {success_rate:.1f}% of long runs")

    return df_final
