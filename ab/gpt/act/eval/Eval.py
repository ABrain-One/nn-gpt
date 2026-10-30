import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ab.nn.util.Util import release_memory, uuid4
from ab.gpt.util.Util import read_py_file_as_string
from ab.gpt.util.Const import epoch_dir, new_nn_file, nngpt_dir, synth_dir, hp_file, NN_TRAIN_EPOCHS
from ab.gpt.util.Util import verify_nn_code, copy_to_lemur
from ab.gpt.util.data.CycleResults import generate_cycle_results, collect_cycle_metrics, save_cycle_results
from ab.gpt.util.eval import nneval_worker_pool as NNEvalWorkerPool

# Default evaluation parameters used by the CLI entrypoint when no per-model
# metadata is available.
TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"

# Default hyperparameters. The final "epoch" value is always overridden by
# nn_train_epochs for the current evaluation run.
LR = 0.01
#: Evaluation batch size.
#:
#: Was 64. The evaluation worker shares the card and the host with a parked 9B
#: trainer, and one measured cycle logged eight "host MemAvailable below floor"
#: warnings and nine worker recycles -- each recycle costing a process start
#: and a dataset load. Halving the batch halves activation memory, and because
#: every candidate in a cycle is measured at the same value it cannot bias the
#: comparison between them. The OOM retry at half this value remains the
#: backstop for an architecture that still does not fit.
BATCH = 32
DROPOUT = 0.2
MOMENTUM = 0.9
TRANSFORM = "norm_256_flip"
STOCHASTIC_DEPTH_PROB = 0.0
NORM_EPS = 1e-5
NORM_STD = 0.5
TIE_WEIGHTS = 0.0
DROPOUT_AUX = 0.0
ATTENTION_DROPOUT = 0.0
NORM_MOMENTUM = 0.1
SCORE_THRESH = 0.01
NMS_THRESH = 0.45
IOU_THRESH = 0.5
DETECTIONS_PER_IMG = 0.5
TOPK_CANDIDATES = 0.5
NEG_TO_POS_RATIO = 0.5
PRETRAINED = 0.0
PATCH_SIZE = 0.125
# Optional JSON string passed from CLI to override the assembled prm payload
# after dataframe.df / hp.json have been loaded.
PRM_JSON = None

SAVE_TO_DB = True
NN_NAME_PREFIX = None
NN_ALTER_EPOCHS = None
ONLY_EPOCH = None
EPOCH_LIMIT_MINUTES = None
CUSTOM_SYNTH_DIR = None
# Cycle number is distinct from the NNAlter epoch index. Iterative pipelines
# may keep cycle fixed while still evaluating one epoch directory at a time.
CYCLE = None


def _resolve_use_all_visible_gpus(use_all_visible_gpus: Optional[bool]) -> bool:
    if use_all_visible_gpus is not None:
        return bool(use_all_visible_gpus)
    raw = os.getenv("NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS")
    if raw is None or raw == "":
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _default_eval_prm(
    *,
    lr: float,
    batch: int,
    dropout: float,
    momentum: float,
    transform: str,
    stochastic_depth_prob: float,
    norm_eps: float,
    norm_std: float,
    tie_weights: float,
    dropout_aux: float,
    attention_dropout: float,
    norm_momentum: float,
    score_thresh: float,
    nms_thresh: float,
    iou_thresh: float,
    detections_per_img: float,
    topk_candidates: float,
    neg_to_pos_ratio: float,
    pretrained: float,
    patch_size: float,
) -> Dict[str, Any]:
    # Build the baseline prm payload before per-model metadata or explicit
    # prm_json overrides are applied.
    return {
        "lr": lr,
        "batch": batch,
        "dropout": dropout,
        "momentum": momentum,
        "transform": transform,
        "stochastic_depth_prob": stochastic_depth_prob,
        "norm_eps": norm_eps,
        "norm_std": norm_std,
        "tie_weights": tie_weights,
        "dropout_aux": dropout_aux,
        "attention_dropout": attention_dropout,
        "norm_momentum": norm_momentum,
        "score_thresh": score_thresh,
        "nms_thresh": nms_thresh,
        "iou_thresh": iou_thresh,
        "detections_per_img": detections_per_img,
        "topk_candidates": topk_candidates,
        "neg_to_pos_ratio": neg_to_pos_ratio,
        "pretrained": pretrained,
        "patch_size": patch_size,
    }


def _extract_accuracy_from_eval_payload(payload: Dict[str, Any]) -> Optional[float]:
    eval_results = payload.get("eval_results")
    if not isinstance(eval_results, dict):
        return None
    accuracy = eval_results.get("accuracy", eval_results.get("acc"))
    if accuracy is None:
        epochs_data = eval_results.get("epochs", [])
        if epochs_data:
            first_epoch = epochs_data[0] or {}
            if isinstance(first_epoch, dict):
                accuracy = first_epoch.get("accuracy", first_epoch.get("acc"))
    if accuracy is None:
        return None
    try:
        return float(accuracy)
    except (TypeError, ValueError):
        return None


def _load_existing_success_result(
    model_dir_path: Path,
    *,
    require_eval_checkpoint: bool = False,
) -> Optional[Dict[str, Any]]:
    eval_info_path = model_dir_path / "eval_info.json"
    if not eval_info_path.exists():
        return None
    if require_eval_checkpoint and not (model_dir_path / "eval_checkpoint.pth").exists():
        return None
    try:
        payload = json.loads(eval_info_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    accuracy = _extract_accuracy_from_eval_payload(payload)
    if accuracy is None:
        return None
    return {
        "model_id": model_dir_path.name,
        "success": True,
        "accuracy": float(accuracy),
        "skipped": True,
        "code_file": str(model_dir_path / new_nn_file),
    }


def _validate_full_stat_artifact(model_dir_path: Path, *, save_to_db: bool) -> None:
    stat_path = model_dir_path / "1.json"
    if not stat_path.exists():
        if save_to_db:
            raise FileNotFoundError(
                f"Expected full nn-dataset stat artifact at {stat_path}; "
                "NNEval must not replace it with a slim summary."
            )
        return

    try:
        payload = json.loads(stat_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid stat artifact JSON at {stat_path}: {exc}") from exc

    rows = payload if isinstance(payload, list) else [payload]
    required = {"uid", "transform", "duration", "accuracy"}
    if not any(isinstance(row, dict) and required.issubset(row) for row in rows):
        raise ValueError(
            f"Stat artifact at {stat_path} is not a full nn-dataset trial JSON; "
            f"missing required fields {sorted(required)}."
        )


def _write_success_outputs(spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    model_dir_path = Path(spec["model_dir"])
    accuracy = float(result["accuracy"])
    _validate_full_stat_artifact(model_dir_path, save_to_db=bool(spec.get("save_to_db")))
    eval_info_data = {
        "eval_args": result.get("eval_args", {}),
        "eval_results": {
            "checksum": result.get("checksum"),
            "accuracy": accuracy,
            "full_result": result.get("full_result", ""),
        },
        "cli_args": {
            "task": spec["task"],
            "dataset": spec["dataset"],
            "metric": spec["metric"],
            "lr": spec["prm"].get("lr"),
            "batch": spec["prm"].get("batch"),
            "dropout": spec["prm"].get("dropout"),
            "momentum": spec["prm"].get("momentum"),
            "transform": spec["prm"].get("transform"),
        },
    }
    (model_dir_path / "eval_summary.json").write_text(
        json.dumps([{"epoch": int(spec["prm"].get("epoch", 1)), "accuracy": accuracy}], indent=2),
        encoding="utf-8",
    )
    (model_dir_path / "eval_info.json").write_text(
        json.dumps(eval_info_data, indent=4, default=str),
        encoding="utf-8",
    )
    error_path = model_dir_path / "error.txt"
    if error_path.exists():
        error_path.unlink()
    verification_failure_path = model_dir_path / "eval_verification_failed.txt"
    if verification_failure_path.exists():
        verification_failure_path.unlink()

    nn_name = uuid4(read_py_file_as_string(spec["code_file"]))
    lemur_prefix = spec.get("lemur_prefix")
    if lemur_prefix:
        nn_name = str(lemur_prefix) + "-" + nn_name
    copy_to_lemur(
        model_dir_path,
        nn_name,
        spec["task"],
        spec["dataset"],
        spec["metric"],
    )
    return {
        "model_id": spec["model_id"],
        "success": True,
        "accuracy": accuracy,
        "skipped": False,
        "code_file": spec["code_file"],
    }


#: Smallest batch an OOM retry will go down to.
MIN_RETRY_BATCH = 4


def _retry_oom_at_half_batch(requests, results, use_all_visible_gpus):
    """Re-run the entries that ran out of memory, at half the batch size.

    One retry, not a loop. Returns the results list with the successful
    retries substituted in; everything else is untouched.
    """
    if os.environ.get("NNGPT_OOM_RETRY", "1").strip().lower() in ("0", "false", "no"):
        return results
    retry_idx = []
    for i, (req, res) in enumerate(zip(requests, results)):
        if res.get("success") or not res.get("is_oom"):
            continue
        batch = (req.get("prm") or {}).get("batch")
        try:
            batch = int(batch)
        except (TypeError, ValueError):
            continue
        if batch // 2 < MIN_RETRY_BATCH:
            continue
        retry_idx.append(i)
    if not retry_idx:
        return results

    retries = []
    for i in retry_idx:
        req = dict(requests[i])
        prm = dict(req.get("prm") or {})
        prm["batch"] = int(prm["batch"]) // 2
        req["prm"] = prm
        retries.append(req)
    print(f"  Retrying {len(retries)} out-of-memory model(s) at half the "
          f"batch size: "
          + ", ".join(f"{r['model_id']}->batch {r['prm']['batch']}" for r in retries),
          flush=True)
    try:
        retry_results = NNEvalWorkerPool.evaluate_model_entries(
            [{"payload": r} for r in retries],
            use_all_visible_gpus=use_all_visible_gpus,
        )
    except Exception as exc:
        print(f"  OOM retry failed to run: {exc}", flush=True)
        return results

    out = list(results)
    recovered = 0
    for i, req, res in zip(retry_idx, retries, retry_results):
        if res.get("success"):
            requests[i]["prm"] = req["prm"]      # record what it actually ran at
            out[i] = res
            recovered += 1
    print(f"  OOM retry recovered {recovered} of {len(retries)}", flush=True)
    return out


def _short_error(worker_result) -> str:
    """The first line of the worker's error, enough to read at a glance."""
    text = str(worker_result.get("error") or "unknown").strip()
    first = text.splitlines()[0] if text else "unknown"
    return first[:160]


def _write_failure_outputs(spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    model_dir_path = Path(spec["model_dir"])
    error_text = str(result.get("error", "Unknown evaluation error"))
    traceback_text = str(result.get("traceback", ""))
    (model_dir_path / "error.txt").write_text(
        f"{error_text}\n\n{traceback_text}".strip() + "\n",
        encoding="utf-8",
    )
    return {
        "model_id": spec["model_id"],
        "success": False,
        "error": error_text,
        "is_oom": bool(result.get("is_oom", False)),
        "skipped": False,
    }


def _build_eval_request(
    *,
    model_id: str,
    model_dir_path: Path,
    code_file_path: Path,
    task: str,
    dataset: str,
    metric: str,
    prm: Dict[str, Any],
    save_to_db: bool,
    prefix_for_db: Optional[str],
    epoch_limit_minutes: Optional[int],
    lemur_prefix: Optional[str],
    save_eval_checkpoint: bool = False,
) -> Dict[str, Any]:
    request = {
        "model_id": str(model_id),
        "model_dir": str(model_dir_path),
        "code_file": str(code_file_path),
        "task": str(task),
        "dataset": str(dataset),
        "metric": str(metric),
        "prm": dict(prm),
        "save_to_db": bool(save_to_db),
        "prefix": prefix_for_db,
        "save_path": str(model_dir_path),
        "epoch_limit_minutes": epoch_limit_minutes,
        "lemur_prefix": lemur_prefix,
        "use_ast_validation": None,
    }
    if save_eval_checkpoint:
        request["save_eval_checkpoint"] = True
        request["checkpoint_path"] = str(model_dir_path / "eval_checkpoint.pth")
    return request


def _collect_epoch_requests(
    *,
    models_base_dir: Path,
    nn_name_prefix: Optional[str],
    nn_train_epochs: int,
    save_to_db: bool,
    task: str,
    dataset: str,
    metric: str,
    lr: float,
    batch: int,
    dropout: float,
    momentum: float,
    transform: str,
    stochastic_depth_prob: float,
    norm_eps: float,
    norm_std: float,
    tie_weights: float,
    dropout_aux: float,
    attention_dropout: float,
    norm_momentum: float,
    score_thresh: float,
    nms_thresh: float,
    iou_thresh: float,
    detections_per_img: float,
    topk_candidates: float,
    neg_to_pos_ratio: float,
    pretrained: float,
    patch_size: float,
    prm_json: Optional[Dict[str, Any]],
    epoch_limit_minutes: Optional[int],
    save_eval_checkpoint: bool = False,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requests: List[Dict[str, Any]] = []
    immediate_results: List[Dict[str, Any]] = []
    base_nngpt_path = nngpt_dir

    def _display_model_path(path: Path) -> Path:
        """
        Return a stable display path for logs without assuming the model path
        is always inside out/nngpt (mobile iterative flow uses curation_output).
        """
        try:
            return path.relative_to(base_nngpt_path)
        except ValueError:
            return path

    for model_id in sorted(os.listdir(models_base_dir)):
        model_dir_path = models_base_dir / model_id
        if not model_dir_path.is_dir():
            continue

        code_file_path = model_dir_path / new_nn_file
        df_file_path = model_dir_path / "dataframe.df"

        if not code_file_path.exists():
            print(f"Code file {new_nn_file} not found in {model_dir_path}. Skipping.")
            continue

        existing_result = _load_existing_success_result(
            model_dir_path,
            require_eval_checkpoint=save_eval_checkpoint,
        )
        if existing_result is not None:
            print(
                f"  [SKIP] {_display_model_path(model_dir_path)} already evaluated "
                f"({existing_result['accuracy'] * 100:.2f}%)"
            )
            immediate_results.append(existing_result)
            continue

        print(f"\n--- Evaluating Model: {_display_model_path(model_dir_path)} ---")
        if not verify_nn_code(model_dir_path, code_file_path):
            print(f"Code verification failed for {code_file_path}. Skipping evaluation.")
            (model_dir_path / "eval_verification_failed.txt").write_text(
                "Initial code verification failed.\n",
                encoding="utf-8",
            )
            immediate_results.append(
                {
                    "model_id": model_id,
                    "success": False,
                    "error": "Initial code verification failed.",
                    "is_oom": False,
                    "skipped": False,
                }
            )
            continue

        # Start from command-line/default values, then let per-model files
        # override them if available.
        resolved_task = task
        resolved_dataset = dataset
        resolved_metric = metric
        # NNGPT_EVAL_PRM_MODE: "seed" (default) lets the seed's DB prm override the
        # generated hp.txt; "generated" keeps hp.txt and only fills missing keys from
        # the seed; "fixed" does the same as "generated" (the model's own keys, e.g.
        # hidden_units, stay available) and then forces the training protocol:
        # lr, batch, momentum from the Eval defaults and NNGPT_FIXED_EVAL_TRANSFORM.
        prm_mode = os.environ.get("NNGPT_EVAL_PRM_MODE", "seed").strip().lower()
        fixed_transform = os.environ.get("NNGPT_FIXED_EVAL_TRANSFORM") or transform
        prm = None
        hp_path = model_dir_path / hp_file
        if hp_path.exists():
            try:
                prm = json.loads(hp_path.read_text(encoding="utf-8"))
                print(f"Training model {model_id} with LLM recommended prm {prm}")
            except Exception as exc:
                print(f"Error loading LLM recommended training params from {hp_path}: {exc}.")
        if not prm:
            prm = _default_eval_prm(
                lr=lr,
                batch=batch,
                dropout=dropout,
                momentum=momentum,
                transform=fixed_transform if prm_mode == "fixed" else transform,
                stochastic_depth_prob=stochastic_depth_prob,
                norm_eps=norm_eps,
                norm_std=norm_std,
                tie_weights=tie_weights,
                dropout_aux=dropout_aux,
                attention_dropout=attention_dropout,
                norm_momentum=norm_momentum,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                iou_thresh=iou_thresh,
                detections_per_img=detections_per_img,
                topk_candidates=topk_candidates,
                neg_to_pos_ratio=neg_to_pos_ratio,
                pretrained=pretrained,
                patch_size=patch_size,
            )
            print(f"Training model {model_id} with command-line/default training params {prm}")

        prefix_for_db = nn_name_prefix
        orig_pref = None
        if df_file_path.exists():
            try:
                origdf = pd.read_pickle(df_file_path)
                resolved_task = origdf.get("task", resolved_task)
                resolved_dataset = origdf.get("dataset", resolved_dataset)
                resolved_metric = origdf.get("metric", resolved_metric)
                orig_pref = origdf["nn"].split("-")[0]
                original_prm_from_df = origdf.get("prm")
                if isinstance(original_prm_from_df, dict):
                    if prm_mode in ("generated", "fixed"):
                        for _k, _v in original_prm_from_df.items():
                            prm.setdefault(_k, _v)
                    else:
                        prm.update(original_prm_from_df)
                prefix_for_db = nn_name_prefix or (
                    origdf.get("nn", "unknown").split("-")[0]
                    if "nn" in origdf
                    else prefix_for_db
                )
                print(
                    f"  Loaded metadata from dataframe.df: "
                    f"task={resolved_task}, dataset={resolved_dataset}, metric={resolved_metric}"
                )
            except Exception as exc:
                print(
                    f"  Error loading dataframe.df from {df_file_path}: {exc}. "
                    "Using command-line/default parameters."
                )
        else:
            print("  No dataframe.df found. Using command-line/default evaluation parameters.")

        if prm_json:
            prm.update(prm_json)
            print(f"  Applied --prm_json overrides: {prm_json}")

        if prm_mode == "fixed":
            prm.update({"lr": lr, "batch": batch, "momentum": momentum, "transform": fixed_transform})
            prm.setdefault("dropout", dropout)

        # Whatever prm came from, the trainer needs a complete protocol. A
        # seed row carries whatever its own generator happened to record, and
        # with a diverse seed set that is not the same set of keys every time:
        # measured, six of thirty seeds in one cycle had no "batch" at all and
        # every one of them died on KeyError 'batch' after the worker had been
        # started and the dataset loaded. The defaults fill only what is
        # missing, so a value the seed or the model did state still wins.
        for _k, _v in (("batch", batch), ("lr", lr), ("momentum", momentum),
                       ("dropout", dropout)):
            if prm.get(_k) is None:
                prm[_k] = _v
                print(f"  prm['{_k}'] was not set by the seed or the model; "
                      f"using the run default {_v}")

        # Measurements are not hyperparameters. LEMUR rows carry what a past
        # run observed alongside what it set, and those leak through prm into
        # the trainer: metric_acc, duration_seconds and epoch_max=50 are
        # records of an earlier training, not instructions for this one.
        for _k in ("metric_acc", "accuracy", "duration", "duration_seconds",
                   "epoch_max", "uid", "nn", "task", "dataset", "metric"):
            prm.pop(_k, None)

        # Always evaluate with nn_train_epochs for this run, regardless of what
        # may have been stored in the original prm payload.
        prm["epoch"] = int(nn_train_epochs)
        # NNDataset expects a real transform name, never None.
        if prm.get("transform") is None or not isinstance(prm.get("transform"), str):
            prm["transform"] = transform if transform else TRANSFORM

        print(f"  Final parameters for Eval: {prm}")
        print(
            f"  Task: {resolved_task}, Dataset: {resolved_dataset}, "
            f"Metric: {resolved_metric}, Prefix: {prefix_for_db}"
        )

        requests.append(
            _build_eval_request(
                model_id=model_id,
                model_dir_path=model_dir_path,
                code_file_path=code_file_path,
                task=resolved_task,
                dataset=resolved_dataset,
                metric=resolved_metric,
                prm=prm,
                save_to_db=save_to_db,
                prefix_for_db=prefix_for_db,
                epoch_limit_minutes=epoch_limit_minutes,
                lemur_prefix=nn_name_prefix or orig_pref,
                save_eval_checkpoint=save_eval_checkpoint,
            )
        )

    return requests, immediate_results


def _predict_taylor(spec):
    accuracies, table, run_prm = [], ["epoch\ttaylor_imp"], None
    for epoch in (1, 2, 3):
        path = Path(spec["model_dir"]) / f"{epoch}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows = rows if isinstance(rows, list) else [rows]
        if len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError(f"Expected one trial record in {path}")
        row = rows[0]
        epoch_prm = {
            key: value for key, value in row.items()
            if key not in {"uid", "accuracy", "duration", "duration_seconds", "train_stat", "layer_stat"}
            and not key.startswith("metric_")
        }
        if run_prm is not None and epoch_prm != run_prm:
            raise ValueError(f"Training configuration changed between epochs in {path}")
        run_prm = epoch_prm
        accuracy = float(row["accuracy"])
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"Expected measured accuracy in [0,1] in {path}")
        accuracies.append(100 * accuracy)
        values = [float(layer["taylor_imp"]) for layer in row["layer_stat"]["layers"]
                  if layer.get("taylor_imp") is not None and math.isfinite(float(layer["taylor_imp"]))]
        if not values:
            raise ValueError(f"No finite Taylor values in {path}")
        table.append(f"{epoch}\t{sum(values) / len(values):.6g}")

    from ab.gpt.act.acc_predictor import config, infer
    settings = dict(USE_LAYER_STATS=True, ACTIVE_LAYER_FEATURES=("taylor_imp",),
                    USE_THIRD_EARLY_EPOCH=True)
    previous = {key: getattr(config, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(config, key, value)
        accuracy, epoch = infer.predict_best_accuracy(
            task=spec["task"], dataset=spec["dataset"], metric=spec["metric"],
            nn_code=read_py_file_as_string(spec["code_file"]), prm=spec["prm"],
            epoch_1_accuracy=accuracies[0], epoch_2_accuracy=accuracies[1],
            epoch_3_accuracy=accuracies[2], layer_stats="\n".join(table),
        )
    finally:
        for key, value in previous.items():
            setattr(config, key, value)
    return {"status": "complete", "predicted_best_accuracy": accuracy,
            "predicted_best_epoch": epoch, "accuracy_unit": "percent",
            "observed_epochs": 3, "prediction_horizon": infer.PREDICTOR_DEFAULT_MAX_EPOCHS}


def main(
    nn_name_prefix=NN_NAME_PREFIX,
    nn_train_epochs=NN_TRAIN_EPOCHS,
    only_epoch=ONLY_EPOCH,
    save_to_db=SAVE_TO_DB,
    nn_alter_epochs=NN_ALTER_EPOCHS,
    task=TASK,
    dataset=DATASET,
    metric=METRIC,
    lr=LR,
    batch=BATCH,
    dropout=DROPOUT,
    momentum=MOMENTUM,
    transform=TRANSFORM,
    stochastic_depth_prob=STOCHASTIC_DEPTH_PROB,
    norm_eps=NORM_EPS,
    norm_std=NORM_STD,
    tie_weights=TIE_WEIGHTS,
    dropout_aux=DROPOUT_AUX,
    attention_dropout=ATTENTION_DROPOUT,
    norm_momentum=NORM_MOMENTUM,
    score_thresh=SCORE_THRESH,
    nms_thresh=NMS_THRESH,
    iou_thresh=IOU_THRESH,
    detections_per_img=DETECTIONS_PER_IMG,
    topk_candidates=TOPK_CANDIDATES,
    neg_to_pos_ratio=NEG_TO_POS_RATIO,
    pretrained=PRETRAINED,
    patch_size=PATCH_SIZE,
    prm_json=PRM_JSON,
    epoch_limit_minutes=EPOCH_LIMIT_MINUTES,
    custom_synth_dir=CUSTOM_SYNTH_DIR,
    cycle=CYCLE,
    use_all_visible_gpus: Optional[bool] = None,
    save_eval_checkpoint: bool = False,
    culling_nn: bool = False,
):
    if culling_nn:
        if not save_to_db or save_eval_checkpoint:
            raise ValueError("Culling requires save_to_db=True and no eval checkpoint")
        nn_train_epochs = 3
    base_nngpt_path = nngpt_dir
    # NNGPT_EVAL_SAVE_TO_DB overrides the caller. A cycle screens every
    # candidate with one training run of one epoch, and those numbers used to
    # be written to LEMUR, where the next rebuild turned them into training
    # pairs -- so the corpus was built on single measurements whose own noise
    # (0.9562 vs 0.9621 for the same model) is larger than the gains it
    # selects on. With the stage-2 ladder on, screening runs set this to 0 and
    # the re-measured pass sets it to 1, so only re-measured results enter the
    # database. Models are still written to the submission directory either
    # way.
    _save_env = os.environ.get("NNGPT_EVAL_SAVE_TO_DB", "").strip().lower()
    if _save_env:
        save_to_db = _save_env in ("1", "true", "yes", "on")
        print(f"[eval] save_to_db={save_to_db} (NNGPT_EVAL_SAVE_TO_DB)")
    custom_synth_path = Path(custom_synth_dir) if custom_synth_dir else None
    if nn_alter_epochs is None:
        if custom_synth_path is not None:
            nn_alter_epochs = 1
        elif epoch_dir().is_dir():
            nn_alter_epochs = len(os.listdir(epoch_dir()))
        else:
            print(f"Directory {epoch_dir()} doesn't exist", file=sys.stderr)
            nn_alter_epochs = 0

    run_summary = {"epochs": []}
    resolved_use_all_visible_gpus = _resolve_use_all_visible_gpus(use_all_visible_gpus)

    try:
        if nn_alter_epochs:
            epoch_indices = [only_epoch] if only_epoch is not None else list(range(nn_alter_epochs))
            for i in epoch_indices:
                # Track cycle number and epoch number separately. Iterative
                # pipelines may reuse a cycle label across different epoch dirs.
                current_cycle = cycle if cycle is not None else i
                current_epoch = i
                cycle_start_time = time.time()
                # Path to one NNAlter epoch output, e.g. out/nngpt/llm/epoch/A0.
                # If a caller passes a synth dir directly, do not infer anything
                # from the default epoch root.
                if custom_synth_path is not None:
                    models_base_dir = custom_synth_path
                    current_alter_epoch_path = custom_synth_path.parent
                else:
                    current_alter_epoch_path = epoch_dir(i)
                    models_base_dir = synth_dir(current_alter_epoch_path)

                if not models_base_dir.exists():
                    print(f"Directory {models_base_dir} for NNAlter epoch {i} not found. Skipping.")
                    continue

                print(f"\n--- Scanning NNAlter Epoch Directory: {current_alter_epoch_path} ---")
                print(f"--- Synthesized Models Directory: {models_base_dir} ---")

                # Build eval requests first so already successful models can be
                # skipped without waking workers unnecessarily.
                if culling_nn:
                    for folder in models_base_dir.iterdir():
                        if (folder / new_nn_file).is_file() and (
                            any(folder.glob("[0-9]*.json")) or any(
                                (folder / name).exists() for name in
                                ("eval_info.json", "eval_summary.json", "error.txt", "prediction.json")
                            )
                        ):
                            raise FileExistsError(f"Culling requires fresh output files: {folder}")
                requests, epoch_results = _collect_epoch_requests(
                    models_base_dir=models_base_dir,
                    nn_name_prefix=nn_name_prefix,
                    nn_train_epochs=nn_train_epochs,
                    save_to_db=save_to_db,
                    task=task,
                    dataset=dataset,
                    metric=metric,
                    lr=lr,
                    batch=batch,
                    dropout=dropout,
                    momentum=momentum,
                    transform=transform,
                    stochastic_depth_prob=stochastic_depth_prob,
                    norm_eps=norm_eps,
                    norm_std=norm_std,
                    tie_weights=tie_weights,
                    dropout_aux=dropout_aux,
                    attention_dropout=attention_dropout,
                    norm_momentum=norm_momentum,
                    score_thresh=score_thresh,
                    nms_thresh=nms_thresh,
                    iou_thresh=iou_thresh,
                    detections_per_img=detections_per_img,
                    topk_candidates=topk_candidates,
                    neg_to_pos_ratio=neg_to_pos_ratio,
                    pretrained=pretrained,
                    patch_size=patch_size,
                    prm_json=prm_json,
                    epoch_limit_minutes=epoch_limit_minutes,
                    save_eval_checkpoint=save_eval_checkpoint,
                )

                if culling_nn:
                    for request in requests:
                        if request["task"] != "img-classification" or request["metric"] != "acc":
                            raise ValueError("Taylor prediction currently supports img-classification/acc")
                        request["layer_analysis"] = True

                if requests:
                    NNEvalWorkerPool.prewarm_nneval_workers(
                        use_all_visible_gpus=resolved_use_all_visible_gpus,
                        timeout_seconds=60.0,
                    )
                    entries = [{"payload": request} for request in requests]
                    worker_results = NNEvalWorkerPool.evaluate_model_entries(
                        entries,
                        use_all_visible_gpus=resolved_use_all_visible_gpus,
                    )
                    # An OOM is a property of the batch size, not of the
                    # model: the card is shared with a parked 9B trainer, and a
                    # candidate that does not fit at batch 32 usually fits at
                    # 16. There was no retry, so each one cost the cycle a
                    # paired observation it could have had.
                    worker_results = _retry_oom_at_half_batch(
                        requests, worker_results, resolved_use_all_visible_gpus)
                    trained, refused = [], []
                    for request, worker_result in zip(requests, worker_results):
                        model_id = request["model_id"]
                        if worker_result.get("success"):
                            print(f"  Evaluation results for {model_id}: {worker_result}")
                            trained.append((model_id, worker_result.get("accuracy")))
                            epoch_results.append(_write_success_outputs(request, worker_result))
                        else:
                            print(f"  Error evaluating model {model_id}: {worker_result.get('error')}")
                            refused.append((model_id, _short_error(worker_result)))
                            epoch_results.append(_write_failure_outputs(request, worker_result))
                        release_memory()

                    # The verdict of the only build that matters, printed before
                    # the fine-tuner is handed the GPU, so one cycle's log says
                    # whether every candidate was judged and how.
                    print(f"[eval] epoch {current_epoch}: {len(trained)} trained, "
                          f"{len(refused)} refused by the worker "
                          f"(of {len(requests)} sent)", flush=True)
                    for model_id, acc in sorted(trained, key=lambda t: -(t[1] or 0)):
                        print(f"[eval]   trained {model_id}: "
                              f"{'n/a' if acc is None else f'{float(acc):.4f}'}",
                              flush=True)
                    for model_id, why in refused:
                        print(f"[eval]   refused {model_id}: {why}", flush=True)

                    if culling_nn:
                        NNEvalWorkerPool.shutdown_nneval_workers()
                        from ab.gpt.act.acc_predictor import model as predictor_model
                        cached = predictor_model._predictor_model, predictor_model._predictor_tokenizer
                        try:
                            for request, result in zip(requests, worker_results):
                                if not result.get("success"):
                                    continue
                                try:
                                    prediction = _predict_taylor(request)
                                except Exception as exc:
                                    prediction = {"status": "failed", "error": str(exc)}
                                path = Path(request["model_dir"]) / "prediction.json"
                                with path.open("x", encoding="utf-8") as stream:
                                    json.dump(prediction, stream, indent=2, allow_nan=False)
                                print(f"Prediction for {request['model_id']}: {prediction}")
                        finally:
                            predictor_model._predictor_model, predictor_model._predictor_tokenizer = cached
                            release_memory()

                # Rebuild cycle_results.json from the artifacts produced in this
                # epoch directory after all worker results have been written.
                cycle_end_time = time.time()
                cycle_time_minutes = (cycle_end_time - cycle_start_time) / 60.0
                eval_results_list, model_dirs_list, successful_models, failed_models = collect_cycle_metrics(
                    models_base_dir,
                    current_alter_epoch_path,
                )
                cycle_results = generate_cycle_results(
                    cycle=current_cycle,
                    models_base_dir=models_base_dir,
                    eval_results_list=eval_results_list,
                    model_dirs_list=model_dirs_list,
                    successful_models=successful_models,
                    failed_models=failed_models,
                    cycle_time_minutes=cycle_time_minutes,
                    current_alter_epoch_path=current_alter_epoch_path,
                )
                # The per-epoch file is the canonical record. Previously only a
                # single cycle_results.json was kept, and the rolling backup was
                # named from `i-1` -- so the number on the file did not match the
                # data inside it and it was impossible to tell which epoch a
                # result belonged to. Name it after the epoch it describes.
                epoch_results_path = base_nngpt_path / f"cycle_results_{current_epoch}.json"
                save_cycle_results(cycle_results, epoch_results_path)
                cycle_results_path = base_nngpt_path / "cycle_results.json"
                save_cycle_results(cycle_results, cycle_results_path)
                print(f"Per-epoch results saved -> {epoch_results_path}")

                try:
                    from ab.gpt.util.data.EpochStats import update_epoch_report
                    # Tune calls this again once evaluation has finished, and
                    # the two wrote the same record twice -- which is how one
                    # run recorded epoch 0 with n_above_ceiling 1 in the
                    # tracker and 0 in cycle_results_0.json. The later call
                    # sees every result, so this one defers to it.
                    if os.environ.get("NNGPT_EPOCH_REPORT_FROM_EVAL", "").strip() in ("1", "true", "yes"):
                        update_epoch_report(models_base_dir, current_epoch, base_nngpt_path)
                except Exception as exc:
                    print(f"[EPOCH STATS] Could not update the epoch report: {exc}")

                print(
                    f"\n--- Cycle {current_cycle} (Epoch {current_epoch}) results saved to: "
                    f"{cycle_results_path} ---"
                )
                print(f"  Found {len(eval_results_list)} successful evaluations from eval_info.json files")
                print(f"  Found {len(failed_models)} failed models")
                run_summary["epochs"].append(
                    {
                        "epoch": current_epoch,
                        "cycle": current_cycle,
                        "models_base_dir": str(models_base_dir),
                        "cycle_results_path": str(cycle_results_path),
                        "epoch_results_path": str(epoch_results_path),
                        "model_results": sorted(epoch_results, key=lambda item: str(item.get("model_id", ""))),
                        "successful_models": len(successful_models),
                        "failed_models": len(failed_models),
                    }
                )
    finally:
        NNEvalWorkerPool.shutdown_nneval_workers()

    return run_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Neural Networks generated by NNAlter.py.")
    parser.add_argument(
        "-ae",
        "--nn_alter_epochs",
        type=int,
        default=NN_ALTER_EPOCHS,
        help="Number of epochs NNAlter.py was run for.",
    )
    parser.add_argument(
        "-oe",
        "--only_epoch",
        type=int,
        default=ONLY_EPOCH,
        help="Run NNAlter.py for the specified epoch only.",
    )
    parser.add_argument(
        "-te",
        "--nn_train_epochs",
        type=int,
        default=NN_TRAIN_EPOCHS,
        help=f"Number of epochs to train each altered NN during evaluation (default: {NN_TRAIN_EPOCHS}).",
    )
    # Configurable evaluation parameters.
    parser.add_argument("--task", type=str, default=TASK, help=f"Default task for NNEval (default: {TASK}).")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help=f"Default dataset for NNEval (default: {DATASET}).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=METRIC,
        help=f"Default metric for NNEval (default: {METRIC}).",
    )
    # Configurable hyperparameters stored inside the prm payload.
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate for NNEval if not in dataframe.df's prm (default: {LR}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH,
        help=f"Batch size for NNEval if not in dataframe.df's prm (default: {BATCH}).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT,
        help=f"Dropout rate for NNEval if not in dataframe.df's prm (default: {DROPOUT}).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=MOMENTUM,
        help=f"Momentum for NNEval if not in dataframe.df's prm (default: {MOMENTUM}).",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default=TRANSFORM,
        help=f"Default transform for NNEval if not in dataframe.df's prm (default: {TRANSFORM}).",
    )
    parser.add_argument(
        "--stochastic_depth_prob",
        type=float,
        default=STOCHASTIC_DEPTH_PROB,
        help=f"Stochastic depth probability (default: {STOCHASTIC_DEPTH_PROB}).",
    )
    parser.add_argument(
        "--norm_eps",
        type=float,
        default=NORM_EPS,
        help=f"Normalization epsilon (default: {NORM_EPS}).",
    )
    parser.add_argument(
        "--norm_std",
        type=float,
        default=NORM_STD,
        help=f"Normalization std (default: {NORM_STD}).",
    )
    parser.add_argument(
        "--tie_weights",
        type=float,
        default=TIE_WEIGHTS,
        help=f"Tie weights flag as float (default: {TIE_WEIGHTS}).",
    )
    parser.add_argument(
        "--dropout_aux",
        type=float,
        default=DROPOUT_AUX,
        help=f"Auxiliary dropout rate (default: {DROPOUT_AUX}).",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=ATTENTION_DROPOUT,
        help=f"Attention dropout rate (default: {ATTENTION_DROPOUT}).",
    )
    parser.add_argument(
        "--norm_momentum",
        type=float,
        default=NORM_MOMENTUM,
        help=f"Normalization momentum (default: {NORM_MOMENTUM}).",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=SCORE_THRESH,
        help=f"Score threshold for detection tasks (default: {SCORE_THRESH}).",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=NMS_THRESH,
        help=f"NMS threshold for detection tasks (default: {NMS_THRESH}).",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=IOU_THRESH,
        help=f"IoU threshold for detection tasks (default: {IOU_THRESH}).",
    )
    parser.add_argument(
        "--detections_per_img",
        type=float,
        default=DETECTIONS_PER_IMG,
        help=f"Detections per image value (default: {DETECTIONS_PER_IMG}).",
    )
    parser.add_argument(
        "--topk_candidates",
        type=float,
        default=TOPK_CANDIDATES,
        help=f"Top-k candidates value (default: {TOPK_CANDIDATES}).",
    )
    parser.add_argument(
        "--neg_to_pos_ratio",
        type=float,
        default=NEG_TO_POS_RATIO,
        help=f"Negative-to-positive ratio value (default: {NEG_TO_POS_RATIO}).",
    )
    parser.add_argument(
        "--pretrained",
        type=float,
        default=PRETRAINED,
        help=f"Pretrained flag as float (default: {PRETRAINED}).",
    )
    parser.add_argument(
        "--patch_size",
        type=float,
        default=PATCH_SIZE,
        help=f"Patch size value (default: {PATCH_SIZE}).",
    )
    parser.add_argument(
        "--prm_json",
        type=str,
        default=PRM_JSON,
        help='JSON string of hyperparameter overrides, e.g. \'{"lr": 0.017, "batch": 32}\'.',
    )
    # Other NNEval options.
    parser.add_argument(
        "--save_to_db",
        action=argparse.BooleanOptionalAction,
        default=SAVE_TO_DB,
        help="Whether to save evaluation results to the database.",
    )
    parser.add_argument(
        "--nn_name_prefix",
        type=str,
        default=NN_NAME_PREFIX,
        help=f"Default neural network name prefix (default: {NN_NAME_PREFIX}).",
    )
    parser.add_argument(
        "--custom_synth_dir",
        dest="custom_synth_dir",
        type=str,
        default=CUSTOM_SYNTH_DIR,
        help="Custom directory containing generated models",
    )
    parser.add_argument(
        "--epoch_limit_minutes",
        type=int,
        default=EPOCH_LIMIT_MINUTES,
        help="Max minutes allowed per epoch (default: specified in NN Dataset).",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=CYCLE,
        help="Cycle number (finetuning iteration, separate from epoch).",
    )

    parser.add_argument("--culling_nn", action="store_true",
                        help="Train three epochs with Taylor statistics and save predicted best accuracy.")
    args = parser.parse_args()
    prm_json_dict = None
    if args.prm_json:
        try:
            prm_json_dict = json.loads(args.prm_json)
        except json.JSONDecodeError as exc:
            print(f"Error parsing --prm_json: {exc}", file=sys.stderr)
            sys.exit(1)
    print("Starting evaluation of altered NNs...")
    print(f"NNAlter run epochs to scan: {args.nn_alter_epochs}")
    print(f"Each altered NN will be trained for: {3 if args.culling_nn else args.nn_train_epochs} epochs for evaluation.")
    print(f"Base task: {args.task}, Base dataset: {args.dataset}, Base metric: {args.metric}")
    print("Base Hyperparameters for NNEval (before df override):")
    print(
        f"  LR: {args.lr}, Batch Size: {args.batch_size}, Dropout: {args.dropout}, "
        f"Momentum: {args.momentum}, Transform: {args.transform}"
    )
    print(f"Save to DB: {args.save_to_db}")
    print(f"Prefix for the names of generated neural network: {args.nn_name_prefix}")

    main(
        nn_name_prefix=args.nn_name_prefix,
        nn_train_epochs=args.nn_train_epochs,
        only_epoch=args.only_epoch,
        save_to_db=args.save_to_db,
        nn_alter_epochs=args.nn_alter_epochs,
        task=args.task,
        dataset=args.dataset,
        metric=args.metric,
        lr=args.lr,
        batch=args.batch_size,
        dropout=args.dropout,
        momentum=args.momentum,
        transform=args.transform,
        stochastic_depth_prob=args.stochastic_depth_prob,
        norm_eps=args.norm_eps,
        norm_std=args.norm_std,
        tie_weights=args.tie_weights,
        dropout_aux=args.dropout_aux,
        attention_dropout=args.attention_dropout,
        norm_momentum=args.norm_momentum,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        iou_thresh=args.iou_thresh,
        detections_per_img=args.detections_per_img,
        topk_candidates=args.topk_candidates,
        neg_to_pos_ratio=args.neg_to_pos_ratio,
        pretrained=args.pretrained,
        patch_size=args.patch_size,
        prm_json=prm_json_dict,
        epoch_limit_minutes=args.epoch_limit_minutes,
        custom_synth_dir=args.custom_synth_dir,
        cycle=args.cycle,
        culling_nn=args.culling_nn,
    )
