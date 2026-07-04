import importlib.util
from pathlib import Path
from typing import Any, Dict
import torch


def _get_dataset_specs(dataset_name: str) -> tuple[tuple[int, int, int], int]:
    """Return (in_shape, num_classes) for a given dataset name."""
    name = str(dataset_name).lower().strip()
    if "mnist" in name:
        return (1, 28, 28), 10
    elif "cifar-100" in name:
        return (3, 32, 32), 100
    elif "cifar-10" in name or "cifar10" in name:
        return (3, 32, 32), 10
    elif "imagenette" in name:
        return (3, 224, 224), 10
    elif "imagenet" in name:
        return (3, 224, 224), 1000
    else:
        # Default fallback
        return (3, 224, 224), 10


def _get_param_count(code_file_path: Path, prm: dict, in_shape: tuple, num_classes: int, device: str = "cpu") -> tuple[int, int, int]:
    """Load the Net class from code_file_path, instantiate it.

    Returns:
        (total_params, trainable_params, backbone_params)
        backbone_params is the sum of parameters inside net_inst.backbones (0 if absent).
    """
    try:
        spec_module = importlib.util.spec_from_file_location("temp_net", str(code_file_path))
        temp_module = importlib.util.module_from_spec(spec_module)
        spec_module.loader.exec_module(temp_module)

        net_inst = temp_module.Net(
            in_shape=in_shape,
            out_shape=(num_classes,),
            prm=prm,
            device=torch.device(device)
        )
        total_params = sum(p.numel() for p in net_inst.parameters())
        trainable_params = sum(p.numel() for p in net_inst.parameters() if p.requires_grad)
        backbone_params = (
            sum(p.numel() for bb in net_inst.backbones for p in bb.parameters())
            if hasattr(net_inst, "backbones") else 0
        )
        del net_inst
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_params, trainable_params, backbone_params
    except Exception as e:
        print(f"Failed to calculate parameter count for {code_file_path}: {e}")
        return 0, 0, 0


# ── forward-pattern labels ────────────────────────────────────────────────────
_FUSION_LABELS: dict[str, str] = {
    "Parallel_Triple":                "parallel concat (all streams see raw input)",
    "Parallel_Residual_Sum":          "parallel residual sum",
    "Fractal_Then_Sequential_Backbones": "fractal → sequential backbones",
    "Sequential_Fractal_to_Backbones": "sequential fractal → backbones",
    "Ensemble_Backbones_to_Fractal":  "backbone ensemble → fractal",
    "Split_A_Parallel_BF":            "split-A parallel BF",
    "Split_Fractal_Parallel_AB":      "split-fractal parallel AB",
}

_GAP_LABEL: list[tuple[float, str]] = [
    (5.0,  "very small"),
    (10.0, "small"),
    (20.0, "moderate"),
    (35.0, "large"),
    (1e9,  "very large"),
]


def _classify_gap(gap_pp: float) -> str:
    for threshold, label in _GAP_LABEL:
        if gap_pp < threshold:
            return label
    return "very large"


def _generate_nl_summary(description: dict) -> str:
    """Generate a factual natural-language summary from model_description fields.

    Covers:
    - Architecture (fractal units, columns, backbone names, fusion style)
    - Parameter budget and backbone fraction
    - Per-dataset test accuracy with overfitting-gap classification
    """
    struct       = description.get("structure", {})
    n            = struct.get("fractal_N", "?")
    cols         = struct.get("fractal_cols", "?")
    backbones    = struct.get("backbones", [])
    total_p      = struct.get("total_parameters", 0)
    bb_frac      = struct.get("backbone_param_fraction")   # may be None
    fusion       = struct.get("fusion_pattern", "")
    ds_results   = description.get("dataset_results", {})
    hyperparams  = description.get("hyperparameters", {})

    # ── Part 1: architecture ────────────────────────────────────────────────
    bb_count   = len(backbones)
    bb_str     = " + ".join(backbones) if backbones else "no backbone"
    bb_noun    = f"backbone{'s' if bb_count != 1 else ''}"
    fusion_str = _FUSION_LABELS.get(fusion, fusion.replace("_", " ").lower())

    arch_sent = (
        f"A {n}-unit fractal CNN ({cols} columns) fused with "
        f"{bb_count} {bb_noun} ({bb_str}) via {fusion_str}."
    )

    # ── Part 2: parameter budget ────────────────────────────────────────────
    params_m   = total_p / 1e6 if total_p else 0.0
    if bb_frac is not None:
        param_sent = (
            f"The model has {params_m:.1f}M parameters, "
            f"{bb_frac * 100:.0f}% from the {bb_noun}."
        )
    else:
        param_sent = f"The model has {params_m:.1f}M parameters."

    # ── Part 3: per-dataset performance ────────────────────────────────────
    ds_sents: list[str] = []
    for ds_name, res in ds_results.items():
        if res.get("success") is False:
            ds_sents.append(f"On {ds_name} evaluation failed.")
            continue

        acc       = res.get("accuracy")
        train_acc = res.get("train_accuracy")
        if acc is None:
            continue

        acc_pct = acc * 100

        gap_clause = ""
        observation = ""
        if train_acc is not None:
            gap_pp = (train_acc - acc) * 100

            if gap_pp < 0:
                # test_acc > train_acc: model scored higher on test than train.
                # This is typically training instability or too-few training steps.
                gap_clause = f" (test accuracy exceeded train by {abs(gap_pp):.1f}pp)"
                observation = ", likely due to insufficient training steps or high train-set variance"
            else:
                gap_label  = _classify_gap(gap_pp)
                gap_clause = f" with a {gap_label} overfitting gap ({gap_pp:.1f}pp)"

                # Dataset-specific observations (only meaningful for non-negative gaps)
                ds_lower = ds_name.lower()
                if "imagenette" in ds_lower or "imagenet" in ds_lower:
                    if gap_pp < 10:
                        observation = ", suggesting the pretrained backbone provides strong regularization"
                    elif gap_pp > 20:
                        observation = ", indicating the model overfits on this large-resolution dataset"
                elif "cifar" in ds_lower:
                    if gap_pp > 15:
                        observation = ", consistent with resolution domain-shift from the ImageNet-pretrained backbone"
                    elif gap_pp < 5:
                        observation = ", suggesting good generalisation despite the small image size"
                elif "mnist" in ds_lower:
                    if gap_pp < 5:
                        observation = ", suggesting the model is well-regularised for this simple task"

        ds_sents.append(
            f"On {ds_name} it achieved {acc_pct:.1f}% test accuracy{gap_clause}{observation}."
        )

    return " ".join([arch_sent, param_sent] + ds_sents)


def _write_markdown_description(model_dir_path: Path, desc: dict) -> None:
    md_path = model_dir_path / "model_description.md"
    
    lines = []
    lines.append(f"# Model Overview: {desc.get('model_name')}")
    lines.append("")
    
    lines.append("## Architecture Details")
    struct = desc.get("structure", {})
    backbones_list = struct.get("backbones", [])
    lines.append(f"- **Backbones Used**: {', '.join(backbones_list) if backbones_list else 'None'}")
    lines.append(f"- **Fusion Pattern**: `{struct.get('fusion_pattern', 'N/A')}`")
    lines.append(f"- **Fractal Depth (N)**: {struct.get('fractal_N', 'N/A')}")
    lines.append(f"- **Fractal Columns**: {struct.get('fractal_cols', 'N/A')}")
    
    total_p = struct.get("total_parameters", 0)
    trainable_p = struct.get("trainable_parameters", 0)
    if total_p:
        lines.append(f"- **Total Parameters**: {total_p:,}")
        lines.append(f"- **Trainable Parameters**: {trainable_p:,}")
    lines.append("")
    
    conv_block_list = struct.get("conv_block", [])
    if conv_block_list:
        lines.append("### Procedural Conv Block Layers")
        lines.append("```python")
        for layer in conv_block_list:
            lines.append(f"  {layer}")
        lines.append("```")
        lines.append("")
        
    lines.append("## Training Configuration")
    hp = desc.get("hyperparameters", {})
    lines.append("| Parameter | Value |")
    lines.append("| :--- | :--- |")
    for k, v in hp.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    
    lines.append("## Evaluation Performance")
    lines.append(f"- **Status**: {'★ Success' if desc.get('success') else '⚠️ Failed'}")
    lines.append(f"- **Task**: `{desc.get('task')}`")
    lines.append(f"- **Metric**: `{desc.get('metric')}`")
    if "avg_accuracy" in desc:
        try:
            lines.append(f"- **Average Test Accuracy**: `{float(desc['avg_accuracy']):.4%}`")
        except (TypeError, ValueError):
            lines.append(f"- **Average Test Accuracy**: `{desc['avg_accuracy']}`")
    lines.append("")

    ds_res = desc.get("dataset_results", {})
    if ds_res:
        lines.append("### Performance per Dataset")
        lines.append("| Dataset | Status | Test Accuracy | Train Accuracy | Test Loss | Train Loss | Throughput |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for ds_name, res in ds_res.items():
            if res.get("success", True) is not False:
                acc_val = res.get("accuracy")
                acc_str = f"{float(acc_val):.4%}" if acc_val is not None else "N/A"
                tr_acc_val = res.get("train_accuracy")
                tr_acc_str = f"{float(tr_acc_val):.4%}" if tr_acc_val is not None else "N/A"
                t_loss = res.get("test_loss", "N/A")
                tr_loss = res.get("train_loss", "N/A")
                speed = res.get("samples_per_second")
                speed_str = f"{float(speed):.2f}/s" if speed is not None else "N/A"
                lines.append(f"| {ds_name} | ★ Success | `{acc_str}` | `{tr_acc_str}` | `{t_loss}` | `{tr_loss}` | `{speed_str}` |")
            else:
                lines.append(f"| {ds_name} | ⚠️ Failed | - | - | - | - | - |")
        lines.append("")

    if desc.get("success"):
        res = desc.get("results", {})
        lines.append("## Hardware Footprint")
        hw = desc.get("hardware", {})
        lines.append(f"- **GPU**: `{hw.get('gpu_type', 'N/A')}`")
        lines.append(f"- **CPU**: `{hw.get('cpu_type', 'N/A')} ({hw.get('cpu_count', 'N/A')} cores)`")
    else:
        lines.append(f"- **Error details**: {desc.get('error')}")
        
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _update_model_description(
    model_dir_path: Path,
    spec: Dict[str, Any],
    result: Dict[str, Any],
    success: bool
) -> None:
    import json
    desc_path = model_dir_path / "model_description.json"
    
    description = {}
    if desc_path.exists():
        try:
            description = json.loads(desc_path.read_text(encoding="utf-8"))
        except Exception:
            pass
            
    if not description:
        description = {
            "model_name": model_dir_path.name,
            "structure": {}
        }
        
    description["success"] = success
    description["task"] = spec.get("task")
    description["dataset"] = spec.get("dataset")
    description["metric"] = spec.get("metric")
    
    # Store multiple datasets
    description["datasets"] = list(set(description.get("datasets", []) + [spec.get("dataset")]))
    if "dataset_results" not in description:
        description["dataset_results"] = {}
        
    prm = spec.get("prm", {})
    description["hyperparameters"] = {
        "lr": prm.get("lr"),
        "batch": prm.get("batch"),
        "dropout": prm.get("dropout"),
        "momentum": prm.get("momentum"),
        "transform": prm.get("transform"),
        "max_steps": prm.get("max_steps"),
        "epoch": prm.get("epoch")
    }
    
    ds_name = spec.get("dataset", "unknown")
    if success:
        accuracy = float(result.get("accuracy") or 0.0)
        prm_inner = result.get("eval_args", {}).get("prm", {})
        description["dataset_results"][ds_name] = {
            "accuracy": accuracy,
            "train_loss": prm_inner.get("train_loss"),
            "test_loss": prm_inner.get("test_loss"),
            "train_accuracy": prm_inner.get("train_accuracy"),
            "samples_per_second": prm_inner.get("samples_per_second"),
            "duration_ns": prm_inner.get("duration")
        }
        
        # Calculate average test accuracy across all successful datasets
        accs = [res["accuracy"] for res in description["dataset_results"].values() if "accuracy" in res]
        if accs:
            description["avg_accuracy"] = sum(accs) / len(accs)
            
        description["results"] = description["dataset_results"][ds_name]
        
        code_file_path = Path(spec["code_file"])
        in_shape, num_classes = _get_dataset_specs(ds_name)
        total_p, trainable_p, backbone_p = _get_param_count(code_file_path, prm, in_shape, num_classes)
        description["structure"]["total_parameters"] = total_p
        description["structure"]["trainable_parameters"] = trainable_p
        description["structure"]["backbone_param_fraction"] = (
            round(backbone_p / total_p, 4) if total_p > 0 else None
        )
        
        description["hardware"] = {
            "gpu_type": prm_inner.get("gpu_type"),
            "cpu_type": prm_inner.get("cpu_type"),
            "cpu_count": prm_inner.get("cpu_count")
        }
    else:
        description["dataset_results"][ds_name] = {
            "success": False,
            "error": result.get("error", "Unknown error during evaluation")
        }
        description["error"] = result.get("error", "Unknown error during evaluation")
        
    # Generate natural-language summary (best-effort; runs after all fields are set)
    try:
        description["nl_summary"] = _generate_nl_summary(description)
    except Exception as e:
        print(f"Failed to generate nl_summary for {model_dir_path.name}: {e}")

    desc_path.write_text(json.dumps(description, indent=4), encoding="utf-8")

    try:
        _write_markdown_description(model_dir_path, description)
    except Exception as e:
        print(f"Failed to generate markdown description for {model_dir_path.name}: {e}")
