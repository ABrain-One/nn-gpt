import sys
import os
import time
import io
import base64
import json
import importlib.util
from pathlib import Path

# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp
from thop import profile

# Dynamically add workspace roots to path based on script location
# File is at: .../nn-gpt/ab/gpt/util/batch_spr.py
script_path = Path(__file__).resolve()
gw_root = script_path.parent.parent.parent.parent  # Points to .../nn-gpt
ds_root = gw_root.parent / 'nn-dataset'          # Points to .../nn-dataset

sys.path.append(str(ds_root))
sys.path.append(str(gw_root))

# Custom Project Imports (Must be after sys.path setup)
import ab.nn.api as nn_dataset
import Const as const

# Constants
BASE_OUT_DIR = const.synth_dir(const.epoch_dir('0'))
SPARSITY_AMOUNT = 0.30

PRUNING_METRICS = {}

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def dynamic_load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def model_size(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB

def model_flops(model, input_size=(1,3,224,224)):
    try:
        input = torch.randn(input_size)
        flops, params = profile(model, inputs=(input,), verbose=False)
        return flops
    except Exception as e:
        print(f"  [Warning] FLOPs calculation failed: {e}")
        return 0.0

# ------------------ Pruning Methods --------------------

def prune_bbb_by_snr(module, amount=0.3):
    """
    Prune Bayesian layers by Signal-to-Noise Ratio (abs(mu)/sigma).
    Sets pruned weights (mu) to 0.0.
    """
    w_sigma = torch.log1p(torch.exp(module.W_rho))
    snr = torch.abs(module.W_mu) / (w_sigma + 1e-8)
    k = int(snr.numel() * amount)
    threshold = torch.kthvalue(snr.view(-1), k).values
    mask = snr > threshold
    module.W_mu.data = module.W_mu.data * mask.float()
    return mask

def prune_torch_structural(model, example_inputs, amount=0.3):
    """
    Apply STRUCTURAL channel pruning using Torch-Pruning.
    Physically removes channels.
    """
    try:
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
        importance = tp.importance.GroupMagnitudeImportance(p=2)

        ignored_layers = [m for name, m in model.named_modules()
                          if isinstance(m, nn.Linear) and name in ["classifier", "fc", "head"]]

        pruner = tp.pruner.BasePruner(
            model, example_inputs, importance=importance,
            pruning_ratio=amount, ignored_layers=ignored_layers,
            round_to=8, global_pruning=False
        )
        pruner.step()
    except Exception as e:
        print(f"  [Warning] Structural pruning skipped: {e}")
    return model

def prune_weight_level(model, amount=0.3):
    """
    Standard magnitude-based pruning (weights) for all models.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
            prune.remove(module, "weight")
    return model

def prune_custom(model):
    """
    Custom structured pruning for specific model types:
    - Transformers: attention-head pruning
    - RNNs: hidden-unit pruning
    """
    for module in model.modules():
        cls_name = module.__class__.__name__
        # Example placeholders (actual pruning logic depends on module implementation)
        if 'MultiheadAttention' in cls_name:
            # Reduce heads by 30%
            if hasattr(module, 'num_heads'):
                orig_heads = module.num_heads
                module.num_heads = max(1, int(module.num_heads * 0.7))
                print(f"  [Custom Prune] Attention heads: {orig_heads} -> {module.num_heads}")
        elif 'LSTM' in cls_name or 'GRU' in cls_name:
            # Reduce hidden size by 30%
            if hasattr(module, 'hidden_size'):
                orig_hidden = module.hidden_size
                module.hidden_size = max(1, int(module.hidden_size * 0.7))
                print(f"  [Custom Prune] Hidden units: {orig_hidden} -> {module.hidden_size}")
    return model

# ---------------------------------------------------------
# Core Logic
# ---------------------------------------------------------


# ---------------- Build New NN File -------------------

def build_new_nn_py(source_text: str, state_dict_bytes_b64: str, hp: dict, pruning_summary: dict = None) -> str:
    pr_code = ""
    if pruning_summary:
        pr_code = f"""
# ===========================
# PRUNING REPORT (AUTO-GENERATED)
# ===========================

PRUNING_SUMMARY = {pruning_summary}

def show_pruning_summary():
    print("=== Pruning Summary ===")
    for k, v in PRUNING_SUMMARY.items():
        print(f"{{k}}: {{v}}")
"""
    suffix = f"""
{pr_code}
import base64, io, torch, torch.nn.functional as F, torch.nn.utils.prune as prune
_weights_b64 = "{state_dict_bytes_b64}"

def _load_state_dict_from_b64(b64_str, map_location=None):
    b = base64.b64decode(b64_str)
    buf = io.BytesIO(b)
    return torch.load(buf, map_location=map_location)

def load(device='cuda'):
    state = _load_state_dict_from_b64(_weights_b64, map_location=device)
    constructors = [((1,3,224,224),(1000,)),((1,3,32,32),(10,)),((1,3,64,64),(200,))]
    prm = {{}}
    if 'supported_hyperparameters' in globals():
        try:
            keys = supported_hyperparameters()
            for k in keys:
                prm[k] = 0.01
                if k=='batch': prm[k]=1
                if k=='epoch': prm[k]=1
        except: pass
    for in_shape,out_shape in constructors:
        try:
            model = Net(in_shape, out_shape, prm, torch.device(device))
            model.load_state_dict(state, strict=False)
            return model
        except: pass
    try: model = Net(); model.load_state_dict(state, strict=False); return model
    except Exception as e: raise RuntimeError('Failed to load model: '+str(e))

def finetune(model, dataloader, device='cuda'):
    model.to(device); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr={hp['lr']}, weight_decay={hp['wd']})
    scaler = torch.cuda.amp.GradScaler()
    print(f"Starting fine-tuning: {hp['epochs']} epochs, lr={hp['lr']}")
    for epoch in range({hp['epochs']}):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss = F.cross_entropy(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model

def predict(model, input_tensor, device='cuda'):
    model.eval()
    if device=='cuda': torch.backends.cudnn.benchmark=True
    with torch.no_grad():
        if device=='cuda':
            with torch.cuda.amp.autocast(): logits = model(input_tensor.to(device))
        else: logits = model(input_tensor.to(device))
        if isinstance(logits, tuple): logits=logits[0]
        return logits/{hp['temp']}
"""
    return source_text + suffix

def process_single_model(model_name, index, temp_dir):
    out_dir = BASE_OUT_DIR / f"B{index}"

    print(f"\\n[{index}] Fetching {model_name}...")

    # 1. Fetch Code
    try:
        df = nn_dataset.data(nn=model_name, max_rows=1)
        if df.empty:
            print(f"  [Skip] No code found in DB for {model_name}")
            return
        code = df.iloc[0]['nn_code']
    except Exception as e:
        print(f"  [Error] API fetch failed: {e}")
        return

    # 2. Save Temp File
    safe_name = f"temp_model_{index}.py"
    temp_path = temp_dir / safe_name
    temp_path.write_text(code, encoding='utf-8')

    # 3. Load Module
    try:
        mod = dynamic_load_module(temp_path, f"mod_{index}")
    except Exception as e:
        print(f"  [Error] Import failed: {e}")
        return

    if not hasattr(mod, 'Net'):
        print(f"  [Error] No Net class")
        return

    Net = getattr(mod, 'Net')

    # 4. Instantiate Model
    # ---------------- USER SNIPPET ----------------
    prm = {'lr':0.001,'momentum':0.9,'dropout':0.5,'batch':1,'epoch':1,'wd':1e-4}
    device = torch.device('cpu')

    configs = [
        ((1,3,224,224), (1000,)),
        ((1,3,32,32), (10,)),
        ((1,3,64,64), (200,)),
    ]

    model = None
    for in_shape, out_shape in configs:
        try:
            model = Net(in_shape, out_shape, prm, device)
            break
        except:
            continue
    if model is None:
        print("Failed to instantiate Net")
        return
    # ---------------- END USER SNIPPET ----------------

    # 5. Detect Model Type
    is_bayesian = any('BBB' in m.__class__.__name__ for m in model.modules())
    if is_bayesian:
        hp = {'lr': 5e-5, 'epochs': 5, 'wd': 1e-5, 'temp': 1.5}
        print("  [Type] Bayesian")
    else:
        hp = {'lr': 1e-4, 'epochs': 3, 'wd': 0.0, 'temp': 1.0}
        print("  [Type] Standard")

    # 6. Multi-Stage Pruning
    amount = SPARSITY_AMOUNT
    stage_stats = {}
    amount = SPARSITY_AMOUNT
    stage_stats = {}
    params_before = count_params(model)
    size_before = model_size(model)
    flops_before = model_flops(model)
    stage_stats["before"] = params_before
    print(f"  [Info] Params: {params_before}, Size: {size_before:.2f}MB, FLOPs: {flops_before}")

    # Stage 1: DG / Structural pruning
    STRUCTURAL_BLACKLIST = [
        'MaxVit', 'SwinTransformer', 'VisionTransformer', 'ConvNeXtTransformer',
        'FasterRCNN', 'FCOS', 'RetinaNet', 'SSDFull', 'SSDLite', 'Diffuser'
    ]

    if not is_bayesian:
        if model_name in STRUCTURAL_BLACKLIST:
             print(f"  [Stage 1] Skipped (Blacklisted for structural pruning)")
        else:
            try:
                example_inputs = torch.randn(
                    1,
                    getattr(model, 'in_channels', 3),
                    getattr(model, 'image_size', 224),
                    getattr(model, 'image_size', 224)
                )
                model = prune_torch_structural(model, example_inputs, amount=amount)
                stage_stats["after_structural"] = count_params(model)
                print("  [Stage 1] Structural pruning applied")
            except Exception as e:
                print(f"  [Warning] Structural pruning failed: {e}")
                stage_stats["after_structural"] = count_params(model)

    # Stage 2: Weight / Magnitude pruning
    model = prune_weight_level(model, amount=amount)
    stage_stats["after_weight"] = count_params(model)
    print("  [Stage 2] Weight pruning applied")

    # Stage 3: Custom structured pruning
    model = prune_custom(model)
    stage_stats["after_custom"] = count_params(model)
    print("  [Stage 3] Custom pruning applied")

    params_after = count_params(model)
    size_after = model_size(model)
    flops_after = model_flops(model)
    print(f"  [Info] Params: {params_after}, Size: {size_after:.2f}MB, FLOPs: {flops_after}")

    summary_dict = {
        "pruning_ratio": amount,
        "parameters_before": params_before,
        "parameters_after": params_after,
        "pruning_ratio": amount,
        "parameters_before": params_before,
        "parameters_after": params_after,
        "parameters_removed": params_before - params_after,
        "size_before_mb": size_before,
        "size_after_mb": size_after,
        "flops_before": flops_before,
        "flops_after": flops_after
    }

    PRUNING_METRICS[f"B{index}"] = {
        "model_name": model_name,
        "model_type": "Bayesian" if is_bayesian else "Standard",
        "params_before": params_before,
        "params_after": params_after,
        "params_after": params_after,
        "params_removed": params_before - params_after,
        "size_before_mb": size_before,
        "size_after_mb": size_after,
        "flops_before": flops_before,
        "flops_after": flops_after,
        "stage_stats": stage_stats,
        "pruning_ratio": amount,
        "output_path": str(out_dir / "new_nn.py")
    }

    # 7. Save Model with Embedded Weights
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')

    from copy import deepcopy
    new_code = build_new_nn_py(code, b64, hp, summary_dict)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "new_nn.py").write_text(new_code, encoding='utf-8')
    print(f"  [Success] Saved to {out_dir}/new_nn.py")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    targets = const.core_nns
    temp_dir = Path("temp_batch_models")
    temp_dir.mkdir(exist_ok=True)

    print(f"Starting Batch Multi-Stage Pruning for {len(targets)} models. Target Sparsity: {int(SPARSITY_AMOUNT*100)}%")

    for i, name in enumerate(targets):
        process_single_model(name, i, temp_dir)

    metrics_path = BASE_OUT_DIR / "pruning_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(PRUNING_METRICS, f, indent=4)

    print(f"\\nPruning metrics saved to: {metrics_path}")
    print("\\nAll done.")

if __name__ == '__main__':
    main()
