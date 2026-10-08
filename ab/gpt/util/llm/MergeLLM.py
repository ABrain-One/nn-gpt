


import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ab.gpt.util.Const import nngpt_dir, nngpt_upload

# Config
TOLERANCE = 0.0


def infer_base_model():
    """Detect base model from run_config.json, llm_conf, or adapter_config.json"""
    print("[BASE MODEL] Detection started")
    print(f"[DEBUG] nngpt_dir = {nngpt_dir}")

    run_config = nngpt_dir / "run_config.json"

    if run_config.exists():
        print(f"[BASE MODEL] Checking {run_config}")
        try:
            with open(run_config) as f:
                cfg = json.load(f)

            base = cfg.get("base_model_name")
            if base:
                print(f"[BASE MODEL] ✓ Found in run_config.json")
                print(f"[BASE MODEL] → {base}")
                print(f"[DEBUG] Base model type: {type(base)}")
                print(
                    f"[DEBUG] Base model path exists: {Path(base).exists() if not base.startswith('deepseek') else 'N/A (HuggingFace ID)'}")

                return base

            llm_conf = cfg.get("llm_conf")
            if llm_conf:
                from ab.gpt.util.Const import conf_llm_dir
                conf_path = conf_llm_dir / llm_conf

                print(f"[BASE MODEL] Checking llm_conf: {conf_path}")
                if conf_path.exists():
                    try:
                        with open(conf_path) as f:
                            data = json.load(f)
                        base = data.get("base_model_name")
                        if base:
                            print(f"[BASE MODEL] ✓ Found in llm_conf")
                            print(f"[BASE MODEL] → {base}")
                            print(f"[DEBUG] Base model type: {type(base)}")
                            return base
                    except Exception as e:
                        print(f"[BASE MODEL] ✗ Failed to read llm_conf: {e}")
        except Exception as e:
            print(f"[BASE MODEL] ✗ Failed to read run_config.json: {e}")

    # Fallback to adapter configs
    print(f"[BASE MODEL] Falling back to adapter configs...")
    epoch_root = nngpt_dir / "llm" / "epoch"
    if not epoch_root.exists():
        raise RuntimeError("Cannot determine base model")

    latest = find_latest_epoch()
    print(f"[DEBUG] Latest epoch: {latest}")
    for e in range(latest, -1, -1):
        cfg_path = epoch_root / f"A{e}" / "adapter_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    data = json.load(f)
                base = data.get("base_model_name_or_path")
                if base:
                    print(f"[BASE MODEL] ✓ Found in adapter config A{e}")
                    print(f"[BASE MODEL] → {base}")
                    return base
            except Exception as e:
                continue

    raise RuntimeError("Cannot determine base model")


def find_latest_epoch():
    """Find highest epoch number"""
    epoch_root = nngpt_dir / "llm" / "epoch"
    if not epoch_root.exists():
        return 0

    epochs = []
    for p in epoch_root.iterdir():
        if p.is_dir() and p.name.startswith("A"):
            try:
                epochs.append(int(p.name[1:]))
            except ValueError:
                continue
    return max(epochs) if epochs else 0


def find_adapter(epoch: int):
    """Find adapter for epoch (with fallback to earlier epochs)"""
    print(f"\n[ADAPTER] Searching for epoch {epoch}...")

    for e in [epoch]:
        path = nngpt_dir / "llm" / "epoch" / f"A{e}"
        if not path.exists():
            continue

        candidates = list(path.rglob("adapter_model.safetensors"))
        if not candidates:
            print(f"[ADAPTER] A{e} has no weights")
            continue

        adapter_dir = candidates[0].parent
        print(f"[ADAPTER] ✓ Found A{e}")
        print(f"[ADAPTER] Path: {adapter_dir}\n")
        return adapter_dir, e

    print(f"[ADAPTER] ✗ No adapters found ≤ epoch {epoch}\n")
    return None, None


def load_lineage():
    """Load lineage.json"""
    path = nngpt_dir / "lineage.json"
    if not path.exists():
        print(f"[DEBUG] lineage.json not found, starting fresh")
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[DEBUG] Loaded {len(data)} entries from lineage.json")
        return data
    except Exception as e:
        print(f"[LINEAGE] ✗ Failed to load: {e}")
        return []


def save_lineage(lineage):
    """Save lineage.json"""
    path = nngpt_dir / "lineage.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    unique = {}
    for x in lineage:
        key = (x["epoch"], x["used_epoch"])
        unique[key] = x

    with open(path, "w") as f:
        json.dump(list(unique.values()), f, indent=2)

    print(f"[DEBUG] Saved {len(unique)} entries to lineage.json")


def merge_multiple_adapters(base_model, adapter_paths, output_path):

    import gc
    import shutil

    print(f"[MERGE] Base: {base_model}")
    print(f"[MERGE] Adapters: {len(adapter_paths)}")

    offload_dir = Path("/tmp/nngpt_offload")
    offload_dir.mkdir(parents=True, exist_ok=True)

    # --- Load base model on CPU with optional disk offload ---
    print("[MERGE] Loading base model (CPU/offload)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder=str(offload_dir),
        offload_state_dict=True
    )
    print("[MERGE] ✓ Base model loaded")

    # --- Apply adapters sequentially ---
    for i, adapter_path in enumerate(adapter_paths):
        print(f"\n[MERGE] Adapter {i + 1}/{len(adapter_paths)}")
        print(f"[MERGE] Path: {adapter_path}")

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
            device_map="auto",
            offload_folder=str(offload_dir)
        )

        print("[MERGE] → merging...")
        model = model.merge_and_unload()

        # --- aggressive cleanup ---
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[MERGE] ✓ Done {i + 1}/{len(adapter_paths)}")

    # --- Save ---
    print(f"\n[MERGE] Saving → {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    print("[MERGE] Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_paths[-1]), trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # --- Cleanup offload directory ---
    shutil.rmtree(offload_dir, ignore_errors=True)

    print("\n[MERGE] ✓✓✓ COMPLETE ✓✓✓\n")

def rebuild_from_lineage():
    """
    Auto-merge: Pick best epoch from ALL available epochs.
    No buffer, no constraints - just find best and merge.
    """

    print("AUTO MERGE - BEST EPOCH FROM ALL")

    print(f"[DEBUG] nngpt_dir = {nngpt_dir}")
    print(f"[DEBUG] nngpt_upload = {nngpt_upload}")

    # Load epoch tracker
    epoch_tracker_file = nngpt_dir / "epoch_tracker.json"
    print(f"[DEBUG] Looking for epoch_tracker at: {epoch_tracker_file}")

    if not epoch_tracker_file.exists():
        print(f"[ERROR] epoch_tracker.json not found")
        return

    with open(epoch_tracker_file) as f:
        tracker_list = json.load(f)

    if not tracker_list:
        print("[ERROR] epoch_tracker.json is empty")
        return

    print(f"[INFO] Found {len(tracker_list)} epochs")

    # Find best epoch by accuracy
    valid_epochs = []
    for entry in tracker_list:
        epoch = entry.get("epoch")
        acc = entry.get("accuracy")
        score = entry.get("score")  # success_rate × accuracy

        if epoch is None:
            continue

        if acc is None or acc == 0.0:
            print("No accuracy for epoch {}")
            continue
            #acc = min(0.40 + epoch * 0.03, 0.99)  # Fake for testing
            #print(f"[EPOCH {epoch}] Using fake accuracy: {acc:.4f}")

        sort_key = float(score) if score else float(acc)
        valid_epochs.append((epoch, float(acc), sort_key))
        print(f"[EPOCH {epoch}] Accuracy: {acc:.4f}  Score: {sort_key:.4f}")

    if not valid_epochs:
        print("\n[ERROR] No valid epochs")
        return

    # Pick best
    valid_epochs.sort(key=lambda x: x[2], reverse=True)
    best_epoch, best_acc, best_score = valid_epochs[0]
    print(f"\n[SELECTION] Best epoch: {best_epoch}, "
          f"Accuracy: {best_acc:.4f}, Score: {best_score:.4f}")

    print(f"\n[SELECTION] Best epoch: {best_epoch}, Accuracy: {best_acc:.4f}")

    # Find adapter
    adapter_path, used_epoch = find_adapter(best_epoch)
    if adapter_path is None:
        print(f"[ERROR] No adapter for epoch {best_epoch}")
        return

    # Build chain
    lineage = load_lineage()
    merged = [x for x in lineage if x.get("merged")]

    print(f"\n[CHAIN] Building adapter chain...")
    print(f"[DEBUG] Found {len(merged)} previously merged adapters")
    chain = []
    for x in merged:
        p, _ = find_adapter(x["used_epoch"])
        if p:
            chain.append(p)
            print(f"[CHAIN] + A{x['used_epoch']}")

    chain.append(adapter_path)
    print(f"[CHAIN] + A{used_epoch} (best)")
    print(f"\n[CHAIN] Total: {len(chain)} adapters")

    # Merge
    base_model = infer_base_model()
    output_path = nngpt_upload / Path(base_model).name

    print(f"\n[DEBUG] === MERGE PREPARATION ===")
    print(f"[DEBUG] Base model: {base_model}")
    print(f"[DEBUG] Output path: {output_path}")
    print(f"[DEBUG] Chain length: {len(chain)}")
    print(f"[DEBUG] === STARTING MERGE ===\n")

    merge_multiple_adapters(base_model, chain, output_path)

    # Update run_config
    run_config = nngpt_dir / "run_config.json"
    cfg = {}
    if run_config.exists():
        with open(run_config) as f:
            cfg = json.load(f)

    old_base = cfg.get("base_model_name", "None")
    cfg["base_model_name"] = str(output_path)

    with open(run_config, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[CONFIG] Updated run_config.json")
    print(f"[CONFIG] Old base_model_name: {old_base}")
    print(f"[CONFIG] New base_model_name: {output_path}")
    print(f"[CONFIG] Next run will use merged model ✓\n")

    # Update lineage
    new_entry = {
        "epoch": best_epoch,
        "used_epoch": used_epoch,
        "accuracy": best_acc,
        "merged": True,
        "chain_length": len(chain),
        "base_model": base_model
    }
    lineage.append(new_entry)
    save_lineage(lineage)

    print(f"\n[AUTO MERGE] ✓✓✓ COMPLETE ✓✓✓\n")


def merge_nn_llm(epoch: int):
    """Manual merge of specific epoch"""
    print(f"\n[MANUAL] Merging epoch {epoch}")

    adapter_path, used_epoch = find_adapter(epoch)
    if adapter_path is None:
        print(f"[ERROR] No adapter for epoch {epoch}")
        return

    # Build chain
    lineage = load_lineage()
    merged = [x for x in lineage if x.get("merged")]

    chain = []
    for x in merged:
        p, _ = find_adapter(x["used_epoch"])
        if p:
            chain.append(p)

    chain.append(adapter_path)

    base_model = infer_base_model()
    output_path = nngpt_upload / Path(base_model).name
    merge_multiple_adapters(base_model, chain, output_path)

    # Update run_config
    run_config = nngpt_dir / "run_config.json"
    cfg = {}
    if run_config.exists():
        with open(run_config) as f:
            cfg = json.load(f)

    old_base = cfg.get("base_model_name", "None")
    cfg["base_model_name"] = str(output_path)

    with open(run_config, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[CONFIG] Updated run_config.json")
    print(f"[CONFIG] Old base_model_name: {old_base}")
    print(f"[CONFIG] New base_model_name: {output_path}")
    print(f"[CONFIG] Next run will use merged model ✓\n")

    print(f"[MANUAL] ✓ Complete\n")


if __name__ == "__main__":
    rebuild_from_lineage()
    #merge_nn_llm(0)
