
import json
import subprocess
from pathlib import Path

NNGPT_DIR = Path("out/nngpt")
LINEAGE_FILE = NNGPT_DIR / "accepted_adapters.json"
IMPROVEMENT_EPS = 1e-4


def infer_base_model():
    #Read base model from  adapter_config.json
    for cfg_path in (NNGPT_DIR / "llm" / "epoch").rglob("adapter_config.json"):
        if "synth_nn" not in str(cfg_path):
            with open(cfg_path) as f:
                return json.load(f)["base_model_name_or_path"]
    raise RuntimeError("No adapters found to infer base model")


def load_lineage():
   #load previous adapters or make a new one
    if LINEAGE_FILE.exists():
        with open(LINEAGE_FILE) as f:
            data = json.load(f)
        data.setdefault("adapters", [])
        data.setdefault("base_model", infer_base_model())
        return data

    return {
        "adapters": [],
        "base_model": infer_base_model()
    }


def save_lineage(lineage): #save the adapters
    LINEAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LINEAGE_FILE, "w") as f:
        json.dump(lineage, f, indent=2)


def get_cycle_files(): #get cycle_results.json for the
    current = NNGPT_DIR / "cycle_results.json"
    if not current.exists():
        raise RuntimeError("cycle_results.json not found")

    backups = list(NNGPT_DIR.glob("cycle_results_*.json"))
    if not backups:
        raise RuntimeError("No backup cycle_results_*.json found")

    previous = max(backups, key=lambda p: p.stat().st_mtime) #sort by time to see two latest cycle_results.json
    return previous, current


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)

    evaluation = data.get("evaluation", {})

    return {
        "accuracy": evaluation.get("best_accuracy"),
        "cycle": data.get("cycle")
    }


def rebuild_model():
#if accuracy increases adapter is merged
    print("REBUILDING MODEL FROM UPDATED LINEAGE")


    try:
        subprocess.run(
            ["python", "-m", "ab.gpt.util.MergeLLM"],
            check=True
        )
        print("Rebuild complete\n")
        return True
    except subprocess.CalledProcessError as e:
        print("Rebuild failed")
        print(e)
        return False


def main():
    try:
        prev_file, curr_file = get_cycle_files()
        prev_m = load_metrics(prev_file)
        curr_m = load_metrics(curr_file)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    # --- Safe accuracy extraction ---
    prev_acc = prev_m.get("accuracy") or 0.0
    curr_acc = curr_m.get("accuracy") or 0.0

    current_cycle = curr_m.get("cycle")
    if current_cycle is None:
        raise RuntimeError("Cycle number missing in cycle_results.json")

    current_epoch = current_cycle - 1


    print("CYCLE COMPARISON")
    print(f"Previous: {prev_file.name} → Acc: {prev_acc:.6f}")
    print(f"Current : {curr_file.name} → Acc: {curr_acc:.6f}")


    # --- Decision rule: accuracy only ---
    delta = curr_acc - prev_acc

    if delta > IMPROVEMENT_EPS:
        decision = "KEEP"
        reason = f"Improved by {delta:.6f}"
    else:
        decision = "REVERT"
        reason = f"Delta {delta:.6f} below threshold"

    print(f"\nDecision: {decision} ({reason})")

    # --- Load lineage ---
    lineage = load_lineage()
    accepted_epochs = [a["epoch"] for a in lineage["adapters"]]

    # --- Update lineage ---
    if decision == "KEEP":
        if current_epoch not in accepted_epochs:
            lineage["adapters"].append({
                "epoch": current_epoch,
                "path": f"out/nngpt/llm/epoch/A{current_epoch}"
            })
            save_lineage(lineage)
            accepted_epochs.append(current_epoch)
            print(f"Adapter A{current_epoch} ACCEPTED")
        else:
            print(f"Adapter A{current_epoch} already accepted")
    else:
        print(f"Adapter A{current_epoch} REJECTED")

    print(f"Accepted lineage: {accepted_epochs}\n")

    # --- Save decision summary ---
    with open(NNGPT_DIR / "merge_decision.json", "w") as f:
        json.dump({
            "current_epoch": current_epoch,
            "previous_accuracy": prev_acc,
            "current_accuracy": curr_acc,
            "delta": delta,
            "decision": decision,
            "reason": reason,
            "accepted_lineage": accepted_epochs
        }, f, indent=2) #write all info in the file

    print("Decision saved\n")

    # --- Rebuild if KEEP ---
    if decision == "KEEP":
        print("Triggering model rebuild...")
        rebuild_model()
        print("✓ Ready for next cycle\n")
    else:
        print("No rebuild needed\n")


if __name__ == "__main__":
    main()