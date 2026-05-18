#!/usr/bin/env python3
"""
Universal wrapper to evaluate models for any cycle through NNEval.
"""

import json
import sys
from pathlib import Path

from ab.nn.util.Const import out_dir

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from ab.gpt import NNEval


def evaluate_cycle_models(cycle: int, nneval_dir: Path):
    """
    Evaluate all models in a cycle's nneval directory.
    """
    print("=" * 80)
    print(f"EVALUATING CYCLE {cycle} MODELS")
    print("=" * 80)
    print()

    if not nneval_dir.exists():
        print(f"[ERROR] NNEval directory not found: {nneval_dir}", file=sys.stderr)
        sys.exit(1)

    model_dirs = sorted(path for path in nneval_dir.glob("gen_*") if path.is_dir())
    if not model_dirs:
        print(f"[ERROR] No model directories found in {nneval_dir}")
        sys.exit(1)

    print(f"Found {len(model_dirs)} models to evaluate")
    print()

    summary = NNEval.main(
        nn_name_prefix=None,
        nn_train_epochs=1,
        only_epoch=0,
        save_to_db=True,
        nn_alter_epochs=1,
        task="img-classification",
        dataset="cifar-10",
        metric="acc",
        lr=0.01,
        batch=16,
        dropout=0.2,
        momentum=0.9,
        transform="norm_256_flip",
        custom_synth_dir=str(nneval_dir),
        cycle=cycle,
        use_all_visible_gpus=True,
    )

    epoch_summaries = list(summary.get("epochs", []) or [])
    epoch_results = list(epoch_summaries[0].get("model_results", []) or []) if epoch_summaries else []
    all_results_list = sorted(epoch_results, key=lambda item: str(item.get("model_id", "")))
    total_successful = len([item for item in all_results_list if item.get("success") and "accuracy" in item])
    total_failed = len(all_results_list) - total_successful

    print()
    print("=" * 80)
    print(f"EVALUATION SUMMARY - CYCLE {cycle}")
    print("=" * 80)
    print(f"Total models: {len(all_results_list)}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print()

    if total_successful > 0:
        accuracies = [item["accuracy"] for item in all_results_list if item.get("success") and "accuracy" in item]
        print(f"Best accuracy: {max(accuracies) * 100:.2f}%")
        print(f"Average accuracy: {sum(accuracies) / len(accuracies) * 100:.2f}%")
        print()
        threshold = 0.25
        above_threshold = [
            item
            for item in all_results_list
            if item.get("success") and "accuracy" in item and float(item["accuracy"]) >= threshold
        ]
        print(f"Models above {threshold * 100}% threshold: {len(above_threshold)}/{total_successful}")

    results_file = nneval_dir.parent / "evaluation_results.json"
    results_data = {
        "cycle": cycle,
        "total_evaluated": len(all_results_list),
        "successful": total_successful,
        "failed": total_failed,
        "best_accuracy": (
            max([item["accuracy"] for item in all_results_list if item.get("success") and "accuracy" in item])
            if total_successful > 0
            else None
        ),
        "avg_accuracy": (
            sum([item["accuracy"] for item in all_results_list if item.get("success") and "accuracy" in item]) / total_successful
            if total_successful > 0
            else None
        ),
        "models": all_results_list,
    }
    results_file.write_text(json.dumps(results_data, indent=2, default=str), encoding="utf-8")
    print(f"✓ Results saved to: {results_file}")
    return results_data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate models for a cycle via NNEval")
    parser.add_argument("--cycle", type=int, required=True, help="Cycle number")
    parser.add_argument("--nneval_dir", type=str, help="Path to nneval directory (auto-detected if not provided)")

    args = parser.parse_args()

    if args.nneval_dir:
        nneval_dir = Path(args.nneval_dir)
    else:
        nneval_dir = out_dir / "iterative_cycles" / f"cycle_{args.cycle}/nneval"

    evaluate_cycle_models(args.cycle, nneval_dir)


if __name__ == "__main__":
    main()
