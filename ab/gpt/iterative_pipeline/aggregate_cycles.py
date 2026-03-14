#!/usr/bin/env python3
"""
Aggregate Results Across Fine-Tuning Cycles

Compiles metrics from all cycles and computes improvement trends.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

from ab.nn.util.Const import out_dir


def load_cycle_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load results from all_cycles_results.json."""
    if not results_file.exists():
        print(f"[ERROR] Results file not found: {results_file}")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def compute_improvements(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute improvement metrics from cycle 1 to final cycle."""
    successful_cycles = [r for r in results if r.get("success", False)]
    
    if len(successful_cycles) < 2:
        return {}
    
    first_cycle = successful_cycles[0]
    final_cycle = successful_cycles[-1]
    
    improvements = {
        "success_rate": {
            "first": first_cycle["evaluation"]["success_rate"],
            "final": final_cycle["evaluation"]["success_rate"],
            "delta": final_cycle["evaluation"]["success_rate"] - first_cycle["evaluation"]["success_rate"],
            "relative_improvement": (
                (final_cycle["evaluation"]["success_rate"] / max(0.01, first_cycle["evaluation"]["success_rate"]) - 1) * 100
                if first_cycle["evaluation"]["success_rate"] > 0 else 0
            ),
        },
        "best_accuracy": {
            "first": first_cycle["evaluation"]["best_accuracy"],
            "final": final_cycle["evaluation"]["best_accuracy"],
            "delta": final_cycle["evaluation"]["best_accuracy"] - first_cycle["evaluation"]["best_accuracy"],
            "relative_improvement": (
                (final_cycle["evaluation"]["best_accuracy"] / max(0.01, first_cycle["evaluation"]["best_accuracy"]) - 1) * 100
                if first_cycle["evaluation"]["best_accuracy"] > 0 else 0
            ),
        },
        "avg_accuracy": {
            "first": first_cycle["evaluation"]["avg_accuracy"],
            "final": final_cycle["evaluation"]["avg_accuracy"],
            "delta": final_cycle["evaluation"]["avg_accuracy"] - first_cycle["evaluation"]["avg_accuracy"],
            "relative_improvement": (
                (final_cycle["evaluation"]["avg_accuracy"] / max(0.01, first_cycle["evaluation"]["avg_accuracy"]) - 1) * 100
                if first_cycle["evaluation"]["avg_accuracy"] > 0 else 0
            ),
        },
        "training_data_size": {
            "first": first_cycle["training"]["total_examples"],
            "final": final_cycle["training"]["total_examples"],
            "delta": final_cycle["training"]["total_examples"] - first_cycle["training"]["total_examples"],
            "growth_rate": (
                final_cycle["training"]["total_examples"] / max(1, first_cycle["training"]["total_examples"]) - 1
            ) * 100,
        },
    }
    
    return improvements


def aggregate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create comprehensive summary of all cycles."""
    successful_cycles = [r for r in results if r.get("success", False)]
    
    summary = {
        "total_cycles_attempted": len(results),
        "successful_cycles": len(successful_cycles),
        "failed_cycles": len(results) - len(successful_cycles),
    }
    
    if not successful_cycles:
        return summary
    
    # Per-cycle metrics
    summary["cycles"] = []
    for result in successful_cycles:
        cycle_summary = {
            "cycle": result["cycle"],
            "success_rate": result["evaluation"]["success_rate"],
            "best_accuracy": result["evaluation"]["best_accuracy"],
            "avg_accuracy": result["evaluation"]["avg_accuracy"],
            "models_trained": result["evaluation"]["models_trained"],
            "training_examples": result["training"]["total_examples"],
            "new_examples_added": result["training"]["new_examples_added"],
            "training_time_minutes": result["training"]["training_time_minutes"],
            "cycle_time_minutes": result.get("cycle_time_minutes", 0),
        }
        summary["cycles"].append(cycle_summary)
    
    # Compute improvements
    summary["improvements"] = compute_improvements(results)
    
    # Overall statistics
    summary["overall"] = {
        "total_models_generated": sum(r["generation"]["total_generated"] for r in successful_cycles),
        "total_models_evaluated": sum(r["evaluation"]["models_trained"] for r in successful_cycles),
        "total_novel_models": sum(r["generation"].get("novel", 0) for r in successful_cycles),
        "total_examples_added": sum(r["training"]["new_examples_added"] for r in successful_cycles),
        "total_time_hours": sum(r.get("cycle_time_minutes", 0) for r in successful_cycles) / 60,
        "best_accuracy_overall": max(r["evaluation"]["best_accuracy"] for r in successful_cycles),
        "avg_accuracy_final": successful_cycles[-1]["evaluation"]["avg_accuracy"] if successful_cycles else 0,
    }
    
    return summary


def print_summary(summary: Dict[str, Any]):
    """Print formatted summary."""
    print("\\n" + "="*80)
    print("ITERATIVE FINE-TUNING SUMMARY")
    print("="*80 + "\\n")
    
    print(f"Total Cycles: {summary['total_cycles_attempted']}")
    print(f"Successful: {summary['successful_cycles']}")
    print(f"Failed: {summary['failed_cycles']}")
    
    if "cycles" not in summary or not summary["cycles"]:
        print("\\n[ERROR] No successful cycles to report")
        return
    
    print("\\n" + "-"*80)
    print("PER-CYCLE RESULTS")
    print("-"*80)
    print(f"{'Cycle':<8} {'Success Rate':<15} {'Best Acc':<12} {'Avg Acc':<12} {'Examples':<10} {'Time (min)':<12}")
    print("-"*80)
    
    for cycle in summary["cycles"]:
        print(f"{cycle['cycle']:<8} "
              f"{cycle['success_rate']*100:>6.1f}%        "
              f"{cycle['best_accuracy']*100:>6.2f}%     "
              f"{cycle['avg_accuracy']*100:>6.2f}%     "
              f"{cycle['training_examples']:<10} "
              f"{cycle['cycle_time_minutes']:>6.1f}")
    
    if "improvements" in summary and summary["improvements"]:
        print("\\n" + "-"*80)
        print("IMPROVEMENT METRICS (First Cycle → Final Cycle)")
        print("-"*80 + "\\n")
        
        imp = summary["improvements"]
        
        print("Success Rate:")
        print(f"  First Cycle: {imp['success_rate']['first']*100:.1f}%")
        print(f"  Final Cycle: {imp['success_rate']['final']*100:.1f}%")
        print(f"  Delta: {imp['success_rate']['delta']*100:+.1f}%")
        print(f"  Relative Improvement: {imp['success_rate']['relative_improvement']:+.1f}%")
        
        print("\\nBest Accuracy:")
        print(f"  First Cycle: {imp['best_accuracy']['first']*100:.2f}%")
        print(f"  Final Cycle: {imp['best_accuracy']['final']*100:.2f}%")
        print(f"  Delta: {imp['best_accuracy']['delta']*100:+.2f}%")
        print(f"  Relative Improvement: {imp['best_accuracy']['relative_improvement']:+.1f}%")
        
        print("\\nAverage Accuracy:")
        print(f"  First Cycle: {imp['avg_accuracy']['first']*100:.2f}%")
        print(f"  Final Cycle: {imp['avg_accuracy']['final']*100:.2f}%")
        print(f"  Delta: {imp['avg_accuracy']['delta']*100:+.2f}%")
        print(f"  Relative Improvement: {imp['avg_accuracy']['relative_improvement']:+.1f}%")
        
        print("\\nTraining Data Size:")
        print(f"  First Cycle: {imp['training_data_size']['first']} examples")
        print(f"  Final Cycle: {imp['training_data_size']['final']} examples")
        print(f"  Growth: +{imp['training_data_size']['delta']} examples ({imp['training_data_size']['growth_rate']:+.1f}%)")
    
    if "overall" in summary:
        print("\\n" + "-"*80)
        print("OVERALL STATISTICS")
        print("-"*80 + "\\n")
        
        overall = summary["overall"]
        print(f"Total Models Generated: {overall['total_models_generated']}")
        print(f"Total Models Evaluated: {overall['total_models_evaluated']}")
        print(f"Total Novel Models: {overall['total_novel_models']}")
        print(f"Total Training Examples Added: {overall['total_examples_added']}")
        print(f"Total Pipeline Time: {overall['total_time_hours']:.1f} hours")
        print(f"Best Accuracy Achieved: {overall['best_accuracy_overall']*100:.2f}%")
        print(f"Final Average Accuracy: {overall['avg_accuracy_final']*100:.2f}%")
    
    print("\\n" + "="*80)


def save_summary(summary: Dict[str, Any], output_file: Path):
    """Save summary to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\\nSummary saved to: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate results across fine-tuning cycles")
    parser.add_argument("--results_dir", type=str, default=out_dir / 'iterative_cycles',
                        help="Directory containing all_cycles_results.json")
    parser.add_argument("--output_file", type=str, default=out_dir / 'iterative_cycles' / 'cycles_summary.json',
                        help="Output file for aggregated summary")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_dir) / "all_cycles_results.json"
    
    print("Loading cycle results...")
    results = load_cycle_results(results_file)
    
    print("Computing summary...")
    summary = aggregate_summary(results)
    
    print_summary(summary)
    
    output_file = Path(args.output_file)
    save_summary(summary, output_file)


if __name__ == "__main__":
    main()



