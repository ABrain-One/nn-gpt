#!/usr/bin/env python3
"""
Visualize Fine-Tuning Cycle Trends

Generate plots showing improvement across cycles:
- Success rate trend
- Best/average accuracy trend
- Training data size growth
- Novel models discovered per cycle
- Cumulative improvement metrics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib
from ab.nn.util.Const import out_dir

matplotlib.use('Agg')  # Non-interactive backend
import numpy as np


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load cycle results."""
    if not results_file.exists():
        print(f"[ERROR] Results file not found: {results_file}")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: direct list or dict with "cycles" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "cycles" in data:
        # Convert summary format to cycle results format
        # Need to get results_dir to load novel counts from individual files
        results_dir = results_file.parent
        
        cycles = []
        for cycle_summary in data["cycles"]:
            cycle_num = cycle_summary["cycle"]
            
            # Try to load novel count from individual cycle_results.json
            cycle_results_file = results_dir / f"cycle_{cycle_num}" / "cycle_results.json"
            novel_count = 0
            total_generated = 0
            if cycle_results_file.exists():
                try:
                    with open(cycle_results_file, "r") as f2:
                        cycle_data = json.load(f2)
                        if "generation" in cycle_data:
                            novel_count = cycle_data["generation"].get("novel", 0)
                            total_generated = cycle_data["generation"].get("total_generated", 0)
                except:
                    pass
            
            # Handle different data formats
            # Format 1: cycle_analysis.json has training_data nested
            if "training_data" in cycle_summary:
                training_data = cycle_summary["training_data"]
                total_examples = training_data.get("total_examples", 0)
                new_examples_added = training_data.get("new_examples_added", 0)
            # Format 2: Direct keys (all_cycles_results.json summary format)
            else:
                total_examples = cycle_summary.get("training_examples", cycle_summary.get("total_examples", 0))
                new_examples_added = cycle_summary.get("new_examples_added", 0)
            
            # Handle evaluation data (may be nested or at top level)
            eval_data = cycle_summary.get("evaluation", {})
            if not eval_data:
                eval_data = cycle_summary
            
            # Normalize percentages (convert 0-100 to 0-1 if needed)
            def normalize_percent(val):
                if isinstance(val, (int, float)):
                    return val / 100.0 if val > 1.0 else val
                return 0.0
            
            success_rate = normalize_percent(eval_data.get("success_rate", 0))
            best_accuracy = normalize_percent(eval_data.get("best_accuracy", 0))
            avg_accuracy = normalize_percent(eval_data.get("avg_accuracy", 0))
            models_trained = eval_data.get("models_trained", cycle_summary.get("models_trained", 0))
            
            # Handle generation data
            gen_data = cycle_summary.get("generation", {})
            total_gen = total_generated if total_generated > 0 else gen_data.get("total_generated", cycle_summary.get("total_generated", 0))
            
            # Handle selection/novel data
            sel_data = cycle_summary.get("selection", {})
            novel_models = novel_count if novel_count > 0 else sel_data.get("novel_models", cycle_summary.get("novel", 0))
            
            cycle_result = {
                "cycle": cycle_num,
                "success": True,
                "evaluation": {
                    "success_rate": success_rate,
                    "best_accuracy": best_accuracy,
                    "avg_accuracy": avg_accuracy,
                    "models_trained": models_trained,
                },
                "training": {
                    "total_examples": total_examples,
                    "new_examples_added": new_examples_added,
                    "training_time_minutes": cycle_summary.get("training_time_minutes", 0),
                },
                "generation": {
                    "total_generated": total_gen,
                    "selected_for_training": new_examples_added,
                    "novel": novel_models,
                },
                "cycle_time_minutes": cycle_summary.get("cycle_time_minutes", 0),
            }
            cycles.append(cycle_result)
        return cycles
    else:
        print(f"[ERROR] Unexpected results format in {results_file}")
        sys.exit(1)


def plot_success_rate(results: List[Dict[str, Any]], output_path: Path):
    """Plot success rate trend across cycles."""
    successful_results = [r for r in results if r.get("success", False)]
    
    cycles = [r["cycle"] for r in successful_results]
    success_rates = [r["evaluation"]["success_rate"] * 100 for r in successful_results]
    
    # Get full cycle range
    min_cycle = min(cycles) if cycles else 1
    max_cycle = max(cycles) if cycles else 15
    all_cycles = list(range(min_cycle, max_cycle + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, success_rates, marker='o', linewidth=2.5, markersize=10, color='#2E86AB')
    plt.xlabel('Cycle', fontsize=13, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    plt.title('Model Generation Success Rate vs Cycle', fontsize=15, fontweight='bold')
    plt.xticks(all_cycles, all_cycles)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_trends(results: List[Dict[str, Any]], output_path: Path):
    """Plot best and average accuracy trends across cycles."""
    successful_results = [r for r in results if r.get("success", False)]
    
    cycles = [r["cycle"] for r in successful_results]
    best_accs = [r["evaluation"]["best_accuracy"] * 100 for r in successful_results]
    avg_accs = [r["evaluation"]["avg_accuracy"] * 100 for r in successful_results]
    
    # Get full cycle range
    min_cycle = min(cycles) if cycles else 1
    max_cycle = max(cycles) if cycles else 15
    all_cycles = list(range(min_cycle, max_cycle + 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(cycles, best_accs, marker='^', linewidth=2.5, markersize=10, 
             color='#06A77D', label='Best Accuracy')
    plt.plot(cycles, avg_accs, marker='s', linewidth=2.5, markersize=10, 
             color='#A23B72', label='Average Accuracy', linestyle='--')
    
    plt.xlabel('Cycle', fontsize=13, fontweight='bold')
    plt.ylabel('First-Epoch Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title('Model Accuracy Trends vs Cycle', fontsize=15, fontweight='bold')
    plt.xticks(all_cycles, all_cycles)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_training_data_growth(results: List[Dict[str, Any]], output_path: Path):
    """Plot training data size growth across cycles."""
    successful_results = [r for r in results if r.get("success", False)]
    
    cycles = [r["cycle"] for r in successful_results]
    total_examples = [r["training"]["total_examples"] for r in successful_results]
    new_examples = [r["training"]["new_examples_added"] for r in successful_results]
    
    # Get full cycle range
    min_cycle = min(cycles) if cycles else 1
    max_cycle = max(cycles) if cycles else 15
    all_cycles = list(range(min_cycle, max_cycle + 1))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative total
    color1 = '#2E86AB'
    ax1.set_xlabel('Cycle', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Training Examples', fontsize=13, fontweight='bold', color=color1)
    line1 = ax1.plot(cycles, total_examples, marker='o', linewidth=2.5, markersize=10, 
                     color=color1, label='Total Examples')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(all_cycles)
    ax1.set_xticklabels(all_cycles)
    ax1.grid(True, alpha=0.3)
    
    # Plot new examples per cycle on right axis
    ax2 = ax1.twinx()
    color2 = '#F18F01'
    ax2.set_ylabel('New Examples Added', fontsize=13, fontweight='bold', color=color2)
    bars = ax2.bar(cycles, new_examples, alpha=0.5, color=color2, label='New Examples')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines = line1 + [bars]
    labels = ['Total Examples', 'New Examples Added']
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.title('Training Data Growth Across Cycles', fontsize=15, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_novel_models(results: List[Dict[str, Any]], output_path: Path):
    """Plot novel models discovered per cycle."""
    successful_results = [r for r in results if r.get("success", False)]
    
    cycles = [r["cycle"] for r in successful_results]
    novel = [r["generation"].get("novel", 0) for r in successful_results]
    selected = [r["generation"]["selected_for_training"] for r in successful_results]
    
    # Get full cycle range
    min_cycle = min(cycles) if cycles else 1
    max_cycle = max(cycles) if cycles else 15
    all_cycles = list(range(min_cycle, max_cycle + 1))
    
    # Create mapping from cycle to index for positioning bars
    cycle_to_index = {c: i for i, c in enumerate(all_cycles)}
    x_positions = [cycle_to_index.get(c, -1) for c in cycles]
    
    # Create arrays for all cycles (0 for missing cycles)
    novel_all = [0] * len(all_cycles)
    selected_all = [0] * len(all_cycles)
    for i, c in enumerate(cycles):
        idx = cycle_to_index[c]
        novel_all[idx] = novel[i]
        selected_all[idx] = selected[i]
    
    x = np.arange(len(all_cycles))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, novel_all, width, label='Novel Models', color='#06A77D', alpha=0.8)
    bars2 = ax.bar(x + width/2, selected_all, width, label='Selected for Training', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Cycle', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title('Novel Models Discovered and Selected Per Cycle', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_cycles)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_metrics(results: List[Dict[str, Any]], output_path: Path):
    """Plot combined success rate and accuracy on dual axes."""
    successful_results = [r for r in results if r.get("success", False)]
    
    cycles = [r["cycle"] for r in successful_results]
    success_rates = [r["evaluation"]["success_rate"] * 100 for r in successful_results]
    best_accs = [r["evaluation"]["best_accuracy"] * 100 for r in successful_results]
    
    # Get full cycle range
    min_cycle = min(cycles) if cycles else 1
    max_cycle = max(cycles) if cycles else 15
    all_cycles = list(range(min_cycle, max_cycle + 1))
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color1 = '#2E86AB'
    ax1.set_xlabel('Cycle', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold', color=color1)
    line1 = ax1.plot(cycles, success_rates, marker='o', linewidth=2.5, markersize=10, 
                     color=color1, label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(all_cycles)
    ax1.set_xticklabels(all_cycles)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = '#06A77D'
    ax2.set_ylabel('Best First-Epoch Accuracy (%)', fontsize=13, fontweight='bold', color=color2)
    line2 = ax2.plot(cycles, best_accs, marker='^', linewidth=2.5, markersize=10, 
                     color=color2, label='Best Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines = line1 + line2
    labels = ['Success Rate', 'Best Accuracy']
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.title('Iterative Fine-Tuning Progress: Success Rate & Best Accuracy', fontsize=15, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_improvement_summary(results: List[Dict[str, Any]], output_path: Path):
    """Plot summary bar chart showing first vs final cycle."""
    successful_results = [r for r in results if r.get("success", False)]
    
    if len(successful_results) < 2:
        print("[SKIP] Need at least 2 cycles for improvement summary")
        return
    
    first_cycle = successful_results[0]
    final_cycle = successful_results[-1]
    
    metrics = ['Success Rate', 'Avg Accuracy', 'Best Accuracy']
    first_values = [
        first_cycle["evaluation"]["success_rate"] * 100,
        first_cycle["evaluation"]["avg_accuracy"] * 100,
        first_cycle["evaluation"]["best_accuracy"] * 100,
    ]
    final_values = [
        final_cycle["evaluation"]["success_rate"] * 100,
        final_cycle["evaluation"]["avg_accuracy"] * 100,
        final_cycle["evaluation"]["best_accuracy"] * 100,
    ]
    improvements = [f - b for f, b in zip(final_values, first_values)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, first_values, width, label=f'Cycle {first_cycle["cycle"]}', 
                   color='#E76F51', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_values, width, label=f'Cycle {final_cycle["cycle"]}', 
                   color='#2A9D8F', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotations
    for i, imp in enumerate(improvements):
        y_pos = max(first_values[i], final_values[i]) + 2
        color = 'green' if imp > 0 else 'red'
        ax.text(i, y_pos, f'{imp:+.1f}%', 
               ha='center', va='bottom', fontweight='bold', color=color, fontsize=12)
    
    ax.set_ylabel('Value (%)', fontsize=13, fontweight='bold')
    ax.set_title('Improvement from First to Final Cycle', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plots for fine-tuning cycles")
    parser.add_argument("--results_dir", type=str, default=out_dir / 'iterative_cycles',
                        help="Directory containing all_cycles_results.json")
    parser.add_argument("--output_dir", type=str, default=out_dir / 'iterative_cycles' / "plots",
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_dir) / "all_cycles_results.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING CYCLE VISUALIZATION PLOTS")
    print("=" * 80 + "\\n")
    
    print("Loading results...")
    if not results_file.exists():
        # Load from individual cycle_results.json files
        print(f"[INFO] all_cycles_results.json not found, loading from individual cycle files...")
        results = []
        for cycle_dir in sorted(Path(args.results_dir).glob("cycle_*")):
            if not cycle_dir.is_dir():
                continue
            try:
                cycle_num = int(cycle_dir.name.split("_")[1])
            except (ValueError, IndexError):
                continue
            
            cycle_results_file = cycle_dir / "cycle_results.json"
            if cycle_results_file.exists():
                try:
                    with open(cycle_results_file, "r") as f:
                        cycle_data = json.load(f)
                        if cycle_data.get("success", False):
                            results.append(cycle_data)
                except:
                    pass
    else:
        results = load_results(results_file)
    
    successful_results = [r for r in results if r.get("success", False)]
    print(f"Found {len(successful_results)} successful cycles\\n")
    
    if not successful_results:
        print("[ERROR] No successful cycles to plot")
        sys.exit(1)
    
    print("Generating plots...")
    plot_success_rate(results, output_dir / "1_success_rate_trend.png")
    plot_accuracy_trends(results, output_dir / "2_accuracy_trends.png")
    plot_training_data_growth(results, output_dir / "3_training_data_growth.png")
    plot_novel_models(results, output_dir / "4_novel_models.png")
    plot_combined_metrics(results, output_dir / "5_combined_metrics.png")
    plot_improvement_summary(results, output_dir / "6_improvement_summary.png")
    
    print("\\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print(f"All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()



