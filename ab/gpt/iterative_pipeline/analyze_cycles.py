#!/usr/bin/env python3
"""
Analyze results from iterative fine-tuning cycles.

Reads all_cycles_results.json and creates:
- cycle_analysis.json: Detailed analysis with new metrics
- cycle_metrics.csv: Table format for easy viewing
- Visualizations of key trends
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from ab.nn.util.Const import out_dir
from scipy import stats


def load_cycle_results(results_dir: str = None) -> List[Dict[str, Any]]:
    """Load cycle results from JSON file."""
    # Try to auto-detect results directory
    if results_dir is None:
        # Check common output directories
        possible_dirs = [
            out_dir / 'iterative_cycles_v2',
            out_dir / 'iterative_cycles',
            out_dir / 'full_pipeline_test'
        ]
        for dir_path in possible_dirs:
            results_path = dir_path / "all_cycles_results.json"
            if results_path.exists():
                results_dir = dir_path
                break
        
        if results_dir is None:
            raise FileNotFoundError(
                f"Results file not found. Checked: {', '.join(possible_dirs)}. "
                f"Please specify --results_dir or ensure results exist."
            )
    
    results_path = Path(results_dir) / "all_cycles_results.json"
    
    if not results_path.exists():
        # Try to load from individual cycle_results.json files
        print(f"[INFO] all_cycles_results.json not found, loading from individual cycle files...")
        results = []
        for cycle_dir in sorted(Path(results_dir).glob("cycle_*")):
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
        
        if results:
            return results, results_dir
        else:
            raise FileNotFoundError(f"Results file not found: {results_path} and no individual cycle files found")
    
    with open(results_path, "r") as f:
        data = json.load(f)
    
    # Handle both formats: direct list or dict with "cycles" key
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict) and "cycles" in data:
        # Convert summary format to cycle results format
        # Load novel counts from individual cycle_results.json files
        results = []
        for cycle_summary in data["cycles"]:
            cycle_num = cycle_summary["cycle"]
            
            # Try to load novel count from individual cycle_results.json
            cycle_results_file = Path(results_dir) / f"cycle_{cycle_num}" / "cycle_results.json"
            novel_count = 0
            total_generated = 0
            if cycle_results_file.exists():
                try:
                    with open(cycle_results_file, "r") as f:
                        cycle_data = json.load(f)
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
                    "novel": novel_count if novel_count > 0 else cycle_summary.get("selection", {}).get("novel_models", cycle_summary.get("novel", 0)),
                },
                "cycle_time_minutes": cycle_summary.get("cycle_time_minutes", 0),
            }
            results.append(cycle_result)
    else:
        raise ValueError(f"Unexpected results format in {results_path}")
    
    return results, results_dir


def compute_percentile_above_threshold(accuracies: List[float], threshold: float) -> float:
    """Compute percentage of models above threshold."""
    if not accuracies:
        return 0.0
    above = sum(1 for acc in accuracies if acc >= threshold)
    return (above / len(accuracies)) * 100


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a list of values.
    
    Args:
        data: List of numeric values
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not data or len(data) < 2:
        mean = data[0] if data else 0.0
        return (mean, mean, mean)
    
    data_array = np.array(data)
    n = len(data_array)
    mean = np.mean(data_array)
    std_err = stats.sem(data_array)  # Standard error of the mean
    
    # Use t-distribution for small samples, normal for large
    if n < 30:
        # t-distribution
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin = t_critical * std_err
    else:
        # Normal approximation
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin = z_critical * std_err
    
    lower = mean - margin
    upper = mean + margin
    
    return (mean, lower, upper)


def analyze_cycles(results: List[Dict[str, Any]], results_dir: Path = out_dir / 'iterative_cycles') -> Dict[str, Any]:
    """Analyze cycles and compute new metrics."""
    
    analysis = {
        "cycles": [],
        "trends": {},
        "summary": {}
    }
    
    for cycle_result in results:
        if not cycle_result.get("success", False):
            cycle_analysis = {
                "cycle": cycle_result["cycle"],
                "success": False,
                "error": cycle_result.get("error", "unknown")
            }
        else:
            eval_result = cycle_result.get("evaluation", {})
            generation = cycle_result.get("generation", {})
            training_data = cycle_result.get("training", {})
            
            # Load detailed accuracies from nneval directory for percentile calculation
            cycle_dir = Path(results_dir) / f"cycle_{cycle_result['cycle']}" / "nneval"
            accuracies = []
            if cycle_dir.exists():
                for model_dir in cycle_dir.glob("gen_*"):
                    result_file = model_dir / "1.json"
                    if result_file.exists():
                        try:
                            data = json.loads(result_file.read_text())
                            if isinstance(data, list) and len(data) > 0:
                                acc = data[0].get("accuracy", data[0].get("acc", None))
                                if acc is not None:
                                    accuracies.append(acc)
                        except:
                            pass
            
            # Compute new metrics
            percent_above_40 = compute_percentile_above_threshold(accuracies, 0.40)
            percent_above_35 = compute_percentile_above_threshold(accuracies, 0.35)
            percent_above_30 = compute_percentile_above_threshold(accuracies, 0.30)
            
            # Compute confidence intervals for accuracy
            if accuracies:
                mean_acc, ci_lower, ci_upper = compute_confidence_interval(accuracies, confidence=0.95)
                mean_acc_pct = mean_acc * 100
                ci_lower_pct = ci_lower * 100
                ci_upper_pct = ci_upper * 100
                ci_margin_pct = (ci_upper_pct - ci_lower_pct) / 2
            else:
                mean_acc_pct = 0.0
                ci_lower_pct = 0.0
                ci_upper_pct = 0.0
                ci_margin_pct = 0.0
            
            models_trained = eval_result.get("models_trained", 0)
            selected_for_training = generation.get("selected_for_training", 0)
            
            # Check if fallback was used (selected < expected based on threshold)
            fallback_used = selected_for_training < len([a for a in accuracies if a >= 0.40]) if accuracies else False
            
            cycle_analysis = {
                "cycle": cycle_result["cycle"],
                "success": True,
                "generation": {
                    "total_generated": generation.get("total_generated", 0),
                    "success_rate": eval_result.get("success_rate", 0.0) * 100
                },
                "evaluation": {
                    "models_trained": models_trained,
                    "best_accuracy": eval_result.get("best_accuracy", 0.0) * 100,
                    "avg_accuracy": eval_result.get("avg_accuracy", 0.0) * 100,
                    "median_accuracy": np.median(accuracies) * 100 if accuracies else 0.0,
                    "confidence_interval": {
                        "mean": round(mean_acc_pct, 2),
                        "lower_95": round(ci_lower_pct, 2),
                        "upper_95": round(ci_upper_pct, 2),
                        "margin": round(ci_margin_pct, 2),
                        "sample_size": len(accuracies)
                    }
                },
                "quality_metrics": {
                    "percent_above_40": round(percent_above_40, 2),
                    "percent_above_35": round(percent_above_35, 2),
                    "percent_above_30": round(percent_above_30, 2),
                },
                "selection": {
                    "selected_for_training": selected_for_training,
                    "novel_models": generation.get("novel", 0),
                    "fallback_used": fallback_used,
                },
                "training_data": {
                    "new_examples_added": training_data.get("new_examples_added", 0),
                    "total_examples": training_data.get("total_examples", 0),
                }
            }
        
        analysis["cycles"].append(cycle_analysis)
    
    # Compute trends
    successful_cycles = [c for c in analysis["cycles"] if c["success"]]
    
    if len(successful_cycles) >= 2:
        first = successful_cycles[0]
        last = successful_cycles[-1]
        
        analysis["trends"] = {
            "generation_success_rate": {
                "first": first["generation"]["success_rate"],
                "last": last["generation"]["success_rate"],
                "delta": round(last["generation"]["success_rate"] - first["generation"]["success_rate"], 2)
            },
            "best_accuracy": {
                "first": first["evaluation"]["best_accuracy"],
                "last": last["evaluation"]["best_accuracy"],
                "delta": round(last["evaluation"]["best_accuracy"] - first["evaluation"]["best_accuracy"], 2)
            },
            "percent_above_40": {
                "first": first["quality_metrics"]["percent_above_40"],
                "last": last["quality_metrics"]["percent_above_40"],
                "delta": round(last["quality_metrics"]["percent_above_40"] - first["quality_metrics"]["percent_above_40"], 2)
            },
            "selected_for_training": {
                "first": first["selection"]["selected_for_training"],
                "last": last["selection"]["selected_for_training"],
                "delta": last["selection"]["selected_for_training"] - first["selection"]["selected_for_training"]
            }
        }
    
    # Overall summary
    if successful_cycles:
        # Collect all accuracies across cycles for aggregate CI
        all_accuracies = []
        for cycle in successful_cycles:
            cycle_dir = Path(results_dir) / f"cycle_{cycle['cycle']}" / "nneval"
            if cycle_dir.exists():
                for model_dir in cycle_dir.glob("gen_*"):
                    result_file = model_dir / "1.json"
                    if result_file.exists():
                        try:
                            data = json.loads(result_file.read_text())
                            if isinstance(data, list) and len(data) > 0:
                                acc = data[0].get("accuracy", data[0].get("acc", None))
                                if acc is not None:
                                    all_accuracies.append(acc)
                        except:
                            pass
        
        # Compute aggregate confidence intervals
        if all_accuracies:
            agg_mean, agg_lower, agg_upper = compute_confidence_interval(all_accuracies, confidence=0.95)
            agg_mean_pct = agg_mean * 100
            agg_lower_pct = agg_lower * 100
            agg_upper_pct = agg_upper * 100
            agg_margin_pct = (agg_upper_pct - agg_lower_pct) / 2
        else:
            agg_mean_pct = 0.0
            agg_lower_pct = 0.0
            agg_upper_pct = 0.0
            agg_margin_pct = 0.0
        
        # Compute CI for success rates across cycles
        success_rates = [c["generation"]["success_rate"] for c in successful_cycles]
        sr_mean, sr_lower, sr_upper = compute_confidence_interval(success_rates, confidence=0.95)
        
        # Compute CI for best accuracies across cycles
        best_accs = [c["evaluation"]["best_accuracy"] for c in successful_cycles]
        ba_mean, ba_lower, ba_upper = compute_confidence_interval(best_accs, confidence=0.95)
        
        analysis["summary"] = {
            "total_cycles": len(successful_cycles),
            "avg_generation_success_rate": round(np.mean(success_rates), 2),
            "avg_best_accuracy": round(np.mean(best_accs), 2),
            "avg_percent_above_40": round(np.mean([c["quality_metrics"]["percent_above_40"] for c in successful_cycles]), 2),
            "total_novel_models": sum(c["selection"]["novel_models"] for c in successful_cycles),
            "total_training_examples_added": sum(c["training_data"]["new_examples_added"] for c in successful_cycles),
            "final_training_set_size": successful_cycles[-1]["training_data"]["total_examples"] if successful_cycles else 0,
            "aggregate_confidence_intervals": {
                "average_accuracy": {
                    "mean": round(float(agg_mean_pct), 2),
                    "lower_95": round(float(agg_lower_pct), 2),
                    "upper_95": round(float(agg_upper_pct), 2),
                    "margin": round(float(agg_margin_pct), 2),
                    "sample_size": len(all_accuracies)
                },
                "success_rate": {
                    "mean": round(float(sr_mean), 2),
                    "lower_95": round(float(sr_lower), 2),
                    "upper_95": round(float(sr_upper), 2),
                    "margin": round(float((sr_upper - sr_lower) / 2), 2),
                    "sample_size": len(success_rates)
                },
                "best_accuracy": {
                    "mean": round(float(ba_mean), 2),
                    "lower_95": round(float(ba_lower), 2),
                    "upper_95": round(float(ba_upper), 2),
                    "margin": round(float((ba_upper - ba_lower) / 2), 2),
                    "sample_size": len(best_accs)
                }
            }
        }
    
    return analysis


def save_analysis(analysis: Dict[str, Any], output_dir: Path = out_dir / 'iterative_cycles'):
    """Save analysis to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "cycle_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis, indent=2, fp=f)
    print(f"Saved analysis to: {json_path}")
    
    # Save CSV
    csv_path = output_dir / "cycle_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Cycle", "Success", "Generated", "Gen Success %", "Models Trained",
            "Best Acc %", "Avg Acc %", "Median Acc %",
            "CI Lower 95%", "CI Upper 95%", "CI Margin",
            "%>=40%", "%>=35%", "%>=30%",
            "Selected", "Novel", "Fallback Used",
            "New Examples", "Total Examples"
        ])
        
        # Data rows
        for cycle in analysis["cycles"]:
            if cycle["success"]:
                ci = cycle["evaluation"].get("confidence_interval", {})
                writer.writerow([
                    cycle["cycle"],
                    "Yes",
                    cycle["generation"]["total_generated"],
                    f"{cycle['generation']['success_rate']:.1f}",
                    cycle["evaluation"]["models_trained"],
                    f"{cycle['evaluation']['best_accuracy']:.2f}",
                    f"{cycle['evaluation']['avg_accuracy']:.2f}",
                    f"{cycle['evaluation']['median_accuracy']:.2f}",
                    f"{ci.get('lower_95', 0):.2f}",
                    f"{ci.get('upper_95', 0):.2f}",
                    f"{ci.get('margin', 0):.2f}",
                    f"{cycle['quality_metrics']['percent_above_40']:.2f}",
                    f"{cycle['quality_metrics']['percent_above_35']:.2f}",
                    f"{cycle['quality_metrics']['percent_above_30']:.2f}",
                    cycle["selection"]["selected_for_training"],
                    cycle["selection"]["novel_models"],
                    "Yes" if cycle["selection"]["fallback_used"] else "No",
                    cycle["training_data"]["new_examples_added"],
                    cycle["training_data"]["total_examples"]
                ])
            else:
                writer.writerow([cycle["cycle"], "No", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
    
    print(f"Saved CSV to: {csv_path}")


def plot_confidence_intervals(analysis: Dict[str, Any], results_dir: Path = out_dir / 'iterative_cycles'):
    """Create a separate visualization for confidence intervals."""
    successful_cycles = [c for c in analysis["cycles"] if c["success"]]
    
    if not successful_cycles:
        return
    
    cycles = [c["cycle"] for c in successful_cycles]
    
    # Get full cycle range
    min_cycle = min(cycles) if cycles else 1
    max_cycle = max(cycles) if cycles else 15
    all_cycles = list(range(min_cycle, max_cycle + 1))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Confidence Intervals Analysis", fontsize=16, fontweight="bold")
    
    # Plot 1: Average Accuracy with CI bands
    ax = axes[0]
    avg_accs = [c["evaluation"]["avg_accuracy"] for c in successful_cycles]
    ci_lowers = []
    ci_uppers = []
    ci_margins = []
    
    for c in successful_cycles:
        ci = c["evaluation"].get("confidence_interval", {})
        ci_lowers.append(ci.get("lower_95", c["evaluation"]["avg_accuracy"]))
        ci_uppers.append(ci.get("upper_95", c["evaluation"]["avg_accuracy"]))
        ci_margins.append(ci.get("margin", 0))
    
    # Plot mean line
    ax.plot(cycles, avg_accs, marker='o', linewidth=2.5, markersize=8, 
            color='#2E86AB', label='Mean Accuracy')
    
    # Plot CI bands (shaded area)
    ax.fill_between(cycles, ci_lowers, ci_uppers, alpha=0.3, color='#2E86AB', 
                    label='95% Confidence Interval')
    
    # Plot CI bounds
    ax.plot(cycles, ci_lowers, '--', linewidth=1.5, color='#06A77D', alpha=0.7, label='CI Lower')
    ax.plot(cycles, ci_uppers, '--', linewidth=1.5, color='#F18F01', alpha=0.7, label='CI Upper')
    
    ax.set_xlabel("Cycle", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title("Average Accuracy with 95% Confidence Intervals", fontsize=14, fontweight='bold')
    ax.set_xticks(all_cycles)
    ax.set_xticklabels(all_cycles)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: CI Margins (uncertainty over cycles)
    ax = axes[1]
    ax.plot(cycles, ci_margins, marker='s', linewidth=2.5, markersize=8, 
            color='#A23B72', label='CI Margin (±%)')
    ax.set_xlabel("Cycle", fontsize=12, fontweight='bold')
    ax.set_ylabel("CI Margin (%)", fontsize=12, fontweight='bold')
    ax.set_title("Confidence Interval Margin (Uncertainty) Across Cycles", fontsize=14, fontweight='bold')
    ax.set_xticks(all_cycles)
    ax.set_xticklabels(all_cycles)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save CI plot
    ci_output_path = Path(results_dir) / "confidence_intervals.png"
    plt.savefig(ci_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confidence intervals plot to: {ci_output_path}")
    
    plt.close()


def plot_trends(analysis: Dict[str, Any], results_dir: Path = out_dir / 'iterative_cycles'):
    """Create visualizations of key trends."""
    successful_cycles = [c for c in analysis["cycles"] if c["success"]]
    
    if not successful_cycles:
        print("No successful cycles to plot")
        return
    
    cycles = [c["cycle"] for c in successful_cycles]
    
    # Get full cycle range
    min_cycle = min(cycles) if cycles else 1
    max_cycle = max(cycles) if cycles else 15
    all_cycles = list(range(min_cycle, max_cycle + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Iterative Fine-Tuning Cycle Analysis", fontsize=16, fontweight="bold")
    
    # Plot 1: Accuracy Trends
    ax = axes[0, 0]
    best_accs = [c["evaluation"]["best_accuracy"] for c in successful_cycles]
    avg_accs = [c["evaluation"]["avg_accuracy"] for c in successful_cycles]
    median_accs = [c["evaluation"]["median_accuracy"] for c in successful_cycles]
    
    ax.plot(cycles, best_accs, marker='o', label="Best", linewidth=2)
    ax.plot(cycles, avg_accs, marker='s', label="Average", linewidth=2)
    ax.plot(cycles, median_accs, marker='^', label="Median", linewidth=2)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("First-Epoch Accuracy Trends")
    ax.set_xticks(all_cycles)
    ax.set_xticklabels(all_cycles)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Quality Distribution (%>=40%)
    ax = axes[0, 1]
    pct_40 = [c["quality_metrics"]["percent_above_40"] for c in successful_cycles]
    pct_35 = [c["quality_metrics"]["percent_above_35"] for c in successful_cycles]
    pct_30 = [c["quality_metrics"]["percent_above_30"] for c in successful_cycles]
    
    ax.plot(cycles, pct_40, marker='o', label=">=40%", linewidth=2, color='green')
    ax.plot(cycles, pct_35, marker='s', label=">=35%", linewidth=2, color='orange')
    ax.plot(cycles, pct_30, marker='^', label=">=30%", linewidth=2, color='blue')
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Percentage of Models (%)")
    ax.set_title("Quality Distribution by Accuracy Threshold")
    ax.set_xticks(all_cycles)
    ax.set_xticklabels(all_cycles)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Selection and Training Data
    ax = axes[1, 0]
    selected = [c["selection"]["selected_for_training"] for c in successful_cycles]
    novel = [c["selection"]["novel_models"] for c in successful_cycles]
    
    # Create mapping from cycle to index for positioning bars
    cycle_to_index = {c: i for i, c in enumerate(all_cycles)}
    
    # Create arrays for all cycles (0 for missing cycles)
    selected_all = [0] * len(all_cycles)
    novel_all = [0] * len(all_cycles)
    for i, c in enumerate(cycles):
        idx = cycle_to_index[c]
        selected_all[idx] = selected[i]
        novel_all[idx] = novel[i]
    
    x = np.arange(len(all_cycles))
    width = 0.35
    ax.bar(x - width/2, selected_all, width, label="Selected for Training", alpha=0.8)
    ax.bar(x + width/2, novel_all, width, label="Novel Models", alpha=0.8)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Count")
    ax.set_title("Model Selection and Novelty")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cycles)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Training Data Growth (with dual y-axes)
    ax = axes[1, 1]
    new_examples = [c["training_data"]["new_examples_added"] for c in successful_cycles]
    total_examples = [c["training_data"]["total_examples"] for c in successful_cycles]
    
    # Use dual y-axes for better visualization
    # Left axis: New Examples Added (smaller scale, 0-50)
    color1 = '#2E86AB'
    ax.set_xlabel("Cycle")
    ax.set_ylabel("New Examples Added", color=color1, fontweight='bold')
    line1 = ax.plot(cycles, new_examples, marker='o', label="New Examples Added", 
                    linewidth=2.5, color=color1, markersize=8)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_ylim(0, max(new_examples) * 1.1)  # Set appropriate scale for new examples
    ax.set_xticks(all_cycles)
    ax.set_xticklabels(all_cycles)
    ax.grid(True, alpha=0.3)
    
    # Right axis: Total Examples (larger scale, ~1600-2200)
    ax2 = ax.twinx()
    color2 = '#F18F01'
    ax2.set_ylabel("Total Examples", color=color2, fontweight='bold')
    line2 = ax2.plot(cycles, total_examples, marker='s', label="Total Examples", 
                     linewidth=2.5, color=color2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(min(total_examples) * 0.98, max(total_examples) * 1.02)  # Set appropriate scale for total
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_title("Training Data Growth", fontweight='bold')
    ax.set_title("Training Data Growth")
    
    plt.tight_layout()
    
    # Save main plot
    output_path = Path(results_dir) / "cycle_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plots to: {output_path}")
    
    plt.close()
    
    # Create separate confidence interval visualization
    plot_confidence_intervals(analysis, results_dir)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze iterative fine-tuning cycle results")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory (auto-detected if not specified)")
    args = parser.parse_args()
    
    print("="*80)
    print("ITERATIVE FINE-TUNING CYCLE ANALYSIS")
    print("="*80)
    print()
    
    # Load results
    results, results_dir = load_cycle_results(args.results_dir)
    print(f"Loaded {len(results)} cycle results from: {results_dir}")
    
    # Analyze
    analysis = analyze_cycles(results, results_dir)
    
    # Save
    save_analysis(analysis, results_dir)
    
    # Plot
    plot_trends(analysis, results_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    summary = analysis.get("summary", {})
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    if analysis.get("trends"):
        print("\n" + "="*80)
        print("TRENDS (First -> Last Cycle)")
        print("="*80)
        for metric, values in analysis["trends"].items():
            print(f"{metric}:")
            print(f"  First: {values['first']:.2f}")
            print(f"  Last: {values['last']:.2f}")
            print(f"  Delta: {values['delta']:+.2f}")


if __name__ == "__main__":
    main()

