#!/usr/bin/env python3
"""
Plot Wilson confidence intervals for valid generation rate and accuracy > 40%.
Uses Wilson score interval for binomial proportions.
"""

import json
import math
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from ab.nn.util.Const import out_dir


def wilson_ci(k, n, z=1.96):
    """
    Compute Wilson confidence interval for binomial proportion.
    
    Args:
        k: Number of successes
        n: Total number of trials
        z: Z-score for confidence level (1.96 for 95% CI)
    
    Returns:
        (p, lower_bound, upper_bound): Proportion and CI bounds
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p) + z**2/(4*n**2)) / n) / denom
    return p, centre - margin, centre + margin


def plot_wilson_ci(cycle_analysis_path: Path, output_dir: Path):
    """Generate Wilson CI plots for generation rate and accuracy > 40%."""
    
    # Load data
    with open(cycle_analysis_path) as f:
        data = json.load(f)
    
    cycles, gen_p, gen_low, gen_high = [], [], [], []
    p40, p40_low, p40_high = [], [], []
    
    # Process all cycles
    for cinfo in data["cycles"]:
        if not cinfo.get("success", False):
            continue
            
        c = cinfo["cycle"]
        cycles.append(c)
        
        # Valid generation rate
        n_gen = cinfo["generation"]["total_generated"]
        p_gen = cinfo["generation"]["success_rate"] / 100
        k_gen = round(p_gen * n_gen)
        p, lo, hi = wilson_ci(k_gen, n_gen)
        gen_p.append(p*100)
        gen_low.append(lo*100)
        gen_high.append(hi*100)
        
        # Percent above 40% accuracy
        n_tr = cinfo["evaluation"]["models_trained"]
        p_40 = cinfo["quality_metrics"]["percent_above_40"] / 100
        k_40 = round(p_40 * n_tr)
        p2, lo2, hi2 = wilson_ci(k_40, n_tr)
        p40.append(p2*100)
        p40_low.append(lo2*100)
        p40_high.append(hi2*100)
    
    # Plot 1: Valid Generation Rate
    plt.figure(figsize=(9, 4))
    # Calculate error bars as absolute distances (ensure non-negative)
    yerr_gen_lower = [max(0, gp - gl) for gp, gl in zip(gen_p, gen_low)]
    yerr_gen_upper = [max(0, gh - gp) for gh, gp in zip(gen_high, gen_p)]
    yerr_gen = [yerr_gen_lower, yerr_gen_upper]
    plt.errorbar(cycles, gen_p, yerr=yerr_gen, fmt='o-', capsize=4)
    # Get actual cycle range for title
    min_cycle = min(cycles) if cycles else 1
    max_cycle = max(cycles) if cycles else 1
    plt.xlabel("Cycle")
    plt.ylabel("Valid generation rate (%)")
    plt.title(f"Valid Generation Rate per Cycle ({min_cycle}–{max_cycle}) with 95% CI")
    plt.grid(True)
    plt.xticks(cycles)
    plt.tight_layout()
    
    output_path = output_dir / "wilson_ci_generation_rate.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Plot 2: Accuracy > 40%
    plt.figure(figsize=(9, 4))
    # Calculate error bars as absolute distances (ensure non-negative)
    yerr_40_lower = [max(0, p - l) for p, l in zip(p40, p40_low)]
    yerr_40_upper = [max(0, h - p) for h, p in zip(p40_high, p40)]
    yerr_40 = [yerr_40_lower, yerr_40_upper]
    plt.errorbar(cycles, p40, yerr=yerr_40, fmt='o-', capsize=4)
    plt.xlabel("Cycle")
    plt.ylabel("Models with accuracy > 40% (%)")
    plt.title(f"Accuracy > 40% per Cycle ({min_cycle}–{max_cycle}) with 95% CI")
    plt.grid(True)
    plt.xticks(cycles)
    plt.tight_layout()
    
    output_path = output_dir / "wilson_ci_accuracy_40.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Wilson confidence intervals")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory (auto-detected if not specified)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: results_dir/plots)")
    
    args = parser.parse_args()
    
    # Auto-detect results directory
    if args.results_dir is None:
        possible_dirs = [
            out_dir / 'iterative_cycles_v2',
            out_dir / 'iterative_cycles',
        ]
        for d in possible_dirs:
            if (d / "cycle_analysis.json").exists():
                args.results_dir = str(d)
                break
        
        if args.results_dir is None:
            print("Error: Could not find cycle_analysis.json")
            print("Please specify --results_dir")
            return
    
    results_dir = Path(args.results_dir)
    cycle_analysis_path = results_dir / "cycle_analysis.json"
    
    if not cycle_analysis_path.exists():
        print(f"Error: {cycle_analysis_path} not found")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "plots"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {cycle_analysis_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    plot_wilson_ci(cycle_analysis_path, output_dir)
    
    print("\n" + "="*80)
    print("WILSON CI PLOTTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

