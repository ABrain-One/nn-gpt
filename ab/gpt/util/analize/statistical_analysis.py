#!/usr/bin/env python3
"""
Statistical Analysis of Output Metrics Across Fine-Tuning Cycles

Calculates mean and 95% confidence intervals for each epoch (A0, A1, A2...)
across all probes (B0, B1, B2...) to track fine-tuning performance improvement.

Usage:
    python3 statistical_analysis.py [--base-path PATH] [--format table|csv]
    python3 statistical_analysis.py --compare nngpt2 nngpt3  # Compare directories
"""

import os
import json
import glob
import sys
import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from math import sqrt

class StatisticalAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.metrics_file = os.path.join(base_path, "output_size_metrics.json")
        self.all_files = []
        self.epoch_data = defaultdict(list)
        self._load_data()
    
    def _load_data(self):
        """Load metrics from JSON file and find all output files"""
        self.all_files = {}
        
        # Scan directory for output files
        pattern = f"{self.base_path}/nngpt/llm/epoch/A*/synth_nn/*/full_output.txt"
        for filepath in glob.glob(pattern):
            match = re.search(r'epoch/(A\d+)/synth_nn/(B\d+)/', filepath)
            if match:
                epoch = match.group(1)
                run = match.group(2)
                key = f"epoch_{epoch}_{run}"
                self.all_files[key] = {"filepath": filepath}
        
        # Try to load cached metrics from JSON
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    cached = json.load(f)
                    # Merge with file-based data
                    for key, data in cached.items():
                        if key in self.all_files:
                            self.all_files[key].update(data)
            except:
                pass
    
    def calculate_ci_95(self, values):
        """Calculate 95% confidence interval using t-distribution approximation"""
        # Convert all values to float
        values = [float(v) for v in values if v is not None]
        
        if len(values) < 2:
            return mean(values) if values else 0, 0, 0
        
        m = mean(values)
        n = len(values)
        
        if n == 1:
            return m, m, m
        
        s = stdev(values)
        # Approximate 95% CI using t-distribution (df = n-1)
        # For large n, t ≈ 1.96; for small n, t is larger
        t_value = 1.96 if n > 30 else [0, 12.7, 4.3, 3.2, 2.8, 2.6, 2.4][min(n-1, 6)]
        ci = t_value * s / sqrt(n)
        
        return m, m - ci, m + ci
    
    def extract_metrics_from_file(self, filepath):
        """Extract metrics from output file"""
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return None
        
        # Find sections
        hp_idx = content.find('<hp>')
        sections = {
            'reasoning': content[:hp_idx] if hp_idx > 0 else "",
        }
        
        for tag in ['hp', 'tr', 'nn']:
            start = content.find(f'<{tag}>')
            end = content.find(f'</{tag}>')
            if start > 0 and end > 0:
                sections[tag] = content[start+len(f'<{tag}>'):end]
            else:
                sections[tag] = ""
        
        # Calculate metrics
        total_chars = len(content)
        total_tokens = len(content) // 4
        total_lines = len(content.split('\n'))
        
        reasoning_chars = len(sections['reasoning'])
        code_chars = len(sections['hp']) + len(sections['tr']) + len(sections['nn'])
        
        reasoning_tokens = len(sections['reasoning']) // 4
        code_tokens = code_chars // 4
        
        reasoning_pct = (reasoning_chars / total_chars * 100) if total_chars > 0 else 0
        code_pct = (code_chars / total_chars * 100) if total_chars > 0 else 0
        
        return {
            'total_chars': total_chars,
            'total_tokens': total_tokens,
            'total_lines': total_lines,
            'reasoning_chars': reasoning_chars,
            'reasoning_tokens': reasoning_tokens,
            'reasoning_pct': reasoning_pct,
            'code_chars': code_chars,
            'code_tokens': code_tokens,
            'code_pct': code_pct,
        }
    
    def group_by_epoch(self):
        """Group metrics by epoch"""
        epoch_metrics = defaultdict(lambda: defaultdict(list))
        
        for key, data in self.all_files.items():
            # Extract epoch and run from key
            match = re.search(r'epoch_(A\d+)_(B\d+)', key)
            if not match:
                continue
            
            epoch = match.group(1)
            
            # If we have metrics cached, use them
            if 'total_chars' in data:
                metrics = data
            else:
                # Extract from file
                filepath = data.get('filepath', '')
                metrics = self.extract_metrics_from_file(filepath)
            
            if metrics:
                for metric_name, value in metrics.items():
                    # Skip non-numeric values
                    if isinstance(value, (int, float)):
                        epoch_metrics[epoch][metric_name].append(value)
        
        return epoch_metrics
    
    def calculate_statistics(self):
        """Calculate mean and CI for each epoch"""
        epoch_metrics = self.group_by_epoch()
        
        stats = {}
        for epoch in sorted(epoch_metrics.keys()):
            stats[epoch] = {}
            for metric_name, values in epoch_metrics[epoch].items():
                # Convert to float and filter None values
                float_values = [float(v) for v in values if v is not None]
                if float_values:
                    mean_val, ci_lower, ci_upper = self.calculate_ci_95(float_values)
                    stats[epoch][metric_name] = {
                        'mean': mean_val,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'count': len(float_values),
                        'std': stdev(float_values) if len(float_values) > 1 else 0
                    }
        
        return stats
    
    def print_table(self, stats):
        """Print statistics in table format"""
        if not stats:
            print("[WARN] No data found")
            return
        
        epochs = sorted(stats.keys())
        
        print("\n" + "="*120)
        print("STATISTICAL ANALYSIS: Fine-Tuning Performance Across Epochs")
        print("="*120)
        print(f"\nNote: Values shown as: Mean [95% CI Lower - Upper] (n=samples)")
        print("-"*120)
        
        # Tokens comparison
        print("\n📊 TOTAL TOKENS (Primary Metric)")
        print(f"{'Epoch':<10} {'Mean Tokens':<20} {'CI 95%':<25} {'Samples':<10}")
        print("-"*120)
        
        for epoch in epochs:
            if 'total_tokens' in stats[epoch]:
                s = stats[epoch]['total_tokens']
                print(f"{epoch:<10} {s['mean']:>10.0f} {s['ci_lower']:>20.0f} - {s['ci_upper']:<8.0f} {s['count']:<10}")
        
        # Characters comparison
        print("\n📊 TOTAL CHARACTERS")
        print(f"{'Epoch':<10} {'Mean Chars':<20} {'CI 95%':<25} {'Samples':<10}")
        print("-"*120)
        
        for epoch in epochs:
            if 'total_chars' in stats[epoch]:
                s = stats[epoch]['total_chars']
                print(f"{epoch:<10} {s['mean']:>10.0f} {s['ci_lower']:>20.0f} - {s['ci_upper']:<8.0f} {s['count']:<10}")
        
        # Reasoning percentage comparison
        print("\n📊 REASONING CONTENT (% of total)")
        print(f"{'Epoch':<10} {'Mean %':<20} {'CI 95%':<25} {'Samples':<10}")
        print("-"*120)
        
        for epoch in epochs:
            if 'reasoning_pct' in stats[epoch]:
                s = stats[epoch]['reasoning_pct']
                print(f"{epoch:<10} {s['mean']:>10.1f}% {s['ci_lower']:>19.1f}% - {s['ci_upper']:<8.1f}% {s['count']:<10}")
        
        # Trend analysis
        print("\n" + "="*120)
        print("TREND ANALYSIS")
        print("="*120)
        
        if len(epochs) >= 2:
            first_epoch = epochs[0]
            last_epoch = epochs[-1]
            
            if 'total_tokens' in stats[first_epoch] and 'total_tokens' in stats[last_epoch]:
                first_tokens = stats[first_epoch]['total_tokens']['mean']
                last_tokens = stats[last_epoch]['total_tokens']['mean']
                token_change = ((last_tokens - first_tokens) / first_tokens * 100) if first_tokens > 0 else 0
                
                first_reasoning = stats[first_epoch]['reasoning_pct']['mean']
                last_reasoning = stats[last_epoch]['reasoning_pct']['mean']
                reasoning_change = last_reasoning - first_reasoning
                
                print(f"\n{first_epoch} → {last_epoch}:")
                print(f"  Tokens:    {first_tokens:>10.0f} → {last_tokens:>10.0f} ({token_change:>+6.1f}%)")
                print(f"  Reasoning: {first_reasoning:>10.1f}% → {last_reasoning:>10.1f}% ({reasoning_change:>+6.1f}%)")
                
                if token_change < -15:
                    print("\n  ✓ Output SIZE REDUCED - Solutions working!")
                elif token_change > 15:
                    print("\n  ✗ Output SIZE INCREASED - Fine-tuning needs adjustment")
                else:
                    print(f"\n  ≈ Output SIZE STABLE - Change within margin ({token_change:+.1f}%)")
                
                if reasoning_change < -15:
                    print("  ✓ Reasoning REDUCED - Solution 3 effective!")
                elif reasoning_change > 15:
                    print("  ✗ Reasoning INCREASED - Solution 3 needs adjustment")
                else:
                    print(f"  ≈ Reasoning STABLE - Change within margin ({reasoning_change:+.1f}%)")
        
        print("\n" + "="*120 + "\n")
    
    def print_csv(self, stats):
        """Print statistics in CSV format"""
        if not stats:
            print("[WARN] No data found")
            return
        
        epochs = sorted(stats.keys())
        
        print("Epoch,Total_Tokens_Mean,Total_Tokens_CI_Lower,Total_Tokens_CI_Upper,Total_Tokens_N,Reasoning_Pct_Mean,Reasoning_Pct_CI_Lower,Reasoning_Pct_CI_Upper,Reasoning_Pct_N")
        
        for epoch in epochs:
            row_parts = [epoch]
            
            # Tokens
            if 'total_tokens' in stats[epoch]:
                s = stats[epoch]['total_tokens']
                row_parts.extend([f"{s['mean']:.0f}", f"{s['ci_lower']:.0f}", f"{s['ci_upper']:.0f}", str(s['count'])])
            else:
                row_parts.extend(['', '', '', ''])
            
            # Reasoning %
            if 'reasoning_pct' in stats[epoch]:
                s = stats[epoch]['reasoning_pct']
                row_parts.extend([f"{s['mean']:.1f}", f"{s['ci_lower']:.1f}", f"{s['ci_upper']:.1f}", str(s['count'])])
            else:
                row_parts.extend(['', '', '', ''])
            
            print(','.join(row_parts))

def compare_directories(dir1_path, dir2_path, format_type="table"):
    """Compare statistics from two directories (nngpt2 vs nngpt3)"""
    
    print("\n" + "="*140)
    print("CROSS-DIRECTORY COMPARISON")
    print("="*140)
    
    analyzer1 = StatisticalAnalyzer(dir1_path)
    analyzer2 = StatisticalAnalyzer(dir2_path)
    
    stats1 = analyzer1.calculate_statistics()
    stats2 = analyzer2.calculate_statistics()
    
    dir1_name = dir1_path.split('/')[-3]  # Extract 'nngpt2' or 'nngpt3'
    dir2_name = dir2_path.split('/')[-3]
    
    all_epochs = sorted(set(list(stats1.keys()) + list(stats2.keys())))
    
    print(f"\nComparing: {dir1_name} vs {dir2_name}")
    print(f"{'Epoch':<10} {dir1_name:<30} {dir2_name:<30} {'Difference':<20}")
    print("-"*140)
    
    for epoch in all_epochs:
        if epoch in stats1 and epoch in stats2:
            if 'total_tokens' in stats1[epoch] and 'total_tokens' in stats2[epoch]:
                tokens1 = stats1[epoch]['total_tokens']['mean']
                tokens2 = stats2[epoch]['total_tokens']['mean']
                diff = ((tokens2 - tokens1) / tokens1 * 100) if tokens1 > 0 else 0
                
                print(f"{epoch:<10} {tokens1:>10.0f} tokens {tokens2:>20.0f} tokens {diff:>+18.1f}%")
        elif epoch in stats1:
            tokens1 = stats1[epoch]['total_tokens']['mean'] if 'total_tokens' in stats1[epoch] else 0
            print(f"{epoch:<10} {tokens1:>10.0f} tokens {'N/A':>20} {'N/A':>18}")
        elif epoch in stats2:
            tokens2 = stats2[epoch]['total_tokens']['mean'] if 'total_tokens' in stats2[epoch] else 0
            print(f"{epoch:<10} {'N/A':>30} {tokens2:>10.0f} tokens {'N/A':>18}")
    
    print("="*140 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of output metrics across fine-tuning cycles"
    )
    parser.add_argument('--base-path', help='Base path to output directory')
    parser.add_argument('--format', choices=['table', 'csv'], default='table',
                       help='Output format')
    parser.add_argument('--compare', nargs=2, metavar=('DIR1', 'DIR2'),
                       help='Compare two directories (e.g., nngpt2 nngpt3)')
    
    args = parser.parse_args()
    
    if args.compare:
        # Cross-directory comparison
        dir1 = args.compare[0]
        dir2 = args.compare[1]
        compare_directories(dir1, dir2, args.format)
    else:
        # Single directory analysis
        base_path = args.base_path or "./out"
        
        analyzer = StatisticalAnalyzer(base_path)
        stats = analyzer.calculate_statistics()
        
        if args.format == 'csv':
            analyzer.print_csv(stats)
        else:
            analyzer.print_table(stats)

if __name__ == "__main__":
    main()
