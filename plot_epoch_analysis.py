#!/usr/bin/env python3
"""
Plot epoch analysis graphs - Creates publication-ready visualizations from epoch comparison data

Usage:
    python3 plot_epoch_analysis.py                    # Auto-detect all available epochs
    python3 plot_epoch_analysis.py --epochs 0 1 2 5  # Analyze specific epochs
    
    --epochs EPOCH_NUMS    List of epochs to analyze (e.g., --epochs 0 1 2 5 10)
                          If not specified, auto-detects all available epochs

Generated graphs:
    - tokens_trend_A{START}-A{END}.png       (Token count trend across epochs)
    - baseline_vs_latest_A{START}-A{END}.png (Comparison: first epoch vs last epoch)
    - reasoning_reduction_A{START}-A{END}.png (Reasoning percentage trend)
    - output_size_analysis_A{START}-A{END}.png (Combined 4-panel analysis)

Notes:
    - Passes individual epoch numbers to analyze_output_sizes.py
    - Gets detailed per-run + aggregate results for each epoch
    - Plots all epochs provided or auto-detected
"""

import subprocess
import re
import argparse
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

class EpochDataExtractor:
    def __init__(self, base_path=None):
        self.epochs = []
        self.epoch_data = {}
        # If base_path not provided, use relative path from script location
        if base_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(script_dir, "out")
        self.base_path = base_path
    
    def detect_all_epochs(self) -> list:
        """Automatically detect all available epochs and return sorted list"""
        try:
            pattern = f"{self.base_path}/nngpt/llm/epoch/A*/synth_nn/*/full_output.txt"
            files = glob.glob(pattern)
            
            if not files:
                print("[WARN] No epoch files found, using default epochs 0-26")
                return list(range(0, 27))
            
            # Extract epoch numbers from file paths
            epochs_found = set()
            for filepath in files:
                match = re.search(r'/epoch/A(\d+)/', filepath)
                if match:
                    epochs_found.add(int(match.group(1)))
            
            if epochs_found:
                epochs_sorted = sorted(list(epochs_found))
                latest = max(epochs_sorted)
                print(f"[INFO] Auto-detected {len(epochs_sorted)} epochs: A{min(epochs_sorted)}-A{latest}")
                return epochs_sorted
            else:
                print("[WARN] Could not detect epochs, using default epochs 0-26")
                return list(range(0, 27))
        except Exception as e:
            print(f"[WARN] Error detecting epochs: {e}, using default epochs 0-26")
            return list(range(0, 27))
    
    def run_analysis(self, epoch_list: list) -> str:
        """Run analyze_output_sizes.py with individual epoch numbers"""
        try:
            cmd = [
                'python3',
                'analyze_output_sizes.py',
                '--compare-epochs-detailed'
            ]
            # Add each epoch as a separate argument
            cmd.extend([str(e) for e in epoch_list])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[ERROR] Analysis failed: {result.stderr}")
                sys.exit(1)
            return result.stdout
        except Exception as e:
            print(f"[ERROR] Could not run analysis: {e}")
            sys.exit(1)
    
    def parse_output(self, output: str) -> bool:
        """Parse the aggregate comparison table from output"""
        lines = output.split('\n')
        
        in_aggregate = False
        for i, line in enumerate(lines):
            # Find the aggregate comparison section
            if 'AGGREGATE COMPARISON' in line:
                in_aggregate = True
                continue
            
            if in_aggregate and 'Epoch' in line and 'Runs' in line and 'Avg Chars' in line:
                # Found the header row, parse data rows after this
                for j in range(i + 2, len(lines)):
                    data_line = lines[j]
                    
                    # Stop at separator or empty line
                    if not data_line.strip() or '---' in data_line or '===' in data_line:
                        break
                    
                    # Parse epoch data line
                    # Format: A0         5        8,859              8,859              219         48.2%       51.8%
                    match = re.match(
                        r'A(\d+)\s+(\d+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)%\s+([\d.]+)%',
                        data_line.strip()
                    )
                    
                    if match:
                        epoch_num = int(match.group(1))
                        num_runs = int(match.group(2))
                        avg_chars = int(match.group(3).replace(',', ''))
                        avg_tokens = int(match.group(4).replace(',', ''))
                        avg_lines = int(match.group(5).replace(',', ''))
                        reasoning_pct = float(match.group(6))
                        code_pct = float(match.group(7))
                        
                        self.epochs.append(epoch_num)
                        self.epoch_data[epoch_num] = {
                            'num_runs': num_runs,
                            'avg_chars': avg_chars,
                            'avg_tokens': avg_tokens,
                            'avg_lines': avg_lines,
                            'reasoning_pct': reasoning_pct,
                            'code_pct': code_pct,
                        }
        
        if not self.epochs:
            print("[ERROR] Could not parse epoch data from output")
            return False
        
        print(f"[INFO] Extracted data for {len(self.epochs)} epochs: A{self.epochs[0]}-A{self.epochs[-1]}")
        return True
    
    def plot_tokens_trend(self, start_epoch: int, end_epoch: int):
        """Plot token count trend across epochs"""
        if not self.epochs:
            return
        
        tokens = [self.epoch_data[e]['avg_tokens'] for e in self.epochs]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot line
        ax.plot(self.epochs, tokens, marker='o', linewidth=2.5, markersize=6, color='#2E86AB', label='Avg Tokens')
        
        # Add reduction percentage annotation
        if len(self.epochs) >= 2:
            first_tokens = tokens[0]
            last_tokens = tokens[-1]
            reduction = ((first_tokens - last_tokens) / first_tokens * 100)
            
            ax.annotate(
                f'{reduction:.1f}% reduction',
                xy=(self.epochs[-1], last_tokens),
                xytext=(self.epochs[-1] - 3, last_tokens + 500),
                fontsize=11,
                color='#06A77D',
                weight='bold',
                arrowprops=dict(arrowstyle='->', color='#06A77D', lw=1.5)
            )
        
        ax.fill_between(self.epochs, tokens, alpha=0.2, color='#2E86AB')
        
        ax.set_xlabel('Epoch', fontsize=12, weight='bold')
        ax.set_ylabel('Average Token Count', fontsize=12, weight='bold')
        ax.set_title(f'Token Count Trend Across Epochs (A{start_epoch}-A{end_epoch})', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        filename = f"tokens_trend_A{start_epoch}-A{end_epoch}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def plot_baseline_vs_latest(self, start_epoch: int, end_epoch: int):
        """Plot comparison between baseline (first epoch) and latest (last epoch)"""
        if len(self.epochs) < 2:
            return
        
        baseline_epoch = self.epochs[0]
        latest_epoch = self.epochs[-1]
        
        baseline_data = self.epoch_data[baseline_epoch]
        latest_data = self.epoch_data[latest_epoch]
        
        metrics = ['Tokens', 'Chars', 'Lines']
        baseline_values = [
            baseline_data['avg_tokens'],
            baseline_data['avg_chars'],
            baseline_data['avg_lines']
        ]
        latest_values = [
            latest_data['avg_tokens'],
            latest_data['avg_chars'],
            latest_data['avg_lines']
        ]
        
        # Calculate reductions
        reductions = [
            ((baseline_values[i] - latest_values[i]) / baseline_values[i] * 100)
            for i in range(len(metrics))
        ]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(metrics))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], baseline_values, width, label=f'A{baseline_epoch} (Baseline)', 
                       color='#E63946', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], latest_values, width, label=f'A{latest_epoch} (Latest)',
                       color='#06A77D', alpha=0.8)
        
        # Add value labels and reduction percentage
        for i, (bar1, bar2, reduction) in enumerate(zip(bars1, bars2, reductions)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{int(height1):,}', ha='center', va='bottom', fontsize=10, weight='bold')
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{int(height2):,}', ha='center', va='bottom', fontsize=10, weight='bold')
            
            # Add reduction percentage above
            mid_x = (bar1.get_x() + bar1.get_width()/2. + bar2.get_x() + bar2.get_width()/2.) / 2
            max_height = max(height1, height2)
            reduction_text = f'↓ {reduction:.1f}%' if reduction > 0 else f'↑ {abs(reduction):.1f}%'
            reduction_color = '#06A77D' if reduction > 0 else '#E63946'
            
            ax.text(mid_x, max_height + 200, reduction_text, ha='center', fontsize=11,
                   weight='bold', color=reduction_color)
        
        ax.set_ylabel('Value', fontsize=12, weight='bold')
        ax.set_title(f'Baseline vs Latest: A{baseline_epoch} vs A{latest_epoch}', fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        filename = f"baseline_vs_latest_A{start_epoch}-A{end_epoch}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def plot_reasoning_reduction(self, start_epoch: int, end_epoch: int):
        """Plot reasoning percentage reduction across epochs"""
        if not self.epochs:
            return
        
        reasoning_pcts = [self.epoch_data[e]['reasoning_pct'] for e in self.epochs]
        code_pcts = [self.epoch_data[e]['code_pct'] for e in self.epochs]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot both reasoning and code as stacked area
        ax.fill_between(self.epochs, 0, reasoning_pcts, alpha=0.6, color='#F77F00', label='Reasoning %')
        ax.fill_between(self.epochs, reasoning_pcts, 100, alpha=0.6, color='#06A77D', label='Code %')
        
        # Add lines
        ax.plot(self.epochs, reasoning_pcts, marker='o', linewidth=2, markersize=5, color='#E63946', label='Reasoning Trend')
        ax.plot(self.epochs, code_pcts, marker='s', linewidth=2, markersize=5, color='#2E86AB', label='Code Trend')
        
        # Add reduction annotation
        if len(self.epochs) >= 2:
            first_reasoning = reasoning_pcts[0]
            last_reasoning = reasoning_pcts[-1]
            reasoning_reduction = first_reasoning - last_reasoning
            
            ax.annotate(
                f'↓ {reasoning_reduction:.1f}%',
                xy=(self.epochs[-1], last_reasoning),
                xytext=(self.epochs[-1] - 3, last_reasoning - 5),
                fontsize=11,
                color='#E63946',
                weight='bold',
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5)
            )
        
        ax.set_xlabel('Epoch', fontsize=12, weight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, weight='bold')
        ax.set_title(f'Reasoning vs Code Distribution (A{start_epoch}-A{end_epoch})', fontsize=14, weight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(fontsize=10, loc='center right')
        
        filename = f"reasoning_reduction_A{start_epoch}-A{end_epoch}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def plot_combined_analysis(self, start_epoch: int, end_epoch: int):
        """Plot 4-panel combined analysis"""
        if not self.epochs:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Complete Output Analysis: A{start_epoch}-A{end_epoch}', fontsize=16, weight='bold', y=0.995)
        
        # Panel 1: Token trend
        tokens = [self.epoch_data[e]['avg_tokens'] for e in self.epochs]
        axes[0, 0].plot(self.epochs, tokens, marker='o', linewidth=2.5, markersize=5, color='#2E86AB')
        axes[0, 0].fill_between(self.epochs, tokens, alpha=0.2, color='#2E86AB')
        axes[0, 0].set_title('Token Count Trend', fontsize=12, weight='bold')
        axes[0, 0].set_ylabel('Tokens', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Panel 2: Character trend
        chars = [self.epoch_data[e]['avg_chars'] for e in self.epochs]
        axes[0, 1].plot(self.epochs, chars, marker='s', linewidth=2.5, markersize=5, color='#E63946')
        axes[0, 1].fill_between(self.epochs, chars, alpha=0.2, color='#E63946')
        axes[0, 1].set_title('Character Count Trend', fontsize=12, weight='bold')
        axes[0, 1].set_ylabel('Characters', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Panel 3: Reasoning percentage
        reasoning_pcts = [self.epoch_data[e]['reasoning_pct'] for e in self.epochs]
        axes[1, 0].plot(self.epochs, reasoning_pcts, marker='o', linewidth=2.5, markersize=5, color='#F77F00')
        axes[1, 0].fill_between(self.epochs, reasoning_pcts, alpha=0.2, color='#F77F00')
        axes[1, 0].set_title('Reasoning Percentage (Lower is Better)', fontsize=12, weight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=10)
        axes[1, 0].set_ylabel('Reasoning %', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Panel 4: Code percentage
        code_pcts = [self.epoch_data[e]['code_pct'] for e in self.epochs]
        axes[1, 1].plot(self.epochs, code_pcts, marker='s', linewidth=2.5, markersize=5, color='#06A77D')
        axes[1, 1].fill_between(self.epochs, code_pcts, alpha=0.2, color='#06A77D')
        axes[1, 1].set_title('Code Percentage (Higher is Better)', fontsize=12, weight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=10)
        axes[1, 1].set_ylabel('Code %', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        filename = f"output_size_analysis_A{start_epoch}-A{end_epoch}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def generate_all_plots(self, start_epoch: int, end_epoch: int):
        """Generate all plots"""
        print(f"\n[INFO] Generating plots for epochs A{start_epoch}-A{end_epoch}...\n")
        
        self.plot_tokens_trend(start_epoch, end_epoch)
        self.plot_baseline_vs_latest(start_epoch, end_epoch)
        self.plot_reasoning_reduction(start_epoch, end_epoch)
        self.plot_combined_analysis(start_epoch, end_epoch)
        
        print(f"\n[SUCCESS] All graphs generated successfully!")

def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready analysis graphs")
    parser.add_argument('--epochs', type=int, nargs='*', default=None, 
                       help='Specific epochs to analyze (e.g., --epochs 0 1 2 5 10) - auto-detects if not specified')
    parser.add_argument('--start-epoch', type=int, default=None, 
                       help='[Deprecated] Use --epochs instead')
    parser.add_argument('--end-epoch', type=int, default=None, 
                       help='[Deprecated] Use --epochs instead')
    
    args = parser.parse_args()
    
    extractor = EpochDataExtractor()
    
    # Determine which epochs to analyze
    if args.epochs is not None:
        # User specified specific epochs
        epoch_list = sorted(args.epochs)
        print(f"[INFO] Analyzing user-specified epochs: {epoch_list}")
    else:
        # Auto-detect all available epochs
        epoch_list = extractor.detect_all_epochs()
        print(f"[INFO] Analyzing all detected epochs: {epoch_list}")
    
    # Run analysis with all epochs as separate arguments
    print(f"[INFO] Running analysis for epochs A{epoch_list[0]}-A{epoch_list[-1]}...")
    output = extractor.run_analysis(epoch_list)
    
    # Parse output
    print("[INFO] Parsing results...")
    if not extractor.parse_output(output):
        sys.exit(1)
    
    # Generate plots
    extractor.generate_all_plots(epoch_list[0], epoch_list[-1])

if __name__ == "__main__":
    main()
