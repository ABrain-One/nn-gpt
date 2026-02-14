#!/usr/bin/env python3
"""
Output Size Analyzer - Track metrics across epochs and runs

Measures:
- Character count
- Token count (using tokenizer)
- Line count
- Reasoning vs Code split

Usage:
    python3 analyze_output_sizes.py [--epoch EPOCH_NUM] [--watch]
    
    --epoch EPOCH_NUM    Analyze specific epoch (default: latest)
    --watch              Monitor new output files as they're created
    --compare            Show comparison with previous runs
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse
import time
import re

# Try to import tokenizer
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("[WARN] Tokenizer not available - will skip token count")

class OutputAnalyzer:
    def __init__(self, base_path=None):
        # If base_path not provided, use relative path from script location
        if base_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(script_dir, "out")
        
        self.base_path = base_path
        self.tokenizer = None
        self.metrics = defaultdict(list)
        self.metrics_file = os.path.join(base_path, "output_size_metrics.json")
        self._init_tokenizer()
        self._load_existing_metrics()
    
    def _init_tokenizer(self):
        """Initialize tokenizer for token counting"""
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
                print("[INFO] Tokenizer loaded successfully")
            except Exception as e:
                print(f"[WARN] Could not load tokenizer: {e}")
                self.tokenizer = None
    
    def _load_existing_metrics(self):
        """Load previously saved metrics"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = defaultdict(list, json.load(f))
                print(f"[INFO] Loaded {len(self.metrics)} previous runs from {self.metrics_file}")
            except Exception as e:
                print(f"[WARN] Could not load metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(dict(self.metrics), f, indent=2)
            print(f"[INFO] Saved metrics to {self.metrics_file}")
        except Exception as e:
            print(f"[WARN] Could not save metrics: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer"""
        if self.tokenizer is None:
            # Fallback: estimate tokens as ~4 chars per token
            return len(text) // 4
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"[WARN] Tokenization failed: {e}")
            return len(text) // 4
    
    def extract_sections(self, text: str) -> dict:
        """Extract reasoning, hyperparameters, transform, and model code sections"""
        sections = {
            "reasoning": "",
            "hyperparameters": "",
            "transform": "",
            "model": "",
        }
        
        # Extract reasoning (everything before <hp>)
        hp_match = re.search(r'<hp>(.*?)</hp>', text, re.DOTALL)
        if hp_match:
            sections["reasoning"] = text[:text.find('<hp>')]
            sections["hyperparameters"] = hp_match.group(1).strip()
        
        # Extract transform
        tr_match = re.search(r'<tr>(.*?)</tr>', text, re.DOTALL)
        if tr_match:
            sections["transform"] = tr_match.group(1).strip()
        
        # Extract model
        nn_match = re.search(r'<nn>(.*?)</nn>', text, re.DOTALL)
        if nn_match:
            sections["model"] = nn_match.group(1).strip()
        
        return sections
    
    def analyze_file(self, filepath: str) -> dict:
        """Analyze a single output file"""
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"[ERROR] Could not read {filepath}: {e}")
            return None
        
        # Calculate basic metrics
        char_count = len(content)
        line_count = len(content.split('\n'))
        token_count = self.count_tokens(content)
        
        # Extract sections
        sections = self.extract_sections(content)
        
        # Calculate per-section metrics
        metrics = {
            "filepath": filepath,
            "timestamp": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
            "total_chars": char_count,
            "total_lines": line_count,
            "total_tokens": token_count,
            "reasoning_chars": len(sections["reasoning"]),
            "reasoning_lines": len(sections["reasoning"].split('\n')),
            "reasoning_tokens": self.count_tokens(sections["reasoning"]),
            "code_chars": len(sections["hyperparameters"]) + len(sections["transform"]) + len(sections["model"]),
            "code_lines": (len(sections["hyperparameters"].split('\n')) + 
                          len(sections["transform"].split('\n')) + 
                          len(sections["model"].split('\n'))),
            "code_tokens": (self.count_tokens(sections["hyperparameters"]) + 
                           self.count_tokens(sections["transform"]) + 
                           self.count_tokens(sections["model"])),
            "hp_chars": len(sections["hyperparameters"]),
            "tr_chars": len(sections["transform"]),
            "nn_chars": len(sections["model"]),
        }
        
        return metrics
    
    def find_output_files(self, epoch: int = None) -> list:
        """Find all full_output.txt files"""
        if epoch is not None:
            pattern = f"{self.base_path}/nngpt/llm/epoch/A{epoch}/synth_nn/*/full_output.txt"
        else:
            pattern = f"{self.base_path}/nngpt/llm/epoch/A*/synth_nn/*/full_output.txt"
        
        files = sorted(glob.glob(pattern))
        return files
    
    def print_metrics(self, metrics: dict, label: str = ""):
        """Pretty print metrics"""
        if metrics is None:
            return
        
        print(f"\n{'='*80}")
        if label:
            print(f"File: {label}")
        print(f"Path: {metrics['filepath']}")
        print(f"Timestamp: {metrics['timestamp']}")
        print(f"{'='*80}")
        print(f"{'METRIC':<30} {'TOTAL':<15} {'REASONING':<15} {'CODE':<15}")
        print(f"{'-'*75}")
        print(f"{'Characters':<30} {metrics['total_chars']:<15} {metrics['reasoning_chars']:<15} {metrics['code_chars']:<15}")
        print(f"{'Lines':<30} {metrics['total_lines']:<15} {metrics['reasoning_lines']:<15} {metrics['code_lines']:<15}")
        print(f"{'Tokens':<30} {metrics['total_tokens']:<15} {metrics['reasoning_tokens']:<15} {metrics['code_tokens']:<15}")
        print(f"{'-'*75}")
        print(f"Section breakdown (chars):")
        print(f"  - Hyperparameters: {metrics['hp_chars']:>10} chars")
        print(f"  - Transform:       {metrics['tr_chars']:>10} chars")
        print(f"  - Model:           {metrics['nn_chars']:>10} chars")
        print(f"{'='*80}\n")
    
    def compare_runs(self, metrics_list: list):
        """Compare multiple runs"""
        if len(metrics_list) < 2:
            return
        
        print(f"\n{'='*100}")
        print("COMPARISON ACROSS RUNS")
        print(f"{'='*100}")
        print(f"{'Run #':<8} {'Chars':<15} {'Lines':<15} {'Tokens':<15} {'Reasoning%':<15}")
        print(f"{'-'*100}")
        
        for idx, m in enumerate(metrics_list, 1):
            reasoning_pct = (m['reasoning_chars'] / m['total_chars'] * 100) if m['total_chars'] > 0 else 0
            print(f"{idx:<8} {m['total_chars']:<15} {m['total_lines']:<15} {m['total_tokens']:<15} {reasoning_pct:<14.1f}%")
        
        # Calculate reduction (positive = reduction, negative = increase)
        if len(metrics_list) >= 2:
            first = metrics_list[0]
            last = metrics_list[-1]
            char_reduction = ((first['total_chars'] - last['total_chars']) / first['total_chars'] * 100) if first['total_chars'] > 0 else 0
            token_reduction = ((first['total_tokens'] - last['total_tokens']) / first['total_tokens'] * 100) if first['total_tokens'] > 0 else 0
            reasoning_reduction = ((first['reasoning_chars'] - last['reasoning_chars']) / first['reasoning_chars'] * 100) if first['reasoning_chars'] > 0 else 0
            
            # Format status
            char_status = "✓ REDUCED" if char_reduction > 0 else "✗ INCREASED"
            token_status = "✓ REDUCED" if token_reduction > 0 else "✗ INCREASED"
            reasoning_status = "✓ REDUCED" if reasoning_reduction > 0 else "✗ INCREASED"
            
            print(f"{'-'*100}")
            print(f"CHANGE (Run 1 → Run {len(metrics_list)}):")
            print(f"  Characters: {char_status:<15} {char_reduction:>+6.1f}% ({first['total_chars']:,} → {last['total_chars']:,} chars)")
            print(f"  Tokens:     {token_status:<15} {token_reduction:>+6.1f}% ({first['total_tokens']:,} → {last['total_tokens']:,} tokens)")
            print(f"  Reasoning:  {reasoning_status:<15} {reasoning_reduction:>+6.1f}% ({first['reasoning_chars']:,} → {last['reasoning_chars']:,} chars)")
            print()
            print(f"  Overall Result: {'✓ Solutions Working' if char_reduction > 15 else '✗ Solutions NOT Working (< 15% reduction)'}")
        
        print(f"{'='*100}\n")
    
    def analyze_epoch(self, epoch: int, compare: bool = False):
        """Analyze all runs in an epoch"""
        files = self.find_output_files(epoch=epoch)
        
        if not files:
            print(f"[WARN] No output files found for epoch A{epoch}")
            return
        
        print(f"\n[INFO] Found {len(files)} output files for epoch A{epoch}")
        
        metrics_list = []
        for idx, filepath in enumerate(files, 1):
            # Extract run identifier (B0, B1, B2, etc.)
            match = re.search(r'/synth_nn/(B\d+)/', filepath)
            run_id = match.group(1) if match else f"Run{idx}"
            
            metrics = self.analyze_file(filepath)
            if metrics:
                metrics_list.append(metrics)
                self.print_metrics(metrics, label=f"Epoch A{epoch} - {run_id}")
                
                # Store in metrics dict
                key = f"epoch_A{epoch}_{run_id}"
                self.metrics[key] = metrics
        
        # Compare across runs
        if compare and len(metrics_list) > 1:
            self.compare_runs(metrics_list)
        
        self._save_metrics()
    
    def watch_new_files(self, check_interval: int = 30):
        """Watch for new output files and analyze them"""
        print("[INFO] Starting file watcher (Ctrl+C to stop)")
        seen_files = set(self.find_output_files())
        
        try:
            while True:
                time.sleep(check_interval)
                current_files = set(self.find_output_files())
                new_files = current_files - seen_files
                
                if new_files:
                    print(f"\n[NEW] Found {len(new_files)} new output file(s)")
                    for filepath in sorted(new_files):
                        match = re.search(r'epoch/(A\d+)/synth_nn/(B\d+)/', filepath)
                        if match:
                            epoch_id = match.group(1)
                            run_id = match.group(2)
                            
                            metrics = self.analyze_file(filepath)
                            if metrics:
                                self.print_metrics(metrics, label=f"{epoch_id} - {run_id}")
                                
                                key = f"{epoch_id}_{run_id}"
                                self.metrics[key] = metrics
                        
                        seen_files.add(filepath)
                    
                    self._save_metrics()
        
        except KeyboardInterrupt:
            print("\n[INFO] Watcher stopped")
    
    def calculate_epoch_mean(self, epoch: int) -> dict:
        """Calculate mean metrics for an epoch across all runs"""
        files = self.find_output_files(epoch=epoch)
        
        if not files:
            print(f"[WARN] No output files found for epoch A{epoch}")
            return None
        
        metrics_list = []
        for filepath in files:
            metrics = self.analyze_file(filepath)
            if metrics:
                metrics_list.append(metrics)
        
        if not metrics_list:
            return None
        
        # Calculate means
        num_runs = len(metrics_list)
        mean_metrics = {
            'epoch': epoch,
            'num_runs': num_runs,
            'mean_chars': sum(m['total_chars'] for m in metrics_list) / num_runs,
            'mean_lines': sum(m['total_lines'] for m in metrics_list) / num_runs,
            'mean_tokens': sum(m['total_tokens'] for m in metrics_list) / num_runs,
            'mean_reasoning_chars': sum(m['reasoning_chars'] for m in metrics_list) / num_runs,
            'mean_code_chars': sum(m['code_chars'] for m in metrics_list) / num_runs,
        }
        
        # Calculate reasoning percentage
        if mean_metrics['mean_chars'] > 0:
            mean_metrics['mean_reasoning_pct'] = (mean_metrics['mean_reasoning_chars'] / mean_metrics['mean_chars'] * 100)
            mean_metrics['mean_code_pct'] = (mean_metrics['mean_code_chars'] / mean_metrics['mean_chars'] * 100)
        else:
            mean_metrics['mean_reasoning_pct'] = 0
            mean_metrics['mean_code_pct'] = 0
        
        return mean_metrics
    
    def compare_epochs(self, epochs: list):
        """Compare mean metrics across multiple epochs"""
        epoch_means = []
        
        for epoch in epochs:
            mean_data = self.calculate_epoch_mean(epoch)
            if mean_data:
                epoch_means.append(mean_data)
        
        if not epoch_means:
            print("[WARN] No data found for comparison")
            return
        
        print(f"\n{'='*120}")
        print("EPOCH COMPARISON - MEAN METRICS ACROSS RUNS")
        print(f"{'='*120}")
        print(f"{'Epoch':<10} {'Runs':<8} {'Avg Chars':<15} {'Avg Tokens':<15} {'Avg Lines':<12} {'Reasoning%':<12}")
        print(f"{'-'*120}")
        
        for data in epoch_means:
            print(f"A{data['epoch']:<9} {data['num_runs']:<8} {data['mean_chars']:>14,.0f} {data['mean_tokens']:>14,.0f} {data['mean_lines']:>11,.0f} {data['mean_reasoning_pct']:>11.1f}%")
        
        # Compare consecutive epochs
        if len(epoch_means) >= 2:
            print(f"{'-'*120}")
            print("TREND ANALYSIS (Changes Between Epochs):")
            print(f"{'-'*120}")
            
            for i in range(len(epoch_means) - 1):
                current = epoch_means[i]
                next_epoch = epoch_means[i + 1]
                
                char_change = ((next_epoch['mean_chars'] - current['mean_chars']) / current['mean_chars'] * 100) if current['mean_chars'] > 0 else 0
                token_change = ((next_epoch['mean_tokens'] - current['mean_tokens']) / current['mean_tokens'] * 100) if current['mean_tokens'] > 0 else 0
                reasoning_change = next_epoch['mean_reasoning_pct'] - current['mean_reasoning_pct']
                
                char_status = "✓ REDUCED" if char_change < 0 else "✗ INCREASED"
                token_status = "✓ REDUCED" if token_change < 0 else "✗ INCREASED"
                reasoning_status = "✓ REDUCED" if reasoning_change < 0 else "✗ INCREASED"
                
                print(f"\nA{current['epoch']} → A{next_epoch['epoch']}:")
                print(f"  Characters: {char_status:<15} {char_change:>+6.1f}% ({current['mean_chars']:>10,.0f} → {next_epoch['mean_chars']:>10,.0f})")
                print(f"  Tokens:     {token_status:<15} {token_change:>+6.1f}% ({current['mean_tokens']:>10,.0f} → {next_epoch['mean_tokens']:>10,.0f})")
                print(f"  Reasoning:  {reasoning_status:<15} {reasoning_change:>+6.1f}% ({current['mean_reasoning_pct']:>10.1f}% → {next_epoch['mean_reasoning_pct']:>10.1f}%)")
                
                # Overall verdict
                if char_change < -15 or token_change < -15:
                    print(f"  Result: ✓ Solutions Working (>15% reduction)")
                elif char_change > 15 or token_change > 15:
                    print(f"  Result: ✗ Solutions NOT Working (increase instead)")
                else:
                    print(f"  Result: ≈ Marginal Change (within ±15%)")
        
        print(f"{'='*120}\n")
    
    def compare_epochs_detailed(self, epochs: list):
        """Compare epochs with detailed per-run breakdown for each epoch + aggregate comparison"""
        print(f"\n{'='*120}")
        print("DETAILED EPOCH COMPARISON - PER-RUN + AGGREGATE")
        print(f"{'='*120}\n")
        
        # First, show detailed per-run comparison for each epoch
        for epoch in epochs:
            files = self.find_output_files(epoch=epoch)
            
            if not files:
                print(f"[WARN] No output files found for epoch A{epoch}")
                continue
            
            print(f"\n{'='*100}")
            print(f"EPOCH A{epoch} - DETAILED PER-RUN BREAKDOWN ({len(files)} runs)")
            print(f"{'='*100}")
            
            metrics_list = []
            for idx, filepath in enumerate(files, 1):
                match = re.search(r'/synth_nn/(B\d+)/', filepath)
                run_id = match.group(1) if match else f"Run{idx}"
                
                metrics = self.analyze_file(filepath)
                if metrics:
                    metrics_list.append(metrics)
            
            # Show per-run table
            if metrics_list:
                print(f"{'Run':<8} {'Chars':<15} {'Lines':<15} {'Tokens':<15} {'Reasoning%':<15} {'Code%':<15}")
                print(f"{'-'*100}")
                
                for idx, m in enumerate(metrics_list, 1):
                    run_name = f"B{idx-1}"
                    reasoning_pct = (m['reasoning_chars'] / m['total_chars'] * 100) if m['total_chars'] > 0 else 0
                    code_pct = (m['code_chars'] / m['total_chars'] * 100) if m['total_chars'] > 0 else 0
                    print(f"{run_name:<8} {m['total_chars']:<15} {m['total_lines']:<15} {m['total_tokens']:<15} {reasoning_pct:<14.1f}% {code_pct:<14.1f}%")
                
                # Calculate B0 to Bn reduction
                if len(metrics_list) >= 2:
                    first = metrics_list[0]
                    last = metrics_list[-1]
                    char_reduction = ((first['total_chars'] - last['total_chars']) / first['total_chars'] * 100) if first['total_chars'] > 0 else 0
                    token_reduction = ((first['total_tokens'] - last['total_tokens']) / first['total_tokens'] * 100) if first['total_tokens'] > 0 else 0
                    
                    char_status = "✓ REDUCED" if char_reduction > 0 else "✗ INCREASED"
                    token_status = "✓ REDUCED" if token_reduction > 0 else "✗ INCREASED"
                    
                    print(f"{'-'*100}")
                    print(f"B0 → B{len(metrics_list)-1}: Characters {char_status} {char_reduction:>+6.1f}% | Tokens {token_status} {token_reduction:>+6.1f}%")
        
        # Now show aggregate cross-epoch comparison
        print(f"\n{'='*120}")
        print("AGGREGATE COMPARISON - MEAN METRICS ACROSS EPOCHS")
        print(f"{'='*120}")
        
        epoch_means = []
        for epoch in epochs:
            mean_data = self.calculate_epoch_mean(epoch)
            if mean_data:
                epoch_means.append(mean_data)
        
        if epoch_means:
            print(f"{'Epoch':<10} {'Runs':<8} {'Avg Chars':<15} {'Avg Tokens':<15} {'Avg Lines':<12} {'Reasoning%':<12} {'Code%':<12}")
            print(f"{'-'*120}")
            
            for data in epoch_means:
                code_pct = 100 - data['mean_reasoning_pct']
                print(f"A{data['epoch']:<9} {data['num_runs']:<8} {data['mean_chars']:>14,.0f} {data['mean_tokens']:>14,.0f} {data['mean_lines']:>11,.0f} {data['mean_reasoning_pct']:>11.1f}% {code_pct:>11.1f}%")
            
            # Compare consecutive epochs
            if len(epoch_means) >= 2:
                print(f"{'-'*120}")
                print("TREND ANALYSIS (Changes Between Epochs):")
                print(f"{'-'*120}")
                
                for i in range(len(epoch_means) - 1):
                    current = epoch_means[i]
                    next_epoch = epoch_means[i + 1]
                    
                    char_change = ((next_epoch['mean_chars'] - current['mean_chars']) / current['mean_chars'] * 100) if current['mean_chars'] > 0 else 0
                    token_change = ((next_epoch['mean_tokens'] - current['mean_tokens']) / current['mean_tokens'] * 100) if current['mean_tokens'] > 0 else 0
                    reasoning_change = next_epoch['mean_reasoning_pct'] - current['mean_reasoning_pct']
                    
                    char_status = "✓ REDUCED" if char_change < 0 else "✗ INCREASED"
                    token_status = "✓ REDUCED" if token_change < 0 else "✗ INCREASED"
                    reasoning_status = "✓ REDUCED" if reasoning_change < 0 else "✗ INCREASED"
                    
                    print(f"\nA{current['epoch']} → A{next_epoch['epoch']}:")
                    print(f"  Characters: {char_status:<15} {char_change:>+6.1f}% ({current['mean_chars']:>10,.0f} → {next_epoch['mean_chars']:>10,.0f})")
                    print(f"  Tokens:     {token_status:<15} {token_change:>+6.1f}% ({current['mean_tokens']:>10,.0f} → {next_epoch['mean_tokens']:>10,.0f})")
                    print(f"  Lines:      {reasoning_status if reasoning_change < 0 else '✗ CHANGED':<15} {next_epoch['mean_lines'] - current['mean_lines']:>+6.0f}")
                    print(f"  Reasoning:  {reasoning_status:<15} {reasoning_change:>+6.1f}% ({current['mean_reasoning_pct']:>10.1f}% → {next_epoch['mean_reasoning_pct']:>10.1f}%)")
                    
                    # Overall verdict
                    if char_change < -15 or token_change < -15:
                        print(f"  ✓ Solutions Working (>15% reduction)")
                    elif char_change > 15 or token_change > 15:
                        print(f"  ✗ Solutions NOT Working (increase instead)")
                    else:
                        print(f"  ≈ Marginal Change (within ±15%)")
        
        print(f"{'='*120}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze output sizes across epochs and runs")
    parser.add_argument('--epoch', type=int, help='Analyze specific epoch (A0, A1, etc)')
    parser.add_argument('--compare-epochs', type=int, nargs='+', help='Compare multiple epochs (e.g., --compare-epochs 0 1 2)')
    parser.add_argument('--compare-epochs-detailed', type=int, nargs='+', help='Compare epochs with per-run breakdown for each (e.g., --compare-epochs-detailed 0 1 2)')
    parser.add_argument('--watch', action='store_true', help='Watch for new files in real-time')
    parser.add_argument('--compare', action='store_true', help='Show comparison across runs')
    parser.add_argument('--base-path', default=None,
                       help='Base path for output files (default: ./out relative to script location)')
    
    args = parser.parse_args()
    
    analyzer = OutputAnalyzer(base_path=args.base_path)
    
    # Handle detailed epoch comparison (per-run + aggregate)
    if args.compare_epochs_detailed:
        analyzer.compare_epochs_detailed(args.compare_epochs_detailed)
    # Handle simple epoch comparison (aggregate only)
    elif args.compare_epochs:
        analyzer.compare_epochs(args.compare_epochs)
    # Handle watch mode
    elif args.watch:
        analyzer.watch_new_files()
    # Handle single epoch analysis
    elif args.epoch is not None:
        analyzer.analyze_epoch(args.epoch, compare=args.compare)
    else:
        # Analyze latest epoch
        files = analyzer.find_output_files()
        if files:
            # Extract latest epoch number
            match = re.search(r'epoch/A(\d+)/', files[-1])
            if match:
                latest_epoch = int(match.group(1))
                analyzer.analyze_epoch(latest_epoch, compare=args.compare)
        else:
            print("[WARN] No output files found")

if __name__ == "__main__":
    main()
