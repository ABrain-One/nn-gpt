import os
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from ab.nn.util.Const import out_dir
from scipy import stats
from collections import defaultdict

# 1. Configuration
# Use relative path for portability
base_path = out_dir /  'nngpt/llm/epoch'

def collect_final_data():
    stats_map = defaultdict(lambda: {"success": 0, "already_exists": 0, "total": 0, "acc_list": []})
    a_dirs = glob.glob(os.path.join(base_path, "A*"))
    
    for a_path in a_dirs:
        match = re.search(r'A(\d+)', a_path)
        if not match: continue
        c_idx = int(match.group(1))
        
        synth_path = os.path.join(a_path, "synth_nn")
        if not os.path.exists(synth_path): continue
        
        # Count total attempts (number of B* directories)
        b_dirs = [d for d in os.listdir(synth_path) if d.startswith('B') and os.path.isdir(os.path.join(synth_path, d))]
        stats_map[c_idx]["total"] = len(b_dirs)
        
        for b_dir in b_dirs:
            b_path = os.path.join(synth_path, b_dir)
            json_file = os.path.join(b_path, "1.json")
            error_file = os.path.join(b_path, "error.txt")
            if os.path.exists(json_file):
                stats_map[c_idx]["success"] += 1
                try:
                    with open(json_file, 'r') as jf:
                        data = json.load(jf)
                        # Limit accuracy within physical bounds [0, 100]
                        acc = max(0, min(100, data[0]['accuracy'] * 100))
                        stats_map[c_idx]["acc_list"].append(acc)
                except: continue
            elif os.path.exists(error_file):
                try:
                    with open(error_file, 'r', encoding='utf-8') as ef:
                        if "NN already exists" in ef.read():
                            stats_map[c_idx]["already_exists"] += 1
                except: continue
    return stats_map

# 2. Compute plotting metrics
data_map = collect_final_data()
cycles = sorted(data_map.keys())

gen_rates, total_rates, acc_best, acc_avg, acc_median, err_down, err_up = [], [], [], [], [], [], []

for idx in cycles:
    s = data_map[idx]["success"]
    ae = data_map[idx]["already_exists"]
    t = data_map[idx]["total"]
    accs = np.array(data_map[idx]["acc_list"])
    
    # Calculate success rate
    gen_rates.append((s / t * 100) if t > 0 else 0)
    # Calculate total success rate (including "NN already exists")
    total_rates.append(((s + ae) / t * 100) if t > 0 else 0)
    
    if len(accs) > 0:
        best = np.max(accs)
        avg = np.mean(accs)
        med = np.median(accs)
        
        # Calculate 95% Confidence Interval
        if len(accs) >= 2:
            h = stats.sem(accs) * stats.t.ppf(0.975, len(accs) - 1)
        else:
            h = 0
            
        acc_best.append(best)
        acc_avg.append(avg)
        acc_median.append(med)
        
        # Fix: Clip error bars to ensure they don't exceed [0, 100]
        low_bound = max(0, avg - h)
        high_bound = min(100, avg + h)
        err_down.append(avg - low_bound)
        err_up.append(high_bound - avg)
    else:
        for l in [acc_best, acc_avg, acc_median, err_down, err_up]: l.append(np.nan)

# 3. Plotting
plt.figure(figsize=(13, 6.5), dpi=120)
plt.grid(True, linestyle='--', alpha=0.4, zorder=0)

# A. Accuracy error bars (orange vertical lines)
plt.errorbar(cycles, acc_avg, yerr=[err_down, err_up], 
             fmt='none', ecolor='#FF9800', elinewidth=1.8, capsize=5, 
             alpha=0.6, zorder=1, label='Accuracy 95% CI')

# B. Specific accuracy metrics
# Best Accuracy - Diamond
plt.scatter(cycles, acc_best, marker='D', color='#E65100', s=80, 
            edgecolors='white', linewidth=0.8, label='Best Acc', zorder=5)
# Median Accuracy - Horizontal line
plt.scatter(cycles, acc_median, marker='_', color='#D32F2F', s=200, 
            linewidths=3, label='Median Acc', zorder=6)
# Avg Accuracy - Dot
plt.scatter(cycles, acc_avg, marker='o', color='#F57C00', s=45, 
            edgecolors='black', linewidth=0.5, label='Avg Acc', zorder=4)

# C. Generation success rates
# Total Success Rate (including existing) - Purple dashed line
plt.plot(cycles, total_rates, marker='s', color='#9C27B0', linewidth=2, 
         linestyle='--', markersize=6, label='Total Success Rate (incl. Existing) (%)', zorder=3)

# Valid Generation Rate - Blue main line
plt.plot(cycles, gen_rates, marker='o', color='#1f77b4', linewidth=2.5, 
         markersize=8, label='Valid Generation Rate (%)', zorder=3)

# D. Avg Accuracy Trend - Orange solid line
plt.plot(cycles, acc_avg, color='#F57C00', linewidth=1.5, alpha=0.7, 
         linestyle='-', label='Avg Acc Trend', zorder=2)

# 4. Fine-tuning
plt.title('Model Evaluation: Code Generation Stability & Quality', fontsize=14, pad=15)
plt.xlabel('Cycle (Experiment Index)', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(cycles)

# Lock Y-axis to [0, 100]
plt.ylim(0, 100) 

# Legend in the lower right corner with two columns
plt.legend(loc='lower right', ncol=2, fontsize=10, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig("hybrid_analysis.png")
plt.show()

print("Done")
