import os
import json
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_RESULTS = os.path.join(BASE_DIR, "baseline_results.json")
EVOLVED_RESULTS = os.path.join(BASE_DIR, "evolved_results.json")

def load_results(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def calculate_auc(history):
    # Area under the curve: sum of best fitnesses
    return sum(history)

def main():
    print("\n" + "="*60)
    print("GA INTELLIGENCE COMPARISON")
    print("="*60)
    
    baseline = load_results(BASELINE_RESULTS)
    evolved = load_results(EVOLVED_RESULTS)
    
    if not baseline:
        print("[!] ERROR: baseline_results.json not found. Run the baseline benchmark first.")
        return
        
    if not evolved:
        print("[!] ERROR: evolved_results.json not found. Run the evolved benchmark first.")
        return
        
    # Calculate Metrics
    base_peak = baseline.get("peak_accuracy", 0)
    evol_peak = evolved.get("peak_accuracy", 0)
    delta_peak = ((evol_peak - base_peak) / max(1e-5, base_peak)) * 100
    
    base_top3 = baseline.get("top3_mean", 0)
    evol_top3 = evolved.get("top3_mean", 0)
    delta_top3 = ((evol_top3 - base_top3) / max(1e-5, base_top3)) * 100
    
    base_archive = baseline.get("archive_size", 0)
    evol_archive = evolved.get("archive_size", 0)
    delta_archive = ((evol_archive - base_archive) / max(1, base_archive)) * 100
    
    base_hist = baseline.get("fitness_history", [])
    evol_hist = evolved.get("fitness_history", [])
    
    base_auc = calculate_auc(base_hist)
    evol_auc = calculate_auc(evol_hist)
    delta_auc = ((evol_auc - base_auc) / max(1e-5, base_auc)) * 100
    
    # Print Summary Table
    print(f"\n{ 'Metric':<25} | { 'Baseline':<12} | { 'Evolved':<12} | { 'Delta (%)':<10}")
    print("-" * 65)
    print(f"{ 'Peak Accuracy':<25} | {base_peak:>11.2f}% | {evol_peak:>11.2f}% | {delta_peak:>+9.2f}%")
    print(f"{ 'Top-3 Mean':<25} | {base_top3:>11.2f}% | {evol_top3:>11.2f}% | {delta_top3:>+9.2f}%")
    print(f"{ 'Convergence AUC':<25} | {base_auc:>12.2f} | {evol_auc:>12.2f} | {delta_auc:>+9.2f}%")
    print(f"{ 'Archive Coverage (Cells)':<25} | {base_archive:>12} | {evol_archive:>12} | {delta_archive:>+9.2f}%")
    print("-" * 65)
    
    if evol_auc > base_auc:
        print("\n[RESULT] The LLM-evolved GA is MATHEMATICALLY SMARTER than the baseline!")
    elif evol_auc < base_auc:
        print("\n[RESULT] The LLM-evolved GA has REGRESSED compared to the baseline.")
    else:
        print("\n[RESULT] Both GAs performed exactly the same.")
        
    # Generate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(base_hist) + 1), base_hist, label="Baseline (Dumb) GA", color="red", linestyle="--", marker="o")
    plt.plot(range(1, len(evol_hist) + 1), evol_hist, label="LLM-Evolved GA", color="blue", linewidth=2, marker="s")
    
    plt.title("GA Intelligence: Convergence Speed Comparison")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Accuracy %)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(BASE_DIR, "ga_intelligence_comparison.png")
    plt.savefig(plot_path)
    print(f"\nSaved comparison plot to: {plot_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
