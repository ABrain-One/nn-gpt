import re
import matplotlib.pyplot as plt
import os

log_file = "results.log"
current_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(current_dir, log_file)
output_plot = os.path.join(current_dir, "accuracy_plot.png")

iterations = []
accuracies = []

last_acc = None

with open(log_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        m_iter = re.search(r"iteration:\s*(\d+)", line)
        if not m_iter:
            continue
        iteration = int(m_iter.group(1))
        
        m_acc = re.search(r"accuracy:\s*([\d\.]+)", line)
        if m_acc:
            acc = float(m_acc.group(1))
            last_acc = acc
        elif "error:" in line:
            if last_acc is None:
                last_acc = 0.0
        else:
            continue
            
        iterations.append(iteration)
        accuracies.append(last_acc)

iterations = iterations[:150]
accuracies = accuracies[:150]
runs = []
current_run = {"x": [], "y": []}
last_iter = -1

for it, acc in zip(iterations, accuracies):
    if it <= last_iter:
        runs.append(current_run)
        current_run = {"x": [], "y": []}
    current_run["x"].append(it)
    current_run["y"].append(acc)
    last_iter = it
if current_run["x"]:
    runs.append(current_run)

plt.figure(figsize=(10, 6))
for i, run in enumerate(runs):
    label = f"Run {i+1}" if len(runs) > 1 else "Accuracy"
    plt.plot(run["x"], run["y"], marker='o', linestyle='-', label=label)

plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy over 150 Iterations(2 Hours)")
plt.grid(True)
if len(runs) > 1:
    plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"Plot saved to: {output_plot}")
