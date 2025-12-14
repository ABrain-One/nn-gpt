import json
import os

# Load your scraped data
with open("deepseek_cot_dataset.json", "r") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} entries. Looking for scores...")

for entry in dataset:
    folder_path = os.path.dirname(entry['file_path'])
    eval_path = os.path.join(folder_path, "eval_info.json")
    
    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r") as f:
                data = json.load(f)
            
            # Get the results
            results = data.get("eval_results")

            # Handle List format (What your repo produces: [Name, Accuracy, Time, Score])
            if isinstance(results, list) and len(results) > 1:
                acc = results[1]  # Index 1 is accuracy
                entry["accuracy"] = float(acc)
                entry["status"] = "Success"
                print(f"  [OK] {entry['id']} -> Accuracy: {acc}")
            
            # Handle Dictionary format (Just in case)
            elif isinstance(results, dict):
                acc = results.get("accuracy", 0.0)
                entry["accuracy"] = float(acc)
                entry["status"] = "Success"
                print(f"  [OK] {entry['id']} -> Accuracy: {acc}")
            
            else:
                entry["accuracy"] = 0.0
                entry["status"] = "No Accuracy Found"

        except Exception as e:
            print(f"Error reading {entry['id']}: {e}")
            entry["accuracy"] = 0.0
            entry["status"] = "Eval Corrupted"
    else:
        entry["accuracy"] = 0.0
        entry["status"] = "Failed"

# Save the final labeled dataset
with open("deepseek_cot_labeled.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("\nDone! Created 'deepseek_cot_labeled.json' with corrected scores.")
