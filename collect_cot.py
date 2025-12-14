import os
import json
import re

# 1. Define where to look
# Based on your file structure: ~/nn-gpt/out/nngpt/llm/epoch/
BASE_DIR = os.path.join(os.getcwd(), "out", "nngpt", "llm", "epoch")
OUTPUT_FILE = "deepseek_cot_dataset.json"

print(f"Searching for CoT files in: {BASE_DIR}")

dataset = []

# Regex to separate Thinking (CoT) from Code
# DeepSeek usually puts code in ```python ... ``` blocks.
# Everything before that is the "Chain of Thought".
code_block_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

# 2. Walk through all folders (A0, A1... -> B0, B1...)
for root, dirs, files in os.walk(BASE_DIR):
    if "full_output.txt" in files:
        file_path = os.path.join(root, "full_output.txt")
        
        try:
            # 3. Read the content
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            # 4. Extract Metadata from folder names
            # Path ends in .../epoch/A0/synth_nn/B5
            path_parts = root.split(os.sep)
            
            # Find A-number (Epoch) and B-number (Candidate)
            epoch = "unknown"
            candidate = "unknown"
            for part in path_parts:
                if part.startswith("A") and part[1:].isdigit():
                    epoch = part
                if part.startswith("B") and part[1:].isdigit():
                    candidate = part

            # 5. Separate Reasoning vs Code
            match = code_block_pattern.search(full_text)
            if match:
                # Found a code block
                reasoning = full_text[:match.start()].strip()
                code = match.group(1).strip()
            else:
                # No code block found? Maybe it failed or formatted differently.
                # We save everything as reasoning just in case.
                reasoning = full_text
                code = None

            # 6. Add to dataset
            entry = {
                "id": f"{epoch}_{candidate}",
                "epoch": epoch,
                "candidate": candidate,
                "file_path": file_path,
                "full_llm_output": full_text,
                "chain_of_thought": reasoning,
                "generated_code": code
            }
            dataset.append(entry)
            print(f"  [+] Found: {epoch}/{candidate} ({len(reasoning)} chars of thinking)")

        except Exception as e:
            print(f"  [!] Error reading {file_path}: {e}")

# 7. Save to JSON
print(f"\nProcessing complete.")
print(f"Found {len(dataset)} items.")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Saved dataset to: {os.path.abspath(OUTPUT_FILE)}")

