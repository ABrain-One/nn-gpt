import os
import re
import ast
import random
import subprocess
import textwrap
import shutil
import time
import json
import sys
from datetime import datetime

from ab.gpt.brute.ga.meta_evolution.llm_loader import LocalLLMLoader 
from ab.gpt.brute.ga.meta_evolution.rl_rewards import calculate_meta_reward

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FILE = os.path.join(BASE_DIR, "genetic_algorithm.py")
RUNNER_SCRIPT = os.path.join(BASE_DIR, "run_fractal_evolution.py")
BACKUP_DIR = os.path.join(BASE_DIR, "ga_history_backup")
LOG_FILE = os.path.join(BASE_DIR, "LLM-evolution-logs.jsonl")
ADAPTER_SAVE_PATH = os.path.join(BASE_DIR, "fine_tuned_adapter")

# KEEP BENCHMARKS SMALL FOR FAST FEEDBACK, BUT CONFIGURABLE VIA ENV
BENCH_GENS = int(os.environ.get("GENERATIONS", 3))
BENCH_POP = int(os.environ.get("POPULATION_SIZE", 10)) 

# --- MICRO-MUTATION PROMPTS ---
PROMPTS = {
    "mutate_gene": """
Improve this mutation helper function for a genetic algorithm.
Goal: Select a new gene value to encourage exploration but respect valid choices.

Existing Code:
{code}

Output ONLY the python code starting with 'def mutate_gene(self, current_value, possible_values):'
""",
    "combine_genes": """
Improve this crossover helper function for a genetic algorithm.
Goal: Decide which parent's gene value to use when creating a child chromosome.
You may use blending, uniform crossover, or any creative strategy.
Parameters: gene_name (str), parent1_value, parent2_value, crossover_point (int), gene_index (int), total_genes (int).

Existing Code:
{code}

Output ONLY the python code starting with 'def combine_genes(self, gene_name, parent1_value, parent2_value, crossover_point, gene_index, total_genes):'
""",
    "select_competitor": """
Improve this selection helper function for a genetic algorithm.
Goal: Pick the best individual from a list of tournament competitors.
Each competitor is a dict with 'chromosome' and 'fitness' keys.
You may add fitness-proportional selection, diversity bonuses, or other strategies.

Existing Code:
{code}

Output ONLY the python code starting with 'def select_competitor(self, competitors):'
"""
}

class MetaEvolver:
    def __init__(self, model_path):
        self.llm = LocalLLMLoader(model_path, use_quantization=True, adapter_path=ADAPTER_SAVE_PATH)
        os.makedirs(BACKUP_DIR, exist_ok=True)
        
        # print("[Meta] Running Baseline...")
        # self.baseline_score = self.run_benchmark()
        # print(f"[Meta] Baseline Score: {self.baseline_score:.4f}")
        self.baseline_score = 0.0

    def run_benchmark(self):
        cmd = [sys.executable, RUNNER_SCRIPT, "--gens", str(BENCH_GENS), "--pop", str(BENCH_POP), "--clean"]
        
        try:
            # Increased timeout to 6 hours for large generations
            # Real-Time Logging with Popen
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            full_output = ""
            for line in process.stdout:
                print(line, end="") # Stream line
                full_output += line
                
            process.wait(timeout=21600)
            
            match = re.search(r"META_SCORE:\s*([\d\.]+)", full_output)
            if match:
                return float(match.group(1))
            else:
                print(f"[Meta] Benchmark Output (Snippet):\n{full_output[-1000:]}")
                return 0.0
        except Exception as e: 
             print(f"[Meta] Benchmark Exception: {e}")
             return 0.0

    def _extract_method(self, source_code, method_name):
        try:
            tree = ast.parse(source_code)
            t_class = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GeneticAlgorithm"), None)
            if not t_class: return None, None, 0
            node = next((n for n in t_class.body if isinstance(n, ast.FunctionDef) and n.name == method_name), None)
            if node:
                # Use Line numbers for robust extraction
                lines = source_code.splitlines(keepends=True)
                # node.lineno is 1-indexed. node.end_lineno is inclusive.
                start_line = node.lineno - 1
                end_line = node.end_lineno
                
                # Check for decorators (they are before the def)
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1

                extracted_source = "".join(lines[start_line:end_line])
                
                # Calculate byte offsets
                # We simply join the preceding lines to find start byte
                start_byte = sum(len(line) for line in lines[:start_line])
                end_byte = sum(len(line) for line in lines[:end_line])
                
                return extracted_source, (start_byte, end_byte), node.col_offset
        except: pass
        return None, None, 0

    def _extract_function_body(self, text, method_name):
        """
        Robustly extract a function definition from a string that might contain chatter.
        Handles multiple occurrences by picking the last valid one.
        """
        candidates = []
        lines = text.splitlines()
        
        # Find all start indices of 'def method_name'
        start_indices = [i for i, line in enumerate(lines) if line.strip().startswith(f"def {method_name}")]
        
        for start_idx in start_indices:
            extracted_lines = [lines[start_idx]]
            # Determine indentation of the body (first non-empty line after def)
            body_indent = None
            
            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                if not line.strip(): # Empty line, keep it
                    extracted_lines.append(line)
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                
                if body_indent is None:
                    body_indent = current_indent
                    if body_indent == 0: # Body must be indented!
                        break 
                
                if current_indent < body_indent:
                    break # End of function
                
                extracted_lines.append(line)
            
            # Form candidate code
            code_str = "\n".join(extracted_lines)
            try:
                # Verify syntax
                tree = ast.parse(textwrap.dedent(code_str))
                if tree.body and isinstance(tree.body[0], ast.FunctionDef):
                    candidates.append(ast.unparse(tree))
            except:
                continue

        return candidates[-1] if candidates else None

    def evolve_component(self, method_name):
        print(f"\n[Meta] Evolving: {method_name}")
        with open(TARGET_FILE, 'r') as f: full_code = f.read()
        
        orig_code, span, indent_col = self._extract_method(full_code, method_name)
        if not orig_code: return

        print(f"--- [DEBUG] Input Code ---\n{orig_code}\n-------------------------")

        # LLM Generation
        prompt = PROMPTS[method_name].format(code=orig_code)
        raw_res = self.llm.generate(prompt, max_new_tokens=500)
        
        print(f"--- [DEBUG] Raw Response ---\n{raw_res}\n--------------------------")
        
        # Cleanup & Extraction
        new_code = raw_res
        extracted = None
        if "```python" in new_code: 
            # Try extracting from markdown block first
            try:
                block = new_code.split("```python")[1].split("```")[0]
                extracted = self._extract_function_body(block, method_name)
                if extracted: new_code = extracted
            except: pass
        
        if f"def {method_name}" in raw_res and not extracted:
             extracted = self._extract_function_body(raw_res, method_name)
        
        if extracted:
            new_code = extracted
        else:
            print(f"[Meta] Failed to extract valid {method_name} from LLM response. Skipping.")
            return

        # Normalize Indentation
        new_code = textwrap.dedent(new_code).strip()
        
        print(f"--- [DEBUG] Cleaned Code (Normalized) ---\n{new_code}\n---------------------------")
        
        # Indentation for Injection
        reindented = "\n".join([" " * indent_col + line for line in new_code.splitlines()])

        # Syntax Check & Injection
        valid_syntax = False
        try:
            test_full = full_code[:span[0]] + reindented + "\n" + full_code[span[1]:]

            # DEBUG: Print what we are trying to inject
            print(f"--- [DEBUG] Injection Snippet ---\n{test_full[span[0]-50:span[0]+len(reindented)+50]}\n-------------------------------")

            ast.parse(test_full)
            valid_syntax = True
        except SyntaxError as e:
            print(f"[Meta] Syntax Error: {e}")

        new_score = 0.0
        if valid_syntax:
            target_filename = os.path.basename(TARGET_FILE)
            bkp = os.path.join(BACKUP_DIR, f"{target_filename}_{method_name}.bak")
            shutil.copy(TARGET_FILE, bkp)
            with open(TARGET_FILE, 'w') as f: f.write(test_full)
            
            print("[Meta] Benchmarking...")
            new_score = self.run_benchmark()

        # RL Loop
        reward = calculate_meta_reward(new_score, self.baseline_score, valid_syntax)
        
        fine_tune_expected = (reward > 0 and valid_syntax)
        fine_tune_started = False
        fine_tune_completed = False
        fine_tune_failed = False
        fine_tune_exception = None
        adapter_save_started = False
        adapter_save_completed = False
        adapter_save_failed = False
        adapter_save_exception = None
        adapter_path = ADAPTER_SAVE_PATH
        train_examples_count = 1 if fine_tune_expected else 0
        train_epochs = 1 if fine_tune_expected else 0
        fine_tune_start_time = None
        fine_tune_end_time = None
        adapter_save_start_time = None
        adapter_save_end_time = None
        
        if fine_tune_expected:
            print("--> SUCCESS. Updating Baseline & Fine-tuning.")
            self.baseline_score = new_score
            
            print("[LoRA] Fine-tune start")
            fine_tune_started = True
            fine_tune_start_time = datetime.now().isoformat()
            try:
                self.llm.train_on_buffer([{'prompt': prompt, 'completion': new_code}], epochs=train_epochs)
                fine_tune_completed = True
                print("[LoRA] Fine-tune complete")
            except Exception as e:
                fine_tune_failed = True
                fine_tune_exception = str(e)
                print(f"[LoRA] Fine-tune failed: {e}")
            finally:
                fine_tune_end_time = datetime.now().isoformat()
                
            if fine_tune_completed:
                print("[LoRA] Adapter save start")
                adapter_save_started = True
                adapter_save_start_time = datetime.now().isoformat()
                try:
                    self.llm.save_adapters(ADAPTER_SAVE_PATH)
                    adapter_save_completed = True
                    print("[LoRA] Adapter save complete")
                except Exception as e:
                    adapter_save_failed = True
                    adapter_save_exception = str(e)
                    print(f"[LoRA] Adapter save failed: {e}")
                finally:
                    adapter_save_end_time = datetime.now().isoformat()
        elif valid_syntax:
            print("--> Reverting File.")
            shutil.copy(bkp, TARGET_FILE)
            
        # Log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method_name,
            "prompt": prompt,
            "response": raw_res,
            "cleaned_code": new_code,
            "valid_syntax": valid_syntax,
            "score": new_score,
            "reward": reward,
            "fine_tune_expected": fine_tune_expected,
            "fine_tune_started": fine_tune_started,
            "fine_tune_completed": fine_tune_completed,
            "fine_tune_failed": fine_tune_failed,
            "fine_tune_exception": fine_tune_exception,
            "adapter_save_started": adapter_save_started,
            "adapter_save_completed": adapter_save_completed,
            "adapter_save_failed": adapter_save_failed,
            "adapter_save_exception": adapter_save_exception,
            "adapter_path": adapter_path,
            "train_examples_count": train_examples_count,
            "train_epochs": train_epochs,
            "fine_tune_start_time": fine_tune_start_time,
            "fine_tune_end_time": fine_tune_end_time,
            "adapter_save_start_time": adapter_save_start_time,
            "adapter_save_end_time": adapter_save_end_time
        }
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

if __name__ == "__main__":
    MODEL_PATH = "deepseek-ai/deepseek-coder-6.7b-instruct" 
    evolver = MetaEvolver(MODEL_PATH)
    
    # LOOP: CYCLE THROUGH ALL EVOLVABLE COMPONENTS
    META_ITERATIONS = int(os.environ.get("META_ATTEMPTS"))
    COMPONENTS = ["mutate_gene", "combine_genes", "select_competitor"]
    for i in range(META_ITERATIONS):
        component = COMPONENTS[i % len(COMPONENTS)]
        print(f"\n=== Meta-Evolution Iteration {i+1}/{META_ITERATIONS} — Evolving: {component} ===")
        evolver.evolve_component(component) 
        time.sleep(2)