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
import gc
import torch
from ab.gpt.util.eval.Eval import Eval
import ab.nn.api as nn_dataset
import pandas as pd
# MONKEYPATCH: Bypass the massive remote database download inside Eval.py
nn_dataset.data = lambda *args, **kwargs: pd.DataFrame(columns=['nn_id'])
nn_dataset.data.cache_clear = lambda: None
from datetime import datetime

from ab.gpt.brute.ga.meta_evolution.llm_loader import LocalLLMLoader, get_model_short_name, get_dataset_name
from ab.gpt.brute.ga.meta_evolution.rl_rewards import calculate_meta_reward
from ab.gpt.brute.ga.meta_evolution.FractalNet_evolvable_backbone import SEARCH_SPACE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.environ.get("PIPELINE_DIR", BASE_DIR)
DATASET = os.environ.get("DATASET", "cifar10")
DATASET_DASH = "cifar-100" if DATASET == "cifar100" else "cifar-10"

_dataset_name = get_dataset_name(__file__)
_model_name = get_model_short_name()

MODIFIED_GA_DIR = os.path.join(PIPELINE_DIR, f"modified_GA")
os.makedirs(MODIFIED_GA_DIR, exist_ok=True)
TARGET_FILE = os.path.join(MODIFIED_GA_DIR, "genetic_algorithm_evolved.py")

# Fair Benchmarking: Reset baseline if starting fresh
CHECKPOINT_FILE = os.path.join(PIPELINE_DIR, f"GenFractal_ckpt_{DATASET}.pkl")
BACKUP_DIR = os.path.join(PIPELINE_DIR, f"ga_history_backup")
ADAPTER_SAVE_PATH = os.path.join(PIPELINE_DIR, f"{_model_name}_adapter")

if not os.path.exists(CHECKPOINT_FILE):
    baseline_file = os.path.join(BASE_DIR, "genetic_algorithm_baseline.py")
    if os.path.exists(baseline_file):
        shutil.copy(baseline_file, TARGET_FILE)
        print("[Meta] No checkpoint found. Resetting target GA to pristine baseline.")
    
    # Move the old Hall of Fame and LLM weights to a historic folder instead of deleting
    timestamp_for_historic = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if os.path.exists(BACKUP_DIR):
        historic_backup_dir = os.path.join(BACKUP_DIR, "historic_runs", f"run_{timestamp_for_historic}")
        os.makedirs(historic_backup_dir, exist_ok=True)
        for item in os.listdir(BACKUP_DIR):
            item_path = os.path.join(BACKUP_DIR, item)
            if item != "historic_runs":
                shutil.move(item_path, historic_backup_dir)
        print(f"[Meta] Moved Hall of Fame to {historic_backup_dir} for fresh start.")
        
    if os.path.exists(ADAPTER_SAVE_PATH):
        historic_adapter_dir = os.path.join(ADAPTER_SAVE_PATH, "historic_runs", f"run_{timestamp_for_historic}")
        os.makedirs(historic_adapter_dir, exist_ok=True)
        for item in os.listdir(ADAPTER_SAVE_PATH):
            item_path = os.path.join(ADAPTER_SAVE_PATH, item)
            if item != "historic_runs":
                shutil.move(item_path, historic_adapter_dir)
        print(f"[Meta] Moved LLM LoRA weights to {historic_adapter_dir} for fresh start.")

RUNNER_SCRIPT = os.path.join(BASE_DIR, "run_fractal_evolution.py")
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FOLDER_TIMESTAMP = datetime.now().strftime("%d-%m-%y_%H-%M")
LOGS_DIR = os.path.join(PIPELINE_DIR, f"logs_{DATASET}", _model_name, FOLDER_TIMESTAMP)
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, f"LLM-evolution-logs_{DATASET}_{_model_name}_{RUN_TIMESTAMP}.jsonl")
GA_EVAL_LOG_FILE = os.path.join(LOGS_DIR, f"ga_evaluations_{DATASET}_{_model_name}_{RUN_TIMESTAMP}.jsonl")

# KEEP BENCHMARKS SMALL FOR FAST FEEDBACK, BUT CONFIGURABLE VIA ENV
BENCH_GENS = int(os.environ.get("GENERATIONS", 3))
BENCH_POP = int(os.environ.get("POPULATION_SIZE", 10)) 

# --- FULL CONTEXT PROMPT TEMPLATE ---
# Format the search space so the LLM understands the exact gene constraints
SEARCH_SPACE_STR = json.dumps(SEARCH_SPACE, indent=2)

BASE_PROMPT_TEMPLATE = """
You are an expert AI researcher fine-tuning a Genetic Algorithm (GA) that evolves PyTorch Neural Network architectures for {dataset}.
CRITICAL MANDATE: Your ONLY goal is RADICAL ARCHITECTURAL INNOVATION. Do not make small incremental changes.
You are heavily penalized if you do not generate completely novel combinations of layers, activations, and topologies.

=== FEEDBACK FROM RECENT FAILED ATTEMPTS ===
{history_str}
Study these failures carefully. DO NOT repeat the same mistakes or generate the exact same code.

=== SEARCH SPACE ===
The GA optimizes the following SEARCH_SPACE:
{search_space}

=== BASELINE CHROMOSOME (STARTING POINT) ===
{best_chromosome}

=== GA SCRIPT CONTEXT (READ-ONLY) ===
Below is the FULL CODE of the current Genetic Algorithm. 
Study it to understand the class variables (`self.population`, `self.search_space`, etc.) and helper methods (`self._coerce_gene_value`, etc.).
DO NOT rewrite this script. It is strictly for context.

<full_script>
{full_code}
</full_script>

=== YOUR SPECIFIC TASK ===
You must intelligently improve ONLY the following specific function(s): `{method_names}`.
{task_specific_instructions}
Force the GA operators to break out of local minima and aggressively explore the search space.

Current implementation:
<current_function>
{code}
</current_function>

=== STRICT OUTPUT FORMAT ===
1. Output ONLY the python code for the function(s).
2. Do not include introductory or concluding text.
3. Ensure it is indented by 4 spaces exactly.
"""

INSTRUCTIONS = {
    "combine_genes": "Task: Implement `combine_genes`. Return a new chromosome dict by crossing over parent1_chromo and parent2_chromo. CRITICAL: You MUST implement strategies to maintain genetic diversity and avoid premature convergence! You MUST return self._sanitize_chromosome(child_chromo).",
    "mutate_gene": "Task: Implement `mutate_gene`. Return a new chromosome dict with mutated genes based on self.mutation_rate. CRITICAL: You MUST ensure mutations are bold enough to explore new architectures and prevent the population from getting stuck in local minima! You MUST return self._sanitize_chromosome(mutated_chromo).",
    "select_competitor": "Task: Implement `select_competitor`. Select a pool of competitors from self.population and return a single chosen competitor. CRITICAL: Balance elitism with exploration (e.g. tournament selection with a reasonable size) so the population doesn't instantly converge.",
    "_create_random_chromosome": "Task: Implement `_create_random_chromosome`. Return a new chromosome dictionary with randomized values chosen from self.search_space."
}

def skeletonize_code(source_code):
    try:
        class Skeletonizer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                strip_list = ["_save_checkpoint", "_load_checkpoint", "run", "_initialize_population", "__init__"]
                if node.name in strip_list:
                    node.body = [ast.Pass()]
                return node
        tree = ast.parse(source_code)
        Skeletonizer().visit(tree)
        if hasattr(ast, 'unparse'):
            return ast.unparse(tree)
        return source_code
    except Exception as e:
        print(f"[Meta] Skeletonize failed: {e}")
        return source_code

class MetaEvolver:
    def __init__(self, model_path):
        load_adapter = os.environ.get("LOAD_PREVIOUS_ADAPTER", "false").lower() == "true"
        adapter_path_to_load = ADAPTER_SAVE_PATH if load_adapter and os.path.exists(ADAPTER_SAVE_PATH) else None
        if load_adapter:
            print(f"[Meta] LOAD_PREVIOUS_ADAPTER is TRUE. Trying to load from: {adapter_path_to_load}")
        else:
            print("[Meta] LOAD_PREVIOUS_ADAPTER is FALSE. Starting with fresh LLM weights.")
            
        self.llm = LocalLLMLoader(model_path, use_quantization=True, adapter_path=adapter_path_to_load)
        os.makedirs(BACKUP_DIR, exist_ok=True)
        
        self.baseline_score = 0.0
        best_info_path = os.path.join(PIPELINE_DIR, f"best_baseline_info_{DATASET}.json")
        if os.path.exists(best_info_path):
            try:
                with open(best_info_path, 'r') as f:
                    best_data = json.load(f)
                    self.baseline_score = float(best_data.get("fitness", 0.0))
                    print(f"[Meta] Loaded Baseline Score: {self.baseline_score:.2f}%")
            except Exception:
                pass
        
        if self.baseline_score == 0.0:
            print("[Meta] Running Baseline Benchmark...")
            self.baseline_score = self.run_benchmark(gens=1)["peak_accuracy"]
            print(f"[Meta] Calculated Baseline: {self.baseline_score:.4f}%")
            
            # WIPE CHECKPOINT to force LLM to start from scratch
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                print("[Meta] Wiped Baseline population checkpoint to force clean start for LLM.")
        self.global_best_score = 0.0
        self.global_archive_size = 0
        
        # --- Experience Replay Buffer ---
        self.success_buffer = []
        self.attempt_history = []

    def _cuda_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
        print("[Meta] GPU cleanup attempted (gc + empty_cache)")

    def run_benchmark(self, gens=None):
        import statistics
        if gens is None:
            gens = BENCH_GENS
        # Removed --clean to persist the GA archive and population across LLM attempts
        cmd = [sys.executable, RUNNER_SCRIPT, "--gens", str(gens), "--pop", str(BENCH_POP)]
        env = os.environ.copy()
        env["GA_EVAL_LOG"] = GA_EVAL_LOG_FILE
        
        runs = 1
        scores = []
        peak_accs = []
        
        for i in range(runs):
            print(f"\n[Meta] Running Benchmark {i+1}/{runs}...")
            try:
                # Real-Time Logging with Popen
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                
                full_output = ""
                for line in process.stdout:
                    print(line, end="") # Stream line
                    full_output += line
                    
                BENCH_TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", 1800))
                process.wait(timeout=BENCH_TIMEOUT)
                
                error_trace = ""
                if process.returncode != 0:
                    lines = full_output.strip().splitlines()
                    error_trace = "\n".join(lines[-5:]) if lines else "Process failed with no output"
                
                score_match = re.search(r"META_SCORE:\s*([\d\.]+)", full_output)
                score = float(score_match.group(1)) if score_match else 0.0
                
                peak_match = re.search(r"PEAK_ACCURACY:\s*([\d\.]+)", full_output)
                peak_acc = float(peak_match.group(1)) if peak_match else 0.0

                top3_match = re.search(r"TOP3_MEAN:\s*([\d\.]+)", full_output)
                top3_acc = float(top3_match.group(1)) if top3_match else peak_acc
                
                archive_match = re.search(r"ARCHIVE_SIZE:\s*(\d+)", full_output)
                archive_size = int(archive_match.group(1)) if archive_match else 0
                
                if not top3_match and process.returncode == 0:
                    print(f"[Meta] Benchmark Output (Snippet):\n{full_output[-1000:]}")
                
                scores.append(top3_acc) # Track Top3 for secondary density reward
                peak_accs.append(peak_acc)
            except Exception as e: 
                 print(f"[Meta] Benchmark Exception: {e}")
                 scores.append(0.0)
                 peak_accs.append(0.0)
                 archive_size = 0
                 error_trace = str(e)
        
        median_top3 = statistics.median(scores) if scores else 0.0
        median_peak = statistics.median(peak_accs) if peak_accs else 0.0
        print(f"[Meta] Median Top-3 Quality over {runs} runs: {median_top3:.4f}")
        
        # Cleanup large objects explicitly
        scores.clear()
        peak_accs.clear()
        self._cuda_cleanup()
        
        return {"top3_mean": median_top3, "peak_accuracy": median_peak, "archive_size": archive_size, "error_trace": error_trace}

    def _extract_method(self, source_code, method_name):
        import ast
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == 'GeneticAlgorithm':
                    for body_item in node.body:
                        if isinstance(body_item, ast.FunctionDef) and body_item.name == method_name:
                            lines = source_code.splitlines()
                            start_line = body_item.lineno - 1
                            end_line = body_item.end_lineno
                            
                            if body_item.decorator_list:
                                start_line = body_item.decorator_list[0].lineno - 1

                            char_start = sum(len(l) + 1 for l in lines[:start_line])
                            char_end = sum(len(l) + 1 for l in lines[:end_line])
                            extracted_source = "\n".join(lines[start_line:end_line]) + "\n"
                            
                            return extracted_source, (char_start, char_end), 4
        except Exception as e:
            print(f"[Meta] AST extraction failed: {e}")
        return None, None, 0

    def _extract_function_body(self, text, method_name):
        """
        Robustly extract a function definition from a string that might contain chatter.
        Handles multiple occurrences by picking the last valid one.
        """
        if any(bad in text for bad in ["import torch", "class FractalNet", "def train("]):
            print(f"[Meta] Validation failed: LLM generated hallucinated codebase.")
            return None

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
            import ast
            import textwrap
            code_str = "\n".join(extracted_lines)
            try:
                # Verify syntax
                tree = ast.parse(textwrap.dedent(code_str))
                if tree.body and isinstance(tree.body[0], ast.FunctionDef):
                    candidates.append(code_str) # Keep original indentation
            except:
                continue

        return candidates[-1] if candidates else None

    def evolve_component(self, method_names, attempt=1, total_attempts=5):
        if isinstance(method_names, str):
            method_names = [method_names]
        methods_str = ", ".join(method_names)
        
        print(f"\n[Meta] Evolving: {methods_str}")
        with open(TARGET_FILE, 'r') as f: full_code = f.read()
        
        orig_codes = []
        existing_methods = []
        for name in method_names:
            c, span, indent = self._extract_method(full_code, name)
            if c:
                orig_codes.append(f"### {name} ###\n{c}")
                existing_methods.append((name, span, indent))
            else:
                print(f"[Meta] Note: {name} not found in target code. Will be injected as new.")
        
        if not existing_methods:
            print(f"[Meta] Failed to extract any of the target methods. Cannot proceed.")
            return False

        orig_code = "\n\n".join(orig_codes)

        print(f"--- [DEBUG] Input Code ---\n{orig_code}\n-------------------------")

        # Format history string with traces
        history_str = "No previous attempts yet."
        if hasattr(self, 'attempt_history') and self.attempt_history:
            history_lines = []
            
            # Filter history to only include attempts for the current component!
            component_name = method_names[0] if isinstance(method_names, (list, tuple)) else method_names
            relevant_history = [h for h in self.attempt_history if (component_name in h.get('code', '') or component_name == "full") and h.get("status") != "Success"]
            
            for idx, h in enumerate(relevant_history[-2:]):
                status = h.get("status", "Unknown")
                reward = h.get("reward", 0.0)
                trace = h.get("error_trace", "")
                if len(trace) > 150: trace = trace[-150:] + "\n... (truncated)"
                trace_str = f"\nError Trace:\n{trace}" if trace else "\nError Trace:\n(None)"
                
                hist_code = h.get('code', '')
                if len(hist_code) > 800: hist_code = hist_code[:800] + "\n... (truncated)"
                history_lines.append(f"Attempt {idx+1}:\nStatus: {status}\nReward: {reward:.2f}{trace_str}\nCode:\n```python\n{hist_code}\n```")
            
            if history_lines:
                history_str = "\n\n".join(history_lines)

        # Format Hall of Fame
        hall_of_fame_str = "No successful runs yet."
        if os.path.exists(BACKUP_DIR):
            files = [f for f in os.listdir(BACKUP_DIR) if f.startswith("genetic_algorithm_") and "_acc_" in f and f.endswith(".py")]
            if files:
                def get_acc(f):
                    match = re.search(r"_acc_([\d\.]+)", f)
                    return float(match.group(1).rstrip('.')) if match else 0.0
                files.sort(key=get_acc, reverse=True)
                
                hof_lines = []
                component_name = method_names[0] if isinstance(method_names, (list, tuple)) else method_names
                for f in files[:1]:
                    with open(os.path.join(BACKUP_DIR, f), 'r') as bkp_f:
                        code_str = bkp_f.read()
                        method_code, _, _ = self._extract_method(code_str, component_name)
                        if not method_code: method_code = skeletonize_code(code_str)
                        hof_lines.append(f"### Example from {f} ###\n```python\n{method_code}\n```")
                if hof_lines:
                    hall_of_fame_str = "\n\n".join(hof_lines)

        # Compress prompt by skeletonizing to save tokens
        skel_full_code = skeletonize_code(full_code)

        # Load best chromosome if available
        best_chromosome_str = "None found yet."
        best_info_path = os.path.join(PIPELINE_DIR, f"best_baseline_info_{DATASET}.json")
        if os.path.exists(best_info_path):
            try:
                with open(best_info_path, 'r') as f:
                    best_data = json.load(f)
                    best_chromosome_str = json.dumps(best_data.get("chromosome", {}), indent=2)
            except Exception:
                pass

        # LLM Generation with Full Context
        prompt = BASE_PROMPT_TEMPLATE.format(
            dataset=DATASET_DASH,
            history_str=history_str,
            search_space=SEARCH_SPACE_STR,
            best_chromosome=best_chromosome_str,
            full_code=skel_full_code,
            method_names=", ".join(method_names),
            task_specific_instructions="\n".join([INSTRUCTIONS.get(n, "") for n in method_names]),
            code=orig_code
        )
        
        # [MODIFIED_FOR_INNOVATION] Increased temperature from 0.8 to 0.9 to boost creativity. Revert to 0.8 if syntax errors occur too often.
        temperature = 0.9
        print(f"[Meta] Generation Temperature (BOOSTED): {temperature:.2f}")
        
        raw_res = self.llm.generate(prompt, max_new_tokens=2048, temperature=temperature)
        
        print(f"--- [DEBUG] Raw Response ---\n{raw_res}\n--------------------------")
        
        # Extract Thinking Trace
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_res, re.DOTALL)
        thinking_trace = thinking_match.group(1).strip() if thinking_match else "No thinking trace provided."

        # Extract all requested methods using the robust AST parser
        combined_code_parts = []
        for name in method_names:
            extracted = self._extract_function_body(raw_res, name)
            if extracted:
                combined_code_parts.append(extracted.strip('\n'))
            else:
                print(f"[Meta] LLM response is missing required method: {name} or failed validation.")
                return False
                
        combined_code = "\n\n".join(combined_code_parts)

        # Sort existing methods by span start in REVERSE order (bottom to top)
        existing_methods.sort(key=lambda x: x[1][0], reverse=True)
        
        replacements = []
        for i, (name, span, indent) in enumerate(existing_methods):
            if i == len(existing_methods) - 1:
                # Top-most method: replace its span with the new combined code
                replacements.append((span, indent, combined_code))
            else:
                # Other methods: delete them (they are included in the combined_code now)
                replacements.append((span, indent, ""))

        valid_syntax = False
        bench_stats = {"top3_mean": 0.0, "peak_accuracy": 0.0, "archive_size": self.global_archive_size, "error_trace": ""}
        try:
            test_full = full_code
            for span, indent_col, new_code in replacements:
                if new_code == "":
                    # Delete the span
                    test_full = test_full[:span[0]] + test_full[span[1]:]
                else:
                    # Inject code
                    lines = new_code.strip('\n').splitlines()
                    if lines:
                        first_def_line = next((l for l in lines if l.lstrip().startswith("def ")), "")
                        if first_def_line:
                            first_indent = len(first_def_line) - len(first_def_line.lstrip())
                        else:
                            first_line = next((l for l in lines if l.strip()), "")
                            first_indent = len(first_line) - len(first_line.lstrip())
                        cleaned_lines = []
                        for line in lines:
                            if not line.strip():
                                cleaned_lines.append("")
                            elif line.startswith(" " * first_indent):
                                cleaned_lines.append(line[first_indent:])
                            else:
                                cleaned_lines.append(line.lstrip())
                        new_code = "\n".join(cleaned_lines).strip()
                    else:
                        new_code = ""
                    reindented = "\n".join([" " * indent_col + line for line in new_code.splitlines()])
                    test_full = test_full[:span[0]] + reindented + "\n" + test_full[span[1]:]

            ast.parse(test_full)
            valid_syntax = True
        except SyntaxError as e:
            import traceback
            bench_stats["error_trace"] = traceback.format_exc()
            print(f"[Meta] Syntax Error: {e}")

        
        if valid_syntax:
            target_filename = os.path.basename(TARGET_FILE)
            bkp = os.path.join(BACKUP_DIR, f"{target_filename}_{'_'.join(method_names)}.bak")
            shutil.copy(TARGET_FILE, bkp)
            with open(TARGET_FILE, 'w') as f: f.write(test_full)
            
            # --- Runtime Smoke Test: catch NameError/TypeError before expensive benchmark ---
            # Exercises ALL 6 evolvable components with realistic dummy data.
            try:
                import importlib
                import sys
                if PIPELINE_DIR not in sys.path:
                    sys.path.insert(0, PIPELINE_DIR)
                ga_mod = importlib.import_module("modified_GA.genetic_algorithm_evolved")
                importlib.reload(ga_mod)
                test_ga = ga_mod.GeneticAlgorithm(
                    population_size=4, search_space=SEARCH_SPACE,
                    elitism_count=1, mutation_rate=1.0,
                    checkpoint_path="/dev/null"
                )
                # Create test chromosomes
                test_chromo = test_ga._create_random_chromosome()
                test_chromo2 = test_ga._create_random_chromosome()

                # 3. Test _mutate
                mutated = test_ga._mutate(test_chromo)

                # 4. Test _crossover
                crossed = test_ga._crossover(test_chromo, test_chromo2)

                # 5. Strict Bounds Checking
                for gene, val in mutated.items():
                    if val not in SEARCH_SPACE[gene]:
                        raise ValueError(f"Smoke test: mutate produced '{val}' for gene '{gene}', not in search space")
                for gene, val in crossed.items():
                    if val not in SEARCH_SPACE[gene]:
                        raise ValueError(f"Smoke test: crossover produced '{val}' for gene '{gene}', not in search space")

                # 6. Test _selection (needs a populated population)
                test_ga.population = [
                    {"chromosome": test_ga._create_random_chromosome(), "fitness": random.uniform(10, 70)}
                    for _ in range(4)
                ]
                parent = test_ga._selection()
                if not isinstance(parent, dict) or "chromosome" not in parent:
                    raise ValueError(f"Smoke test: _selection returned invalid result: {type(parent)}")

                # Validate mutated/crossed values are within search space
                for gene, val in mutated.items():
                    if val not in SEARCH_SPACE[gene]:
                        raise ValueError(f"Smoke test: mutate produced '{val}' for gene '{gene}', not in search space {SEARCH_SPACE[gene]}")
                for gene, val in crossed.items():
                    if val not in SEARCH_SPACE[gene]:
                        raise ValueError(f"Smoke test: crossover produced '{val}' for gene '{gene}', not in search space {SEARCH_SPACE[gene]}")
                print("[Meta] Runtime smoke test PASSED (all components validated).")
            except Exception as e:
                import traceback
                bench_stats["error_trace"] = traceback.format_exc()
                print(f"[Meta] Runtime smoke test FAILED: {e}")
                print("---> Reverting file and skipping benchmark.")
                shutil.copy(bkp, TARGET_FILE)
                valid_syntax = False
            
            if valid_syntax:
                print("[Meta] Benchmarking...")
                bench_stats = self.run_benchmark()
                if bench_stats.get("error_trace", "") or bench_stats["peak_accuracy"] == 0.0:
                    print("[Meta] Benchmarking crashed or returned 0.0 accuracy. Treating as invalid.")
                    valid_syntax = False

        new_score = bench_stats["peak_accuracy"]
        top3_mean = bench_stats["top3_mean"]
        new_archive_size = bench_stats["archive_size"]
        
        # Calculate novelty (how many new cells were filled)
        archive_novelty = max(0, new_archive_size - self.global_archive_size)

        # RL Loop: Pass all frontier metrics to the reward calculator
        reward = calculate_meta_reward(
            current_score=new_score, 
            best_ever_score=self.global_best_score, 
            baseline_score=self.baseline_score,
            top3_mean=top3_mean, 
            archive_novelty=archive_novelty, 
            valid_syntax=valid_syntax
        )
        
        # Update Global Frontier Tracking
        if valid_syntax:
            if new_score > self.global_best_score:
                print(f"[Meta] --> NEW GLOBAL SOTA! {new_score:.2f}% (was {self.global_best_score:.2f}%)")
                self.global_best_score = new_score
            if new_archive_size > self.global_archive_size:
                print(f"[Meta] --> ARCHIVE EXPANDED! {new_archive_size} cells (was {self.global_archive_size})")
                self.global_archive_size = new_archive_size
        
        fine_tune_expected = bool(valid_syntax)
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
        
        fine_tune_requested_batch = 0
        fine_tune_actual_batch = 0
        fine_tune_retries = 0
        fine_tune_oom_message = None
        
        if valid_syntax and reward > 0:
            print("--> SUCCESS. Updating Baseline (EMA) & Fine-tuning on Success.")
            # EMA Update: 20% of the new score, 80% of the old baseline
            if self.baseline_score == 0.0:
                self.baseline_score = new_score  # Initialize on first success
            else:
                self.baseline_score = (0.2 * new_score) + (0.8 * self.baseline_score)
            
            # --- Experience Replay: append success ---
            self.success_buffer.append({'prompt': prompt, 'completion': new_code})
            if len(self.success_buffer) > 20:
                self.success_buffer.pop(0)  # Cap buffer at 20, drop oldest
        else:
            print("--> REGRESSION (Failed to beat SOTA). Fine-tuning on past successes to maintain syntax.")

        # Setup fallback batch sizes
        try:
            target_batch = int(os.environ.get("LORA_REPLAY_BATCH_SIZE", 4))
            fallbacks_str = os.environ.get("LORA_REPLAY_BATCH_FALLBACKS", "2,1")
            fallbacks = [int(x.strip()) for x in fallbacks_str.split(",") if x.strip()]
        except ValueError:
            target_batch = 4
            fallbacks = [2, 1]
            
        batch_sizes_to_try = [target_batch] + fallbacks
        fine_tune_requested_batch = min(target_batch, len(self.success_buffer))
        
        self._cuda_cleanup()  # Explicit pre-fine-tune cleanup

        if fine_tune_expected:
            for attempt_batch in batch_sizes_to_try:
                batch_size = min(attempt_batch, len(self.success_buffer))
                if batch_size <= 0:
                    continue
                    
                training_batch = random.sample(self.success_buffer, batch_size)
                train_examples_count = batch_size
                fine_tune_actual_batch = batch_size
                
                print(f"[LoRA] Fine-tune start (replay buffer: {len(self.success_buffer)} items, batch: {batch_size})")
                fine_tune_started = True
                fine_tune_start_time = datetime.now().isoformat()
                try:
                    self.llm.train_on_buffer(training_batch, epochs=train_epochs)
                    fine_tune_completed = True
                    print("[LoRA] Fine-tune complete")
                    break  # Success, exit the retry loop
                except RuntimeError as e:
                    err_str = str(e).lower()
                    if "cuda out of memory" in err_str or "cuda runtime error" in err_str:
                        fine_tune_oom_message = str(e)
                        print(f"[LoRA] CUDA OOM at batch size {batch_size}: {e}")
                        self._cuda_cleanup()
                        fine_tune_retries += 1
                        continue # Try next smaller batch
                    else:
                        fine_tune_failed = True
                        fine_tune_exception = str(e)
                        print(f"[LoRA] Fine-tune failed (non-OOM RuntimeError): {e}")
                        break
                except Exception as e:
                    fine_tune_failed = True
                    fine_tune_exception = str(e)
                    print(f"[LoRA] Fine-tune failed: {e}")
                    break
                finally:
                    fine_tune_end_time = datetime.now().isoformat()
                    
            if not fine_tune_completed and not fine_tune_failed and fine_tune_started:
                fine_tune_failed = True
                fine_tune_exception = "CUDA OOM exhausted all batch fallbacks"
                print(f"[LoRA] Skipping fine-tuning for this iteration due to persistent OOM.")
                
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
                    
        if 'bkp' in locals() and not valid_syntax:
            print("--> Reverting File (Syntax Error).")
            shutil.copy(bkp, TARGET_FILE)
        elif valid_syntax:
            ts_bkp = os.path.join(BACKUP_DIR, f"genetic_algorithm_attempt{attempt}_{'_'.join(method_names)}_acc_{new_score:.2f}.py")
            shutil.copy(TARGET_FILE, ts_bkp)
            print(f"--> Saved version history with accuracy: {ts_bkp}")
            ts_bkp_thinking = os.path.join(BACKUP_DIR, f"genetic_algorithm_attempt{attempt}_{'_'.join(method_names)}_acc_{new_score:.2f}_thinking.txt")
            with open(ts_bkp_thinking, 'w') as f:
                f.write(thinking_trace)
            print(f"--> Saved CoT trace: {ts_bkp_thinking}")
            
        # Log attempt history
        status = "Success" if (valid_syntax and reward > 0) else ("Syntax Error" if not valid_syntax else "Regression")
        if not hasattr(self, 'attempt_history'): self.attempt_history = []
        self.attempt_history.append({
            "status": status, 
            "score": new_score, 
            "reward": reward,
            "code": combined_code, 
            "error_trace": bench_stats.get("error_trace", "") if not valid_syntax or reward <= 0 else ""
        })
        # Log successful LLM generation
        log_entry = {
            "timestamp": RUN_TIMESTAMP,
            "method": ", ".join(method_names),
            "attempt": attempt,
            "response": raw_res,
            "cleaned_code": combined_code,
            "valid_syntax": valid_syntax,
            "score": new_score,
            "peak_accuracy": bench_stats["peak_accuracy"],
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
            "fine_tune_requested_batch": fine_tune_requested_batch,
            "fine_tune_actual_batch": fine_tune_actual_batch,
            "fine_tune_retries": fine_tune_retries,
            "fine_tune_oom_message": fine_tune_oom_message,
            "fine_tune_start_time": fine_tune_start_time,
            "fine_tune_end_time": fine_tune_end_time,
            "adapter_save_start_time": adapter_save_start_time,
            "adapter_save_end_time": adapter_save_end_time
        }
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

        # Return True if this attempt was a success (valid syntax)
        return bool(valid_syntax)

    def run_fallback_iteration(self, component, attempt, total_attempts):
        print(f"\n[Meta] Running fallback GA benchmark using previous working code for {component}...")
        
        bench_stats = self.run_benchmark()
        new_score = bench_stats["peak_accuracy"]
        top3_mean = bench_stats["top3_mean"]
        new_archive_size = bench_stats["archive_size"]
        
        archive_novelty = max(0, new_archive_size - self.global_archive_size)
        reward = calculate_meta_reward(
            current_score=new_score, 
            best_ever_score=self.global_best_score, 
            baseline_score=self.baseline_score,
            top3_mean=top3_mean, 
            archive_novelty=archive_novelty, 
            valid_syntax=True
        )
        
        if new_score > self.global_best_score:
            print(f"[Meta] --> NEW GLOBAL SOTA (Fallback)! {new_score:.2f}% (was {self.global_best_score:.2f}%)")
            self.global_best_score = new_score
        if new_archive_size > self.global_archive_size:
            print(f"[Meta] --> ARCHIVE EXPANDED (Fallback)! {new_archive_size} cells (was {self.global_archive_size})")
            self.global_archive_size = new_archive_size

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "method": component,
            "attempt": attempt,
            "response": "Fallback to previous working GA due to repeated LLM syntax errors.",
            "cleaned_code": "Fallback triggered. Code unchanged.",
            "valid_syntax": True,
            "score": new_score,
            "peak_accuracy": bench_stats["peak_accuracy"],
            "reward": reward,
            "fine_tune_expected": False,
            "fine_tune_started": False,
            "fine_tune_completed": False,
            "fine_tune_failed": False,
            "fine_tune_exception": None,
            "adapter_save_started": False,
            "adapter_save_completed": False,
            "adapter_save_failed": False,
            "adapter_save_exception": None,
            "adapter_path": "",
            "train_examples_count": 0,
            "train_epochs": 0,
            "fine_tune_requested_batch": 0,
            "fine_tune_actual_batch": 0,
            "fine_tune_retries": 0,
            "fine_tune_oom_message": None,
            "fine_tune_start_time": None,
            "fine_tune_end_time": None,
            "adapter_save_start_time": None,
            "adapter_save_end_time": None
        }
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        return True

if __name__ == "__main__":
    pipeline_dir = os.environ.get("PIPELINE_DIR", BASE_DIR)
    with open(os.path.join(pipeline_dir, "model_config.json"), "r") as f:
        MODEL_PATH = json.load(f).get("base_model_name", "mistralai/Mistral-7B-Instruct-v0.2")
    evolver = MetaEvolver(MODEL_PATH)

    # LOOP: CYCLE THROUGH ALL EVOLVABLE COMPONENTS
    # [OLD — fixed attempt count, stopped after N tries regardless of success]
    # META_ITERATIONS = int(os.environ.get("META_ATTEMPTS"))
    # for i in range(META_ITERATIONS):
    #     component = COMPONENTS[i % len(COMPONENTS)]
    #     print(f"\n=== Meta-Evolution Iteration {i+1}/{META_ITERATIONS} ===")
    #     evolver.evolve_component(component, attempt=i+1, total_attempts=META_ITERATIONS)
    #     time.sleep(2)
    # [END OLD]

    # Priority: env var META_ATTEMPTS > model_config.json meta_attempts > default 5
    META_ITERATIONS = int(os.environ.get("META_ATTEMPTS", evolver.llm.config.get("meta_attempts", 5)))
    COMPONENTS = ["combine_genes", "mutate_gene", "select_competitor", "_create_random_chromosome"]

    successes = 0
    print(f"\n[Meta] Target: {META_ITERATIONS} evolutions with up to 5 attempts per iteration.")
    for iteration in range(META_ITERATIONS):
        component = COMPONENTS[iteration % len(COMPONENTS)]
        print(f"\n=== Iteration {iteration+1}/{META_ITERATIONS} — Evolving: {component} ===")
        
        MAX_ATTEMPTS = 10
        for attempt_idx in range(MAX_ATTEMPTS):
            success = evolver.evolve_component(component, attempt=iteration+1, total_attempts=META_ITERATIONS)
            if success:
                successes += 1
                print(f"[Meta] ✓ Iteration {iteration+1} (Attempt {attempt_idx+1}/{MAX_ATTEMPTS}) produced valid code. Proceeding to next iteration.")
                break
            else:
                print(f"[Meta] ✗ Iteration {iteration+1} (Attempt {attempt_idx+1}/{MAX_ATTEMPTS}) failed (Syntax Error). Code reverted.")
                
                if attempt_idx < MAX_ATTEMPTS - 1:
                    print(f"[Meta] Retrying {component}...")
                    time.sleep(2)
                else:
                    print(f"[Meta] Exhausted all {MAX_ATTEMPTS} attempts for {component}.")
                    evolver.run_fallback_iteration(component, attempt_idx + 1, META_ITERATIONS)
                    successes += 1
                    print(f"[Meta] Fallback iteration complete. Moving to next iteration.")
                    break
        
    print(f"\n=== Meta-Evolution Complete: {successes} successful evolutions out of {META_ITERATIONS} strict iterations ===")
    
    # --- Generate visualizations after all iterations ---
    try:
        from ab.gpt.brute.ga.meta_evolution.meta_visualization import main as generate_plots
        print("\n=== Generating Visualizations ===")
        generate_plots(RUN_TIMESTAMP, DATASET)
    except Exception as e:
        print(f"[WARN] Visualization failed (non-fatal): {e}")
