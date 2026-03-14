from .mutation import MutationStrategy
from .llm_loader import LocalLLMLoader as LLMLoader
from .rl_rewards import evaluate_code_and_reward
import random
import time
import os
import re
import datetime

# --- TARGETED MICRO-MUTATION PROMPTS ---
PROMPTS = {
    "mutate_transform_block": (
        "You are an expert. Rewrite the 'self.conv' definition in FractalBlock.\n"
        "GOAL: Change the layer structure (e.g. use Dilated Conv, Separable Conv, or different Kernel Size).\n"
        "CONSTRAINT: Must accept 'channels' and return 'channels' (preserve dimensions).\n"
        "Input Code:\n{slot_code}\n\n"
        "History of previous attempts:\n{history}\n\n"
        "Based on the history, write BETTER code than before. Output ONLY the code block.\n" 
    ),
    "mutate_optimizer": (
        "You are an expert. Rewrite the 'train_setup' method in Net class.\n"
        "GOAL: Change the optimizer (e.g. use Adam, RMSprop, SGD with different momentum).\n"
        "Input Code:\n{slot_code}\n\n"
        "History of previous attempts:\n{history}\n\n"
        "Based on the history, write BETTER code than before. Output ONLY the code block.\n"
    ),
    "mutate_join": (
        "You are an expert. Rewrite the 'forward' method in FractalDropPath.\n"
        "GOAL: Change how paths are combined (e.g. changes in drop logic, or adding small noise).\n"
        "Input Code:\n{slot_code}\n\n"
        "History of previous attempts:\n{history}\n\n"
        "Based on the history, write BETTER code than before. Output ONLY the code block.\n"
    ),
}

class LLMMutationStrategy(MutationStrategy):
    def __init__(self, model_path, log_file="dataset/mutation_log.jsonl", q_table_path=None, use_quantization=True):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isabs(log_file):
            log_file = log_file 
        else:
            log_file = os.path.join(base_dir, log_file)
        
        # ensure dirs exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "adapter_history"), exist_ok=True)

        adapter_path = os.path.join(base_dir, "fine_tuned_adapter")
        
        # --- STARTUP BACKUP ---
        # If we have a previous brain, save it to history before we touch it
        if os.path.exists(adapter_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(base_dir, "adapter_history", f"backup_startup_{timestamp}")
            print(f"[RL-GA] Backing up previous adapter to {backup_path}")
            # Use shutil to copy directory
            import shutil
            shutil.copytree(adapter_path, backup_path)
        
        self.llm_loader = LLMLoader(model_path, use_quantization, adapter_path=adapter_path)
        

        
        self.mutation_history = []
        self.training_buffer = []
        self.best_fitness_seen = 0.0
        self.improved_since_last_checkpoint = False
        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_file


    def _extract_slot(self, code, action):
        """Locates the specific code block to mutate based on the action."""
        target_code = None
        pattern = None
        
        if action == "mutate_transform_block":
            # Target: self.conv = ... inside FractalBlock
            pattern = re.compile(r"class FractalBlock.*?def __init__.*?self\.conv\s*=\s*(.*?)\n\s*def", re.DOTALL)
            # This regex is tricky. Let's rely on finding "self.conv =" inside FractalBlock
            # Simpler approach: Locate FractalBlock class, then find self.conv line
            pass # Use simple string find for now or assumes standard formatting

        # BETTER APPROACH: Use pre-defined markers in the seed code or standard method names
        # For this prototype, we'll try to find method definitions using simple string parsing
        
        start_idx = -1
        end_idx = -1
        
        if action == "mutate_transform_block":
            # Find FractalBlock class
            class_start = code.find("class FractalBlock")
            if class_start == -1: return None, None
            
            # Find self.conv = 
            target_str = "self.conv ="
            start_in_class = code.find(target_str, class_start)
            if start_in_class == -1: return None, None
            
            # Assume it ends at the next "self." or newline if it's simple
            # Actually, to be safe, let's mutate the ENTIRE definition of specific methods
            pass
            
        # --- ROBUST IMPLEMENTATION ---
        # We will extract specific methods:
        # 1. FractalBlock.__init__ (or distinct part) -> difficult without AST
        # 2. Let's target specific METHODS that we know exist
        
        method_name = ""
        if action == "mutate_transform_block":
            method_name = "def _make_conv" # We might need to adjust FractalNet to have this helper
            # Fallback: Just look for self.conv = nn.Sequential(...)
            # Let's target the definition of the block: 'self.conv'
            start_marker = "self.conv = "
            end_marker = "self.bn ="
        
        elif action == "mutate_optimizer":
            method_name = "def train_setup"
            
        elif action == "mutate_join":
            method_name = "def forward" # Inside FractalDropPath
            
        # Helper to extract method body
        if method_name:
            if action == "mutate_join":
                # Need to find FractalDropPath class first
                class_idx = code.find("class FractalDropPath")
                start_rx = code.find(method_name, class_idx)
            elif action == "mutate_optimizer":
                class_idx = code.find("class Net")
                start_rx = code.find(method_name, class_idx)
            else: # Default searches globally or we need to constrain
                start_rx = code.find(method_name)
                
            if start_rx == -1: return None, None
            
            # Find indentation to determine end
            # This is fragile text processing; AST would be better, but we stick to text for now
            # Assume method ends when indentation returns to class level (4 spaces) or method level
            
            # Hack: Extract ~20 lines or until next 'def '
            next_def = code.find("def ", start_rx + 10)
            if next_def == -1: next_def = len(code)
            
            # Refine extraction to be safer
            return code[start_rx:next_def], (start_rx, next_def)
            
        # Fallback for 'mutate_transform_block' (regex for self.conv = ...)
        if action == "mutate_transform_block":
            # searching for: self.conv = nn.Sequential(...)
            # We assume it matches parens.
            # Simplified: Find the line and following lines with indentation
            p = re.compile(r"(\s+self\.conv\s*=\s*.*?)(\n\s+self\.)", re.DOTALL)
            m = p.search(code)
            if m:
                return m.group(1), m.span(1)
                
        return None, None

    def _cleanup_llm_response(self, raw_text, action):
        """Extracts code block from markdown and ensures it's valid-ish."""
        code = raw_text
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        code = code.strip()
        
        # Add indentation correction if needed?
        # For now, assume LLM outputs clean function body or line
        return code

    def mutate(self, chromosome, search_space):
        # 1. Decide if we mutate
        if random.random() > self.mutation_rate:
            return chromosome
        
        full_code = chromosome['code']
        
        # 1. Select Action (Target Slot)
        current_fitness = chromosome.get('cached_fitness', 0.0)
        action_key = random.choice(list(PROMPTS.keys()))
        
        # 2. Extract Slot
        target_content, span = self._extract_slot(full_code, action_key)
        
        if not target_content:
            # print(f"[RL-GA] Extraction failed for {action_key}. Skipping.") # Silence for cleaner logs?
            return chromosome

        prompt_instr = PROMPTS[action_key]
        
        # 3. Construct Focused Prompt
        full_prompt = (
            f"### Task: {prompt_instr}\n"
            f"### Context (Original Code):\n```python\n{target_content}\n```\n"
            "### Instructions:\n"
            "1. Return ONLY the replacement code.\n"
            "2. Do NOT explain.\n"
            "3. Maintain indentation."
        )

        try:
            print(f"[RL-GA] Target: {action_key}")
            # Generate
            raw_response = self.llm_loader.generate(full_prompt, max_new_tokens=500, temperature=0.8)
            # print(f"[DEBUG RAW]: {raw_response[:200]}...")

            # Cleanup
            new_slot_code = self._cleanup_llm_response(raw_response, action_key)
            if len(new_slot_code) < 10: # Sanity check for empty output
                 return chromosome

            # 4. Inject (String Replacement)
            final_code = full_code[:span[0]] + new_slot_code + full_code[span[1]:]

            # 5. Evaluate
            print(f"[RL-GA] Evaluating {action_key}...")
            res = evaluate_code_and_reward(
                final_code,
                prm=chromosome, 
                log_file=self.log_file,
                prompt_used=full_prompt, 
                val_metric_baseline=current_fitness
            )
            
            reward = res.get('reward', 0.0)
            new_fitness = res.get('val_metric')
            if new_fitness is None: new_fitness = 0.0
            
            # Update Global Best & Save to History
            if new_fitness > self.best_fitness_seen:
                self.best_fitness_seen = new_fitness
                self.improved_since_last_checkpoint = True
                
                # Save Best Mutation to Folder (Overwrite single file for THIS run)
                try:
                    best_mut_dir = os.path.join(os.path.dirname(__file__), "best_mutations_history")
                    os.makedirs(best_mut_dir, exist_ok=True)
                    filename = f"best_mutation_{self.session_timestamp}.py"
                    
                    with open(os.path.join(best_mut_dir, filename), "w") as f:
                        f.write(f"# Best Fitness: {new_fitness}\n# Action: {action_key}\n# Date: {datetime.datetime.now()}\n\n")
                        f.write(f"'''\n Prompt Used:\n{full_prompt}\n'''\n\n")
                        f.write(f"# Generated Code:\n{new_slot_code}")
                except Exception as save_err:
                    print(f"[RL-GA] Failed to save best mutation: {save_err}")
            
            # --- ONLINE LEARNING Logic ---
            if reward > 0.0: # Only learn from successes (or at least valid code)
                # 1. In-Context History
                summary = "Modified code"
                if "kernel_size" in new_slot_code: summary = "Changed Kernel"
                elif "Adam" in new_slot_code: summary = "Used Adam"
                elif "dropout" in new_slot_code: summary = "Changed Dropout"
                
                self.mutation_history.append({
                    'action': action_key,
                    'summary': summary,
                    'reward': reward,
                    'fitness': new_fitness
                })
                
                # 2. LoRA Fine-Tuning Buffer
                self.training_buffer.append({
                    'prompt': full_prompt, 
                    'completion': new_slot_code
                })
                
                # 3. Trigger Training
                if len(self.training_buffer) >= 4: # Train every 4 successes
                    print(f"[LoRA-Online] Fine-tuning on {len(self.training_buffer)} samples...")
                    try:
                        self.llm_loader.train_on_buffer(self.training_buffer)
                        # Save Checklist
                        
                        # 1. Save to History (Only if we found something better)
                        if self.improved_since_last_checkpoint:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            history_path = os.path.join(os.path.dirname(__file__), "adapter_history", f"checkpoint_{timestamp}")
                            self.llm_loader.save_adapters(history_path)
                            self.improved_since_last_checkpoint = False
                            print(f"[LoRA] Saved History Checkpoint: {timestamp}")
                        
                        # 2. Save to Latest (Always)
                        adapter_path = os.path.join(os.path.dirname(__file__), "fine_tuned_adapter")
                        self.llm_loader.save_adapters(adapter_path)
                    except Exception as train_err:
                        print(f"[LoRA] Training failed: {train_err}")
                    self.training_buffer = []


            
            print(f"[RL-GA] Reward: {reward:.4f}, Acc: {new_fitness:.4f}")
            
            if reward > 0.0:
                mutated_ind = chromosome.copy()
                mutated_ind['code'] = final_code
                mutated_ind['cached_fitness'] = new_fitness
                return mutated_ind
            else:
                return chromosome

        except Exception as e:
            print(f"[RL-GA] Error: {e}")
            return chromosome
