from .mutation import MutationStrategy
from .llm_loader import LocalLLMLoader
from .rl_agent import RLAgent
from .rl_rewards import evaluate_code_and_reward
import random
import time
import os

# Define a set of prompts (Actions)
PROMPTS = {
    "optimize_accuracy": (
        "You are an expert. Improve this PyTorch code to achieve higher accuracy. "
        "You can change layer sizes, kernel sizes, or activation functions. "
        "Keep the class name 'Net'."
    ),
    "deepen_network": (
        "You are an expert. Add one more Convolutional layer to this PyTorch model "
        "to increase its capacity. Ensure dimensions match. Keep the class name 'Net'."
    ),
    "add_skip_connection": (
        "You are an expert. Add a Residual Skip Connection to this PyTorch model "
        "if possible, or improve gradient flow. Keep the class name 'Net'."
    ),
    "reduce_parameters": (
        "You are an expert. Optimize this PyTorch code to be more efficient "
        "(fewer parameters) while maintaining accuracy. Keep the class name 'Net'."
    ),
    "fix_bugs": (
        "You are an expert. Review this code for any potential bugs or shape mismatches "
        "and fix them. Return the corrected code. Keep the class name 'Net'."
    ),
    "fractal_expand": (
        "You are an expert. The user wants to deepen this FractalNet. "
        "Find a FractalBlock and increase its 'n_columns' by 1. "
        "Ensure you import FractalBlock from modular.fractalnet. "
        "Keep the class name 'Net'."
    ),
    "fractal_prune": (
        "You are an expert. The user wants to make this FractalNet more efficient. "
        "Find a FractalBlock and decrease its 'n_columns' by 1 (min 1). "
        "Ensure you import FractalBlock from modular.fractalnet. "
        "Keep the class name 'Net'."
    ),
    "convert_to_fractal": (
        "You are an expert. Convert this standard CNN into a FractalNet. "
        "Replace the main convolutional layers with a 'FractalBlock(n_columns=2, ...)' "
        "Define 'base_module_fn' to return the original Conv layer. "
        "Ensure you import FractalBlock from modular.fractalnet. "
        "Keep the class name 'Net'."
    ),
    "fractal_expand_depth": (
        "You are an expert. Increase the 'recursion_depth' variable in the Net class "
        "to make the FractalNet deeper. Ensure you update the Chromosome comment to match."
    ),
    "fractal_reduce_depth": (
        "You are an expert. Decrease the 'recursion_depth' variable in the Net class "
        "to make the network shallower and faster. Ensure you update the Chromosome comment."
    ),
    "tune_fractal_hyperparams": (
        "You are an expert. Adjust 'drop_path_prob' or 'dropout' in the Net class "
        "to improve regularization. Update the Chromosome comment."
    )
}

class RLLLMMutation(MutationStrategy):
    def __init__(self, mutation_rate, model_path, use_quantization=True, 
                 q_table_path=None, log_file=None):
        super().__init__(mutation_rate)
        
        if log_file is None:
            log_file = os.path.join(os.path.dirname(__file__), "dataset/mutation_log.jsonl")
        
        if q_table_path is None:
             q_table_path = os.path.join(os.path.dirname(__file__), "q_table.json")
        
        self.llm_loader = LocalLLMLoader(model_path, use_quantization)
        
        # Initialize RL Agent
        self.agent = RLAgent(
            actions=list(PROMPTS.keys()),
            q_table_path=q_table_path
        )
        
        self.log_file = log_file
        self.last_fitness = 0.0 # Track minimal state

    def get_state(self, fitness):
        """Discretize fitness into a state."""
        if fitness is None: return "fit_none"
        # Buckets: 0.0-0.2, 0.2-0.4, ...
        bucket = int(fitness * 5) 
        return f"fit_{bucket}"

    def mutate(self, chromosome, search_space):
        print(f"[DEBUG] RLLLM Mutation called. Rate: {self.mutation_rate}")
        # 1. Check mutation rate
        if random.random() > self.mutation_rate:
            print("[DEBUG] Mutation skipped (rng)")
            return chromosome
        
        # chromosome is expected to be a dict here
        code = chromosome.get('code')
        if not code:
            print("[DEBUG] No code in chromosome!")
            return chromosome

        # 2. Select Action (Prompt) via RL Agent
        # State representation: simplistic (just using last reward or constant)
        # For now, state = "generic"
        current_state = "unknown" # Todo: get better state
        action_key = self.agent.choose_action(current_state)
        prompt_template = PROMPTS[action_key]
        
        # 3. Construct Prompt
        # We wrap the code to provide context
        full_prompt = (
            f"{prompt_template}\n\n"
            f"```python\n{code}\n```\n\n"
            "Provide ONLY the updated Python code."
        )

        try:
            print(f"[RL-GA] Action: {action_key}")
            # Ensure generate interface matches new loader
            new_code = self.llm_loader.generate(full_prompt, max_new_tokens=2048)
            
            # Cleanup code
            if "```python" in new_code:
                new_code = new_code.split("```python")[1].split("```")[0].strip()
            elif "```" in new_code:
                new_code = new_code.split("```")[1].split("```")[0].strip()

            if new_code and "class Net" in new_code and "def forward" in new_code:
                
                # 2. Evaluate immediately to get Reward (for RL update)
                # Note: This is computationally expensive (double evaluation: here + GA loop)
                # But necessary for precise RL feedback on the *mutation step* itself.
                # Use quick evaluation (few batches).
                
                print("[RL-GA] Evaluating mutation reward...")
                res = evaluate_code_and_reward(
                    new_code,
                    prm=chromosome,  # Pass HPs for correct evaluation
                    log_file=self.log_file,
                    prompt_used=full_prompt, # Log for Fine-tuning
                    val_metric_baseline=0.1 # Should be previous fitness
                )
                
                reward = res.get('reward', 0.0)
                new_fitness = res.get('val_metric')
                if new_fitness is None: new_fitness = 0.0
                
                # 3. Update Agent
                next_state = self.get_state(new_fitness)
                self.agent.update(current_state, action_key, reward, next_state)
                self.agent.save()
                
                print(f"[RL-GA] Reward: {reward:.4f}, Acc: {new_fitness:.4f}")
                
                # Preserve other genes (momentum, lr, etc.)
                mutated_ind = chromosome.copy()
                mutated_ind['code'] = new_code
                mutated_ind['cached_fitness'] = new_fitness
                
                return mutated_ind
                
            else:
                print("[RL-GA] Invalid code generated.")
                # Penalty for invalid code?
                self.agent.update(current_state, action_key, -1.0, current_state)
                self.agent.save()

        except Exception as e:
            print(f"[RL-GA] Error: {e}")

        return chromosome
