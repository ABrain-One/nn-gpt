import os
import sys
import random
import datetime

# --- Import Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))

if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Module Imports ---
from .engine import GeneticAlgorithmEngine
from .selection import TournamentSelection
from .crossover import SinglePointCrossover
from .rl_mutation import RLLLMMutation 
from .rl_rewards import evaluate_fitness

# --- Load Seed ---
try:
    seed_path = os.path.join(current_dir, "fractal_seed.py")
    with open(seed_path, "r") as f:
        SEED_CODE = f.read()
except FileNotFoundError:
    print(f"Error: {seed_path} not found. Run gen_fractal_script.py first.")
    exit(1)

# --- Custom Crossover Strategy (CRITICAL FIX) ---
# Custom Crossover removed (Redundant: SinglePointCrossover handles code preservation)

# # --- Configuration ---
# POPULATION_SIZE = 4
# GENERATIONS = 50
# ELITISM_COUNT = 1
# USE_QUANTIZATION = True 
# MUTATION_RATE = 0.5 
# MODEL_PATH = "deepseek-ai/deepseek-coder-1.3b-instruct"

# --- Configuration --- according to cluster config
POPULATION_SIZE = int(os.environ.get("POPULATION_SIZE", 4))
GENERATIONS = int(os.environ.get("GENERATIONS", 5))
ELITISM_COUNT = 1
USE_QUANTIZATION = True 
MUTATION_RATE = float(os.environ.get("MUTATION_RATE", 0.5))
MODEL_PATH = "deepseek-ai/deepseek-coder-1.3b-instruct"
EPOCHS_PER_INDIVIDUAL = 1

# Search Space (Hyperparameters only)
SEARCH_SPACE = {
    "lr": [0.01, 0.005, 0.001],
    "momentum": [0.9, 0.8],
    "drop_path_prob": [0.1, 0.2],
    "dropout": [0.0, 0.1]
}

# --- Glue Code ---
class IndividualWrapper:
    def __init__(self, chromosome):
        self.code = chromosome.get('code', '')
        self.prm = chromosome

def fitness_function(chromosome):
    """
    Evaluation with a 'Safety Net'.
    If the engine wiped the seed, we put it back right here.
    """
    # SAFETY NET: If code is missing (due to random init), inject the Seed!
    if 'code' not in chromosome or not chromosome['code']:
        # print(">>> [System] Re-injecting Seed Code into empty individual")
        chromosome['code'] = SEED_CODE

    chromosome['epochs'] = EPOCHS_PER_INDIVIDUAL

    # Checkcache
    if chromosome.get('cached_fitness') is not None:
        print(f">>> Using Cached Fitness: {chromosome['cached_fitness']:.4f}")
        return chromosome['cached_fitness']
    
    wrapper = IndividualWrapper(chromosome)
    
    # print(">>> Evaluating Individual...")
    accuracy = evaluate_fitness(wrapper)
    
    return accuracy

def main():
    print("--- Starting Single-Loop FractalNet Evolution ---")
    
    # 1. Strategies
    selection = TournamentSelection(tournament_size=3)
    
    # FIX: Use Standard Crossover (it preserves code now)
    crossover = SinglePointCrossover() 
    
    print(f"Using LLM: {MODEL_PATH} (Quantized: {USE_QUANTIZATION})")
    mutation = RLLLMMutation(
        model_path=MODEL_PATH,
        mutation_rate=MUTATION_RATE,
        use_quantization=USE_QUANTIZATION # Add quantization arg which RLLLMMutation expects
    )

    engine = GeneticAlgorithmEngine(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE,
        elitism_count=ELITISM_COUNT,
        selection_strategy=selection,
        crossover_strategy=crossover, # Pass custom class
        mutation_strategy=mutation
    )

    # 2. Run Evolution
    # Note: We rely on fitness_function to inject the seed dynamically now,
    # so we don't need to manually inject before run() if run() wipes it anyway.
    best_ind = engine.run(num_generations=GENERATIONS, fitness_function=fitness_function)
    
    print("--- Evolution Finished ---")
    print(f"Best Fitness: {best_ind.fitness}")
    
    # 3. Save Results
    # best_code_path = os.path.join(current_dir, "best_fractal_net.py")
    # if best_ind.chromosome.get('code'):
    #     with open(best_code_path, "w") as f:
    #         f.write(best_ind.chromosome['code'])
    #     print(f"Saved best model to {best_code_path}")
    # 1. Define Paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = os.path.join(current_dir, "historicalBestFractal")
    
    # Ensure history folder exists
    os.makedirs(history_dir, exist_ok=True)
    
    # Filenames
    history_filename = f"best_fractal_net_{timestamp}.py"
    history_path = os.path.join(history_dir, history_filename)
    
    latest_path = os.path.join(current_dir, "best_fractal_net.py")

    # 2. Write Files
    code = best_ind.chromosome.get('code')
    if code:
        # Save timestamped version in /history/
        with open(history_path, "w") as f:
            f.write(code)
        print(f"Saved historical model to: {history_path}")

        # Overwrite 'latest' version in /modular/ (for validation scripts)
        with open(latest_path, "w") as f:
            f.write(code)
        print(f"Updated latest model at: {latest_path}")
    else:
        print("Error: Best individual has no code to save.")

if __name__ == "__main__":
    main()