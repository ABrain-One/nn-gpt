import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import argparse
import sys
import hashlib
from MutNet_evolvable import Net, SEARCH_SPACE, generate_model_code_string
from genetic_algorithm import GeneticAlgorithm
from ab.gpt.util.eval.Eval import Eval

# --- CONFIGURATION (Defaults) ---
DEFAULT_POP_SIZE = 60
DEFAULT_GENS = 50
MUTATION_RATE = 0.15
ELITISM_COUNT = 5
BATCH_SIZE = 128
NUM_EPOCHS_PER_EVAL = 5

# Paths
ARCHITECTURE_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'ga_architecture')
STATS_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'stats')
CHECKPOINT_FILE = 'ga_evolution_checkpoint.pkl'

os.makedirs(ARCHITECTURE_SAVE_DIR, exist_ok=True)
os.makedirs(STATS_SAVE_DIR, exist_ok=True)

seen_checksums = set()
architecture_counter = 0

def uuid4(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def fitness_function(chromosome: dict) -> float:
    global architecture_counter

    # Quick heuristic check to save time
    if not any(chromosome.get(f'include_conv{i}', 0) for i in range(1, 6)):
        return 0.0

    try:
        model_code_string = generate_model_code_string(chromosome)
    except Exception:
        return 0.0

    model_checksum = uuid4(model_code_string)
    if model_checksum in seen_checksums:
        return 0.0

    # Architecture Saving
    current_arch_number = architecture_counter
    model_base_name = f"ga-mut-{current_arch_number}"
    arch_filepath = os.path.join(ARCHITECTURE_SAVE_DIR, f"{model_base_name}.py")
    
    try:
        with open(arch_filepath, 'w') as f:
            f.write(model_code_string)
        architecture_counter += 1
    except:
        return 0.0

    # Evaluation Params
    eval_prm = {
        'lr': chromosome['lr'],
        'momentum': chromosome['momentum'],
        'dropout': chromosome['dropout'],
        'batch': BATCH_SIZE,
        'epoch': NUM_EPOCHS_PER_EVAL,
        'transform': "norm_256_flip"
    }

    model_stats_dir_name = f"img-classification_cifar-10_acc_{model_base_name}"
    model_stats_dir_path = os.path.join(STATS_SAVE_DIR, model_stats_dir_name)
    os.makedirs(model_stats_dir_path, exist_ok=True)
    
    # Eval Wrapper
    try:
        evaluator = Eval(
            model_source_package=ARCHITECTURE_SAVE_DIR,
            task='img-classification',
            dataset='cifar-10',
            metric='acc',
            prm=eval_prm,
            save_to_db=False, # Disable DB for speed during meta-evolution
            prefix=model_base_name,
            save_path=model_stats_dir_path
        )
        
        # Suppress stdout during eval to keep meta-logs clean? 
        # Optional, but keeping it visible for now.
        result = evaluator.evaluate(arch_filepath)
        
        final_accuracy = 0.0
        if isinstance(result, dict) and 'accuracy' in result:
            final_accuracy = result['accuracy'] * 100
        elif isinstance(result, tuple) and len(result) >= 2:
            final_accuracy = float(result[1]) * 100
        elif isinstance(result, float):
            final_accuracy = result * 100
        
        seen_checksums.add(model_checksum)
        return final_accuracy

    except Exception as e:
        return 0.0

if __name__ == "__main__":
    # --- CLI ARGUMENTS FOR META-CONTROLLER ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=DEFAULT_GENS, help="Number of generations to run")
    parser.add_argument("--pop", type=int, default=DEFAULT_POP_SIZE, help="Population size")
    parser.add_argument("--clean", action="store_true", help="Ignore checkpoint and start fresh")
    args = parser.parse_args()

    # Clean start logic
    if args.clean and os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print("Removed existing checkpoint for clean start.")
        except: pass

    # Data Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ... (Dataset loading code kept brief, assumes data exists or downloads) ...
    # Note: For meta-evolution speed, consider using a smaller subset of CIFAR10 here if possible.

    try:
        # Initialize GA
        ga = GeneticAlgorithm(
            population_size=args.pop,
            search_space=SEARCH_SPACE,
            elitism_count=ELITISM_COUNT,
            mutation_rate=MUTATION_RATE,
            checkpoint_path=CHECKPOINT_FILE
        )

        # Run
        best, history = ga.run(num_generations=args.gens, fitness_function=fitness_function)

        # --- SCORE CALCULATION FOR META-EVOLVER ---
        # We need a single number to tell the LLM how good its code was.
        # Metric: Average Fitness of the Last Generation + Max Fitness Achieved
        
        if history:
            final_gen_score = history[-1]
            max_score = max(history)
            # A mix of stability (final) and peak performance (max)
            meta_score = (final_gen_score * 0.7) + (max_score * 0.3)
        else:
            meta_score = 0.0

        # This output format is CRITICAL. The Meta-Evolver uses Regex to find this line.
        print(f"META_SCORE: {meta_score:.4f}")

    except Exception as e:
        # If the generated GA code crashes (syntax error, logic error), return 0
        print(f"CRITICAL ERROR IN EVOLUTION: {e}")
        print("META_SCORE: 0.0")
        sys.exit(1)