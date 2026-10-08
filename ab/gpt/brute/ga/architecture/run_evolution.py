# run_evolution.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json
from ab.nn.util.Util import uuid4
import time

from AlexNet_evolvable import Net, SEARCH_SPACE, generate_model_code_string
from genetic_algorithm import GeneticAlgorithm
from ab.gpt.util.eval.Eval import Eval

POPULATION_SIZE = 50
NUM_GENERATIONS = 25
MUTATION_RATE = 0.15
ELITISM_COUNT = 5
CHECKPOINT_FILE = 'ga_evolution_checkpoint.pkl'

BATCH_SIZE = 128
NUM_EPOCHS_PER_EVAL = 5

ARCHITECTURE_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'nn')
STATS_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'train')
CHAMPION_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'ga-champ.py')
os.makedirs(ARCHITECTURE_SAVE_DIR, exist_ok=True)
os.makedirs(STATS_SAVE_DIR, exist_ok=True)

seen_checksums = set()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_subset, val_subset = random_split(full_train_set, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    in_shape = (3, 32, 32)
    out_shape = (10,)
    print(f"Input shape: {in_shape}, Output shape: {out_shape}")

    def fitness_function(chromosome: dict) -> float:
        try:
            model_code_string = generate_model_code_string(chromosome, in_shape, out_shape)
        except Exception as e:
            print(f"  - Error generating code string for chromosome {chromosome}: {e}. Assigning fitness 0.")
            return 0.0

        model_checksum = uuid4(model_code_string)

        if model_checksum in seen_checksums:
             print(f"  - Duplicate architecture detected (checksum: {model_checksum}). Skipping evaluation and saving.")
             return 0.0

        try:
            print(f"  - Evaluating unique architecture (checksum: {model_checksum[:8]}...)")
            
            model_base_name = f"ga-{model_checksum}"
            arch_filename = f"{model_base_name}.py"
            arch_filepath = os.path.join(ARCHITECTURE_SAVE_DIR, arch_filename)
            
            # Save architecture file immediately so Eval can read it
            try:
                with open(arch_filepath, 'w') as f:
                    f.write(model_code_string)
                # print(f"  - Unique architecture code saved to: {arch_filepath}")
            except Exception as save_error:
                print(f"  - Error saving architecture file {arch_filepath}: {save_error}")
                return 0.0

            # Prepare filtered parameters for Eval to avoid polluting stats
            eval_prm = {
                'lr': chromosome['lr'],
                'momentum': chromosome['momentum'],
                'dropout': chromosome['dropout'],
                'batch': BATCH_SIZE,
                'epoch': NUM_EPOCHS_PER_EVAL,
                'transform': "norm_256_flip"
            } 


            # Create specific stats directory for this individual to avoid collisions
            model_stats_dir_name = f"img-classification_cifar-10_acc_{model_base_name}"
            model_stats_dir_path = os.path.join(STATS_SAVE_DIR, model_stats_dir_name)
            os.makedirs(model_stats_dir_path, exist_ok=True)

            # Use Eval to evaluate
            evaluator = Eval(
                model_source_package=ARCHITECTURE_SAVE_DIR,
                task='img-classification',
                dataset='cifar-10',
                metric='acc',
                prm=eval_prm,
                save_to_db=True, # Required to generate JSON stats
                prefix=model_base_name,
                save_path=model_stats_dir_path
            )
            
            try:
                result = evaluator.evaluate(arch_filepath)
                
                if isinstance(result, dict) and 'accuracy' in result:
                    final_accuracy = result['accuracy'] * 100 
                elif isinstance(result, tuple) and len(result) >= 2:
                    # Result is (uid, accuracy, ...)
                    final_accuracy = float(result[1]) * 100
                elif isinstance(result, float):
                    final_accuracy = result * 100
                else:
                    final_accuracy = float(result) * 100 if result is not None else 0.0

                print(f"  - Evaluation Complete. Final Fitness: {final_accuracy:.2f}%")
                seen_checksums.add(model_checksum)
                return final_accuracy

            except Exception as eval_error:
                print(f"  - Error during Eval: {eval_error}. Assigning fitness 0.")
                return 0.0

        except Exception as e:
            print(f"  - Error evaluating chromosome: {e}. Assigning fitness 0.")
            return 0.0

    print("\n--- Starting Genetic Algorithm ---")

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE,
        elitism_count=ELITISM_COUNT,
        mutation_rate=MUTATION_RATE,
        checkpoint_path=CHECKPOINT_FILE
    )

    best_individual = ga.run(
        num_generations=NUM_GENERATIONS,
        fitness_function=fitness_function
    )

    print("\n--- Evolution Finished! ---")
    if best_individual:
        print("Best performing network architecture found:")
        print(f"  - Fitness (Validation Accuracy): {best_individual['fitness']:.2f}%")
        print("  - Chromosome (Parameters):")
        for gene, value in best_individual['chromosome'].items():
            print(f"    - {gene}: {value}")

        try:
            champion_code_string = generate_model_code_string(best_individual['chromosome'], in_shape, out_shape)
            with open(CHAMPION_SAVE_PATH, 'w') as f:
                f.write(champion_code_string)
            print(f"\n--- Champion architecture saved to: {CHAMPION_SAVE_PATH} ---")
        except Exception as champ_error:
            print(f"\n--- Error saving champion architecture: {champ_error} ---")

    else:
        print("No successful individual found in any generation (all had errors).")

    print("\nTo fully train this best model, you would now create a new Net with this")
    print("chromosome and train it for many more epochs (e.g., 50-100).")
