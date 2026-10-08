#run_mut_evolution.py:
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json
from uuid import uuid4 as uuid4_orig
import time
import hashlib
from MutNet_evolvable import Net, SEARCH_SPACE, generate_model_code_string
from genetic_algorithm import GeneticAlgorithm
from ab.gpt.util.eval.Eval import Eval

# Config
POPULATION_SIZE = 60
NUM_GENERATIONS = 50
MUTATION_RATE = 0.15
ELITISM_COUNT = 5
CHECKPOINT_FILE = 'ga_evolution_checkpoint.pkl'
BATCH_SIZE = 128
NUM_EPOCHS_PER_EVAL = 5

ARCHITECTURE_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'ga_architecture')
STATS_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'stats')
CHAMPION_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'ga-champ-mut.py')

os.makedirs(ARCHITECTURE_SAVE_DIR, exist_ok=True)
os.makedirs(STATS_SAVE_DIR, exist_ok=True)

seen_checksums = set()
architecture_counter = 0

def uuid4(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def fitness_function(chromosome: dict) -> float:
    global architecture_counter

    if not any(chromosome.get(f'include_conv{i}', 0) for i in range(1, 6)):
        print("  - No conv layers included → fitness = 0")
        return 0.0

    try:
        model_code_string = generate_model_code_string(chromosome)
    except Exception as e:
        print(f"  - Code gen error: {e} → fitness = 0")
        return 0.0

    model_checksum = uuid4(model_code_string)
    if model_checksum in seen_checksums:
        print(f"  - Duplicate (checksum: {model_checksum[:8]}) → skip")
        return 0.0

    try:
        print(f"  - Evaluating unique arch (checksum: {model_checksum[:8]}...)")
        
        current_arch_number = architecture_counter
        model_base_name = f"ga-mut-{current_arch_number}"
        
        arch_filepath = os.path.join(ARCHITECTURE_SAVE_DIR, f"{model_base_name}.py")
        try:
            with open(arch_filepath, 'w') as f:
                f.write(model_code_string)
            print(f"  - Saved architecture to: {arch_filepath}")
            architecture_counter += 1
        except Exception as e:
            print(f"  - Failed to save arch: {e}")
            return 0.0

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
                final_accuracy = float(result[1]) * 100
            elif isinstance(result, float):
                final_accuracy = result * 100
            elif result is not None:
                try:
                    final_accuracy = float(result) * 100
                except:
                    pass
            
            print(f"  - Eval result: {final_accuracy:.2f}%")
            seen_checksums.add(model_checksum)
            return final_accuracy

        except Exception as e:
            print(f"  - Evaluation error (Eval): {e} → fitness = 0")
            return 0.0

    except Exception as e:
        print(f"  - Evaluation error: {e} → fitness = 0")
        return 0.0

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    print(f"Input shape: (3, 32, 32), Output: 10 classes")

    # Resume architecture counter
    try:
        existing = [f for f in os.listdir(ARCHITECTURE_SAVE_DIR) if f.startswith("ga-mut-") and f.endswith(".py")]
        if existing:
            nums = []
            for f in existing:
                try:
                    num = int(f.replace("ga-mut-", "").replace(".py", ""))
                    nums.append(num)
                except:
                    pass
            if nums:
                architecture_counter = max(nums) + 1
                print(f"Resumed architecture counter: {architecture_counter}")
    except Exception as e:
        print(f"Could not resume counter: {e}")

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE,
        elitism_count=ELITISM_COUNT,
        mutation_rate=MUTATION_RATE,
        checkpoint_path=CHECKPOINT_FILE
    )

    best = ga.run(num_generations=NUM_GENERATIONS, fitness_function=fitness_function)

    print("\n--- Evolution Complete ---")
    if best:
        print(f"Best fitness: {best['fitness']:.2f}%")
        champion_code = generate_model_code_string(best['chromosome'])
        with open(CHAMPION_SAVE_PATH, 'w') as f:
            f.write(champion_code)
        print(f"Champion saved to: {CHAMPION_SAVE_PATH}")
    else:
        print("No valid model found.")