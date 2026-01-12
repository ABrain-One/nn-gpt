import sys
import os
import torch

# Ensure we can import from your modular folder
sys.path.append(os.getcwd())

# Import the NEW Local Evaluator we just wrote
from ab.gpt.brute.ga.modular.rl_rewards import evaluate_fitness

# We need a fake "Individual" class because rl_rewards expects an object
class MockIndividual:
    def __init__(self, code, prm):
        self.code = code
        self.prm = prm

def debug_seed_execution():
    print("----------------------------------------------------------------")
    print("DEBUGGING: fractal_seed.py using Local Evaluator (rl_rewards.py)")
    print("----------------------------------------------------------------")

    # 1. Load the seed code
    path = os.path.join(os.path.dirname(__file__), "fractal_seed.py")
    if not os.path.exists(path):
        print("Error: Seed file not found.")
        return

    with open(path, 'r') as f:
        code_str = f.read()

    # 2. Mock parameters
    prm = {
        'lr': 0.01,
        'momentum': 0.9,
        # Any other hyperparameters your seed uses
    }
    
    # Create the mock individual
    ind = MockIndividual(code_str, prm)

    print("Attempting evaluate_fitness()...")
    print("This will download CIFAR-10 to ./data_v2 if missing.")
    
    try:
        # 3. Call the function exactly as the Genetic Algorithm will
        accuracy = evaluate_fitness(ind, train_conf=None)
        
        print("----------------------------------------------------------------")
        if accuracy > 0.0:
            print("SUCCESS!")
            print(f"Training Complete. Accuracy: {accuracy:.4f}")
            print("The pipeline is FIXED.")
        else:
            print("FAILURE.")
            print("Accuracy is 0.0. Check console for compile/training errors.")
        print("----------------------------------------------------------------")

    except Exception as e:
        print("----------------------------------------------------------------")
        print("CRITICAL CRASH")
        print(e)
        import traceback
        traceback.print_exc()
        print("----------------------------------------------------------------")

if __name__ == "__main__":
    debug_seed_execution()