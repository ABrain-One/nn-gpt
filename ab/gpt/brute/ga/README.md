# Neural Architecture Search with Genetic Algorithm (NAS-GA)

This project implements a Neural Architecture Search (NAS) system using a Genetic Algorithm (GA) to automatically discover effective neural network architectures for image classification tasks, specifically targeting the CIFAR-10 dataset.

## Overview

The goal is to evolve the structure and hyperparameters of an AlexNet-like Convolutional Neural Network (CNN) to maximize its accuracy on CIFAR-10. The search is guided by a predefined `SEARCH_SPACE` that defines possible values for network layers (e.g., number of filters, kernel sizes) and training hyperparameters (e.g., learning rate, dropout).

The workflow consists of:
1.  **Initialization:** A population of random network architectures (chromosomes) is created based on the search space.
2.  **Evaluation:** Each unique architecture is built, trained for a few epochs (`NUM_EPOCHS_PER_EVAL`), and its validation accuracy is calculated. This accuracy serves as the "fitness" score. To optimize efficiency, architectures that have been previously evaluated (duplicates) are identified via a checksum of their code and skipped.
3.  **Evolution:** Using the fitness scores, the GA applies selection, crossover, and mutation operators to create the next generation of architectures. A number of the best-performing architectures (elitism) are carried over unchanged.
4.  **Iteration:** Steps 2 and 3 are repeated for a specified number of generations (`NUM_GENERATIONS`).
5.  **Output:** The process identifies the best architecture found. The code for this champion model and all unique evaluated architectures are saved. Per-epoch accuracy statistics for each evaluation are also recorded.

## File Structure

*   `AlexNet_evolvable.py`: Defines the CNN model (`Net`), the search space (`SEARCH_SPACE`), and a function to generate Python code strings for architectures.
*   `genetic_algorithm.py`: Implements the core Genetic Algorithm logic (`GeneticAlgorithm` class).
*   `run_evolution.py`: The main script to configure and run the NAS experiment. It handles data loading, defines the fitness function (including training/evaluation, duplicate checking, and saving results), and orchestrates the GA.
*   `ga_architecture/` (created during run): Stores the Python code (`.py` files) for each unique architecture evaluated, named `ga-alexnet-X.py`.
*   `stats/` (created during run): Stores evaluation statistics. For each unique architecture `ga-alexnet-X`, a subfolder `img-classification_cifar-10_acc_ga-alexnet-X` is created containing per-epoch accuracy files named `0.json`, `1.json`, etc.
*   `ga-champ-alexnet.py`: The Python code for the best-performing architecture found during the search.
*   `ga_evolution_checkpoint.pkl`: Checkpoint file to resume the GA from the last completed generation if interrupted.

## How to Run

1.  **Install Dependencies:**
    Ensure you have created anad acivated a virtual environment in the root directory (as specified in readme in root directory) and installed required packages. In case if you are creating a new environment install these packages using pip to run the script:
    ```bash
    pip install torch torchvision tqdm
    ```

2.  **Run the NAS Experiment:**
    Execute the main script from the command line within the project directory:
    ```bash
    python run_evolution.py
    ```
    This will start the evolution process with the parameters defined in `run_evolution.py` (e.g., Population=50, Generations=25, 5 epochs per evaluation). The script will automatically download the CIFAR-10 dataset if needed.

**NOTE**

Running the script again will not intelligently skip architectures that were evaluated in previous runs. It will likely re-evaluate many of them and overwrite the output files (architecture .py files, stats .json files, and the champion file) from the previous runs.