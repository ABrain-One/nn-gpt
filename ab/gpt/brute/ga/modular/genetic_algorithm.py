import random
from tqdm import tqdm
import pickle
import os

class GeneticAlgorithm:
    def __init__(self, population_size, search_space, elitism_count, mutation_rate,
                 checkpoint_path='ga_checkpoint.pkl'):
        self.population_size = population_size
        self.search_space = search_space
        self.elitism_count = elitism_count
        self.mutation_rate = mutation_rate
        self.population = []
        self.checkpoint_path = checkpoint_path

    def _create_random_chromosome(self):
        chromosome = {}
        for key, values in self.search_space.items():
            chromosome[key] = random.choice(values)
        return chromosome

    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            chromosome = self._create_random_chromosome()
            self.population.append({'chromosome': chromosome, 'fitness': None})
        print(f"Initialized population with {self.population_size} individuals.")

    def _save_checkpoint(self, generation_num):
        state = {
            'generation': generation_num,
            'population': self.population
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        # print(f"--- Checkpoint saved for Generation {generation_num} ---")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
                print(f"--- Resuming from checkpoint at Generation {state['generation']} ---")
                return state['generation'], state['population']
            except:
                return 0, None
        return 0, None

    # --- START LLM: CROSSOVER ---
    def _crossover(self, parent1_chromo, parent2_chromo):
        """
        Standard Single-Point Crossover.
        The LLM may replace this with Two-Point, Uniform, or Uniform-with-Bias crossover.
        """
        child_chromo = {}
        genes = list(self.search_space.keys())
        
        # Logic: Pick a point, take first half from P1, second half from P2
        if len(genes) > 1:
            crossover_point = random.randint(1, len(genes) - 1)
            for i, gene in enumerate(genes):
                if i < crossover_point:
                    child_chromo[gene] = parent1_chromo[gene]
                else:
                    child_chromo[gene] = parent2_chromo[gene]
        else:
            child_chromo = parent1_chromo.copy()
            
        return child_chromo
    # --- END LLM: CROSSOVER ---

    # --- START LLM: MUTATION ---
    def _mutate(self, chromosome):
        """
        Standard Random Resetting Mutation.
        The LLM may replace this with Swap Mutation, Scramble Mutation, or Adaptive Mutation.
        """
        mutated_chromo = chromosome.copy()
        for gene in self.search_space.keys():
            if random.random() < self.mutation_rate:
                current_value = mutated_chromo[gene]
                possible_values = self.search_space[gene]
                
                # Exclude current value if possible to ensure actual change
                other_values = [v for v in possible_values if v != current_value]
                if other_values:
                    mutated_chromo[gene] = random.choice(other_values)
        return mutated_chromo
    # --- END LLM: MUTATION ---

    # --- START LLM: SELECTION ---
    def _selection(self):
        """
        Standard Tournament Selection (k=3).
        The LLM may replace this with Roulette Wheel, Rank Selection, or Boltzmann Selection.
        """
        # Ensure we have enough individuals for a tournament
        k = 3
        tournament_size = min(k, len(self.population))
        competitors = random.sample(self.population, tournament_size)
        
        # Pick the one with the highest fitness
        winner = max(competitors, key=lambda x: x['fitness'] if x['fitness'] is not None else -float('inf'))
        return winner
    # --- END LLM: SELECTION ---

    def run(self, num_generations, fitness_function):
        start_gen, loaded_population = self._load_checkpoint()
        if loaded_population is not None:
            self.population = loaded_population
        else:
            self._initialize_population()

        best_ever_individual = None
        
        # Track improvement for Meta-Score calculation
        fitness_history = []

        # If resuming, find current best
        if loaded_population is not None:
            evaluated = [ind for ind in self.population if ind['fitness'] is not None]
            if evaluated:
                best_ever_individual = max(evaluated, key=lambda x: x['fitness'])

        for gen in range(start_gen, num_generations):
            print(f"\n===== Generation {gen + 1}/{num_generations} =====")

            # Evaluate
            for individual in tqdm(self.population, desc="Evaluating"):
                if individual['fitness'] is None:
                    individual['fitness'] = fitness_function(individual['chromosome'])

            # Sort by fitness (descending)
            self.population.sort(key=lambda x: x['fitness'] if x['fitness'] is not None else -1, reverse=True)

            # Track best ever
            current_best = self.population[0]
            current_fitness = current_best['fitness'] if current_best['fitness'] is not None else 0
            
            if best_ever_individual is None or (
                current_fitness > (best_ever_individual['fitness'] if best_ever_individual['fitness'] else -1)
            ):
                best_ever_individual = current_best.copy()
                print(f"*** New best overall: {current_fitness:.4f} ***")

            print(f"Generation Best: {current_fitness:.4f}")
            fitness_history.append(current_fitness)

            # --- Next Generation Logic ---
            next_generation = []

            # 1. Elitism
            if self.elitism_count > 0:
                elites = self.population[:self.elitism_count]
                next_generation.extend(elites)

            # 2. Reproduction
            num_children = self.population_size - len(next_generation)
            if num_children > 0:
                for _ in range(num_children):
                    parent1 = self._selection()
                    parent2 = self._selection()
                    
                    child_chromo = self._crossover(parent1['chromosome'], parent2['chromosome'])
                    mutated_child = self._mutate(child_chromo)
                    
                    next_generation.append({'chromosome': mutated_child, 'fitness': None})

            self.population = next_generation
            self._save_checkpoint(generation_num=gen + 1)

        print("\n===== Evolution Complete =====")
        return best_ever_individual, fitness_history