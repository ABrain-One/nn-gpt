import pickle
import os
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
from .population import Population
from .individual import Individual

class GeneticAlgorithmEngine:
    def __init__(self, population_size, search_space, elitism_count, 
                 selection_strategy, crossover_strategy, mutation_strategy,
                 checkpoint_path='ga_checkpoint.pkl'):
        self.population = Population(population_size)
        self.search_space = search_space
        self.elitism_count = elitism_count
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.checkpoint_path = checkpoint_path

    def _save_checkpoint(self, generation_num):
        state = {
            'generation': generation_num,
            'population': self.population.individuals
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"--- Checkpoint saved for Generation {generation_num} ---")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            print(f"--- Resuming from checkpoint at Generation {state['generation']} ---")
            return state['generation'], state['population']
        return 0, None

    def run(self, num_generations, fitness_function):
        start_gen, loaded_individuals = self._load_checkpoint()

        if loaded_individuals:
            self.population.individuals = loaded_individuals
        elif not self.population.individuals:
            self.population.initialize(self.search_space)

        best_ever_individual = None
        
        # If resuming, find current best
        if loaded_individuals:
            evaluated = [ind for ind in self.population.individuals if ind.fitness is not None]
            if evaluated:
                best_ever_individual = max(evaluated, key=lambda x: x.fitness)

        for gen in range(start_gen, num_generations):
            print(f"\n===== Generation {gen + 1}/{num_generations} =====")
            print("Evaluating fitness of population...")

            for individual in tqdm(self.population.individuals):
                if individual.fitness is None:
                    individual.fitness = fitness_function(individual.chromosome)

            self.population.sort_by_fitness(descending=True)

            current_best = self.population.get_best()
            
            if best_ever_individual is None or (
                current_best.fitness is not None and
                (best_ever_individual.fitness is None or current_best.fitness > best_ever_individual.fitness)
            ):
                best_ever_individual = Individual(current_best.chromosome, current_best.fitness)
                print(f"*** New best overall fitness: {best_ever_individual.fitness:.4f} ***")

            print(f"Best fitness in Generation {gen + 1}: {current_best.fitness:.4f}")
            print(f"Best chromosome: {current_best.chromosome}")

            next_gen_individuals = []

            # Elitism
            if self.elitism_count > 0:
                elites = self.population.individuals[:self.elitism_count]
                next_gen_individuals.extend(elites)
                print(f"Carried over {len(elites)} elite(s).")

            # Generate offspring
            num_children = self.population.size - len(next_gen_individuals)
            if num_children > 0:
                print(f"Creating {num_children} offspring via crossover + mutation...")
                for _ in range(num_children):
                    parent1 = self.selection_strategy.select(self.population)
                    parent2 = self.selection_strategy.select(self.population)
                    
                    child_chromo = self.crossover_strategy.crossover(parent1, parent2)
                    mutated_child_chromo = self.mutation_strategy.mutate(child_chromo, self.search_space)
                    
                    next_gen_individuals.append(Individual(mutated_child_chromo))

            self.population.individuals = next_gen_individuals
            self._save_checkpoint(generation_num=gen + 1)

        print("\n===== Evolution Complete =====")
        return best_ever_individual
