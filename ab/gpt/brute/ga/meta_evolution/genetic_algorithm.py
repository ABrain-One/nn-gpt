import random
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
        return {key: random.choice(values) for key, values in self.search_space.items()}

    def _initialize_population(self):
        self.population = [{'chromosome': self._create_random_chromosome(), 'fitness': None} for _ in range(self.population_size)]

    def _save_checkpoint(self, generation_num):
        state = {'generation': generation_num, 'population': self.population}
        with open(self.checkpoint_path, 'wb') as f: pickle.dump(state, f)

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
                return state['generation'], state['population']
            except: pass
        return 0, None

    # --- START LLM: CROSSOVER ---
    def _crossover(self, parent1_chromo, parent2_chromo):
        child_chromo = {}
        genes = list(self.search_space.keys())
        point = random.randint(1, len(genes) - 1)
        for i, gene in enumerate(genes):
            if i < point:
                child_chromo[gene] = parent1_chromo[gene]
            else:
                child_chromo[gene] = parent2_chromo[gene]
        return child_chromo
    # --- END LLM: CROSSOVER ---

    # --- START LLM: MUTATION ---
    def mutate_gene(self, current_value, possible_values):
        """
        Return a new gene value.
        """
        if not isinstance(possible_values, list):
            raise ValueError('possible_values should be a list')
        if not possible_values:
            return
        if len(possible_values) == 1:
            return possible_values[0]
        new_value = random.choice(possible_values)
        while new_value == current_value:
            new_value = random.choice(possible_values)
        return new_value
    def _mutate(self, chromosome):
        mutated_chromo = chromosome.copy()
        for gene in self.search_space.keys():
            if random.random() < self.mutation_rate:
                possibles = [v for v in self.search_space[gene] if v != mutated_chromo[gene]]
                if possibles:
                    mutated_chromo[gene] = self.mutate_gene(mutated_chromo[gene], possibles)
        return mutated_chromo
    # --- END LLM: MUTATION ---

    # --- START LLM: SELECTION ---
    def _selection(self):
        k = 3
        competitors = random.sample(self.population, min(k, len(self.population)))
        return max(competitors, key=lambda x: x['fitness'] if x['fitness'] is not None else -1)
    # --- END LLM: SELECTION ---

    def run(self, num_generations, fitness_function):
        start_gen, loaded_population = self._load_checkpoint()
        if loaded_population is not None: self.population = loaded_population
        else: self._initialize_population()
            
        fitness_history = []
        best_overall = None

        for gen in range(start_gen, num_generations):
            print(f"\n\n >>> GENERATION {gen + 1} <<<\n")
            # Evaluate
            for i, ind in enumerate(self.population):
                if ind['fitness'] is None:
                    print(f"  Evaluating Individual {i+1}/{len(self.population)}")
                    ind['fitness'] = fitness_function(ind['chromosome'])
            
            # Sort
            self.population.sort(key=lambda x: x['fitness'] if x['fitness'] is not None else -1, reverse=True)
            
            # Record keeping
            current_best = self.population[0]['fitness']
            fitness_history.append(current_best)
            if best_overall is None or current_best > best_overall['fitness']:
                best_overall = self.population[0].copy()

            # Next Gen
            next_gen = self.population[:self.elitism_count]
            while len(next_gen) < self.population_size:
                p1 = self._selection()
                p2 = self._selection()
                child = self._crossover(p1['chromosome'], p2['chromosome'])
                child = self._mutate(child)
                next_gen.append({'chromosome': child, 'fitness': None})
            
            self.population = next_gen
            # self._save_checkpoint(gen + 1)
            
        return best_overall, fitness_history