"""Generic GA core — copied verbatim (behaviour-preserving) from ga/modular/.

Consolidated into one module so the denoise search does not import
``ga.modular`` (whose ``__init__`` pulls in the LLM mutation strategy). The
classes are unchanged in semantics: chromosome = dict of gene->value, fitness is
a float supplied by a caller-provided ``fitness_function(chromosome)``.

Source files: modular/individual.py, population.py, selection.py, crossover.py,
mutation.py, engine.py.
"""

import os
import pickle
import random
from abc import ABC, abstractmethod

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    def tqdm(iterable, *args, **kwargs):
        return iterable


class Individual:
    def __init__(self, chromosome, fitness=None):
        self.chromosome = chromosome
        self.fitness = fitness

    def __repr__(self):
        return f"Individual(fitness={self.fitness}, chromosome={self.chromosome})"


class Population:
    def __init__(self, size):
        self.size = size
        self.individuals = []

    def initialize(self, search_space):
        self.individuals = []
        for _ in range(self.size):
            chromosome = {}
            for key, values in search_space.items():
                chromosome[key] = random.choice(values)
            self.individuals.append(Individual(chromosome))
        print(f"Initialized population with {self.size} individuals.")

    def add(self, individual):
        self.individuals.append(individual)

    def sort_by_fitness(self, descending=True):
        self.individuals.sort(
            key=lambda x: x.fitness if x.fitness is not None else -float("inf"),
            reverse=descending,
        )

    def get_best(self):
        self.sort_by_fitness()
        if not self.individuals:
            return None
        return self.individuals[0]

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, index):
        return self.individuals[index]


class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population):
        ...


class TournamentSelection(SelectionStrategy):
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size

    def select(self, population):
        actual_size = min(self.tournament_size, len(population))
        competitors = random.sample(population.individuals, actual_size)
        return max(
            competitors,
            key=lambda x: x.fitness if x.fitness is not None else -float("inf"),
        )


class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2):
        ...


class SinglePointCrossover(CrossoverStrategy):
    """Single-offspring single-point crossover over gene dicts."""

    def crossover(self, parent1, parent2):
        child_chromo = {}
        p1 = parent1.chromosome
        p2 = parent2.chromosome

        # 'code' gene (if present) inherited atomically.
        if "code" in p1 and "code" in p2:
            child_chromo["code"] = random.choice([p1["code"], p2["code"]])
        elif "code" in p1:
            child_chromo["code"] = p1["code"]
        elif "code" in p2:
            child_chromo["code"] = p2["code"]

        genes = sorted((set(p1.keys()) | set(p2.keys())) - {"code", "cached_fitness"})
        if not genes:
            return child_chromo

        crossover_point = random.randint(0, len(genes))
        for i, gene in enumerate(genes):
            src = p1 if i < crossover_point else p2
            other = p2 if i < crossover_point else p1
            child_chromo[gene] = src[gene] if gene in src else other[gene]
        return child_chromo


class MutationStrategy(ABC):
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    @abstractmethod
    def mutate(self, chromosome, search_space):
        ...


class RandomResettingMutation(MutationStrategy):
    def mutate(self, chromosome, search_space):
        mutated = chromosome.copy()
        for gene in search_space.keys():
            if random.random() < self.mutation_rate:
                current = mutated.get(gene)
                others = [v for v in search_space[gene] if v != current]
                if others:
                    mutated[gene] = random.choice(others)
        return mutated


class GeneticAlgorithmEngine:
    def __init__(self, population_size, search_space, elitism_count,
                 selection_strategy, crossover_strategy, mutation_strategy,
                 checkpoint_path="ga_checkpoint.pkl"):
        self.population = Population(population_size)
        self.search_space = search_space
        self.elitism_count = elitism_count
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.checkpoint_path = checkpoint_path

    def _save_checkpoint(self, generation_num):
        state = {"generation": generation_num, "population": self.population.individuals}
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        print(f"--- Checkpoint saved for Generation {generation_num} ---")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
            print(f"--- Resuming from checkpoint at Generation {state['generation']} ---")
            return state["generation"], state["population"]
        return 0, None

    def run(self, num_generations, fitness_function, pre_generation_hook=None):
        """pre_generation_hook(individuals, gen) is called once per generation
        BEFORE the per-individual evaluation loop. It may set some individuals'
        .fitness directly (e.g. a cheap surrogate) so the expensive
        fitness_function is skipped for them — this is the accuracy-predictor
        funnel: rank the generation, pre-score the bottom fraction, train only
        the top. Default None = unchanged behaviour."""
        start_gen, loaded = self._load_checkpoint()
        if loaded:
            self.population.individuals = loaded
        elif not self.population.individuals:
            self.population.initialize(self.search_space)

        best_ever = None
        if loaded:
            evaluated = [i for i in self.population.individuals if i.fitness is not None]
            if evaluated:
                best_ever = max(evaluated, key=lambda x: x.fitness)

        for gen in range(start_gen, num_generations):
            print(f"\n===== Generation {gen + 1}/{num_generations} =====")
            if pre_generation_hook is not None:
                pre_generation_hook(self.population.individuals, gen)
            for ind in tqdm(self.population.individuals):
                if ind.fitness is None:
                    ind.fitness = fitness_function(ind.chromosome)

            self.population.sort_by_fitness(descending=True)
            current_best = self.population.get_best()

            if best_ever is None or (
                current_best.fitness is not None
                and (best_ever.fitness is None or current_best.fitness > best_ever.fitness)
            ):
                best_ever = Individual(current_best.chromosome, current_best.fitness)
                print(f"*** New best overall fitness: {best_ever.fitness:.4f} ***")

            bf = current_best.fitness
            print(f"Best fitness in Generation {gen + 1}: "
                  f"{bf:.4f}" if bf is not None else "Best fitness: None")
            print(f"Best chromosome: {current_best.chromosome}")

            next_gen = []
            if self.elitism_count > 0:
                next_gen.extend(self.population.individuals[:self.elitism_count])
                print(f"Carried over {len(next_gen)} elite(s).")

            num_children = self.population.size - len(next_gen)
            for _ in range(max(0, num_children)):
                p1 = self.selection_strategy.select(self.population)
                # Diversity-aware pairing: a strategy may choose the SECOND parent
                # conditioned on the first (e.g. most-distant tournament winner).
                mate = getattr(self.selection_strategy, "select_mate", None)
                p2 = mate(self.population, p1) if mate else \
                    self.selection_strategy.select(self.population)
                child = self.crossover_strategy.crossover(p1, p2)
                child = self.mutation_strategy.mutate(child, self.search_space)
                next_gen.append(Individual(child))

            self.population.individuals = next_gen
            self._save_checkpoint(generation_num=gen + 1)

        print("\n===== Evolution Complete =====")
        return best_ever
