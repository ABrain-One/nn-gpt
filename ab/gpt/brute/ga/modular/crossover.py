from abc import ABC, abstractmethod
import random


class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.

        Returns:
            dict: A SINGLE child chromosome.
        """
        pass


class SinglePointCrossover(CrossoverStrategy):
    """
    Single-offspring single-point crossover.

    - Produces exactly ONE child chromosome
    - 'code' gene is inherited atomically from one parent
    - Remaining hyperparameters are crossed over safely
    """

    def crossover(self, parent1, parent2):
        child_chromo = {}

        p1 = parent1.chromosome
        p2 = parent2.chromosome

        # --- Handle 'code' gene atomically ---
        if 'code' in p1 and 'code' in p2:
            child_chromo['code'] = random.choice([p1['code'], p2['code']])
        elif 'code' in p1:
            child_chromo['code'] = p1['code']
        elif 'code' in p2:
            child_chromo['code'] = p2['code']

        # --- Union of all remaining genes (safe for asymmetric chromosomes) ---
        # Exclude 'cached_fitness' so children are forced to re-evaluate
        genes = sorted((set(p1.keys()) | set(p2.keys())) - {'code', 'cached_fitness'})

        if not genes:
            return child_chromo

        crossover_point = random.randint(0, len(genes))

        for i, gene in enumerate(genes):
            if i < crossover_point:
                if gene in p1:
                    child_chromo[gene] = p1[gene]
                elif gene in p2:
                    child_chromo[gene] = p2[gene]
            else:
                if gene in p2:
                    child_chromo[gene] = p2[gene]
                elif gene in p1:
                    child_chromo[gene] = p1[gene]

        return child_chromo
