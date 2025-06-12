# modules/GeneticAlgorithm.py

import random

class GeneticAlgorithm:
    """
    Generic Genetic Algorithm framework.
    """
    def __init__(
        self,
        problem,
        pop_size=10,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.2
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def initialize(self):
        """Create initial population of random solutions."""
        return [self.problem.random_solution() for _ in range(self.pop_size)]

    def evaluate(self, population):
        """Evaluate all solutions and return list of fitness values."""
        return [self.problem.fitness(sol) for sol in population]

    def select(self, population, fitnesses):
        """Tournament selection: pick pairs and select better for next generation."""
        n = len(population)
        # If too few individuals, return a copy to avoid sampling errors
        if n < 2:
            return population.copy()
        selected = []
        for _ in range(n):
            i, j = random.sample(range(n), 2)
            winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        """Apply crossover with given rate, else return parent1 clone."""
        if random.random() < self.crossover_rate:
            return self.problem.crossover(parent1, parent2)
        # return a shallow copy to avoid aliasing
        return parent1.copy()

    def mutate(self, solution):
        """Apply mutation to a solution and return mutated solution."""
        return self.problem.mutate(solution, self.mutation_rate)

    def evolve(self):
        """
        Run the GA: initialize population, then iteratively select, crossover, and mutate.
        Returns the best solution and its fitness.
        """
        # Initialize
        population = self.initialize()

        # Evolve
        for _ in range(self.generations):
            fitnesses = self.evaluate(population)
            # Selection
            population = self.select(population, fitnesses)
            # Crossover and mutation
            next_pop = []
            for i in range(0, len(population), 2):
                parent1 = population[i]
                parent2 = population[i+1] if i+1 < len(population) else population[i]
                # produce two children
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                next_pop.append(self.mutate(child1))
                next_pop.append(self.mutate(child2))
            # Trim or extend to maintain population size
            population = next_pop[:self.pop_size]

        # Final evaluation to find best
        final_fitnesses = self.evaluate(population)
        best_idx = final_fitnesses.index(max(final_fitnesses))
        return population[best_idx], final_fitnesses[best_idx]
