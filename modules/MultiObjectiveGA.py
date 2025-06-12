# modules/MultiObjectiveGA.py
"""
A *very* small NSGA‑II implementation good enough for coursework.
Works with any problem that offers:
    • random_solution()
    • crossover(p1, p2)
    • mutate(sol, rate)
    • objectives(sol) -> (obj1, obj2, ...)
Here we assume exactly two objectives and *maximise* the first
while *minimising* the others – this is sufficient for the RF
tuning example (acc ↑, size ↓).
"""
from __future__ import annotations
import random
from typing import List, Tuple


def dominates(a_objs, b_objs) -> bool:
    """Return True if *a* dominates *b* (≥ in all, > in at least one)."""
    better_or_equal = all(x >= y if i == 0 else x <= y
                          for i, (x, y) in enumerate(zip(a_objs, b_objs)))
    strictly_better = any(x > y if i == 0 else x < y
                          for i, (x, y) in enumerate(zip(a_objs, b_objs)))
    return better_or_equal and strictly_better


class MultiObjectiveGA:
    """NSGA‑II ( pared‑down )."""
    def __init__(self, problem,
                 pop_size=40, generations=30,
                 crossover_rate=0.8, mutation_rate=0.2):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    # ----------------------------------------------------- helpers
    def _initial_population(self):
        return [self.problem.random_solution() for _ in range(self.pop_size)]

    def _eval_objs(self, population):
        return [self.problem.objectives(sol) for sol in population]

    # ---------- NSGA‑II bits ----------
    @staticmethod
    def _fast_nondominated_sort(pop, objs) -> List[List[int]]:
        fronts: List[List[int]] = []
        S = [[] for _ in pop]          # those dominated by i
        n = [0] * len(pop)             # domination count
        for i in range(len(pop)):
            for j in range(len(pop)):
                if dominates(objs[i], objs[j]):
                    S[i].append(j)
                elif dominates(objs[j], objs[i]):
                    n[i] += 1
            if n[i] == 0:
                if len(fronts) == 0:
                    fronts.append([])
                fronts[0].append(i)

        f = 0
        while f < len(fronts):
            next_front = []
            for i in fronts[f]:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            f += 1
        return fronts

    @staticmethod
    def _crowding_distance(front_ids, objs) -> List[float]:
        if len(front_ids) == 0:
            return []
        m = len(objs[0])               # number of objectives
        distance = {idx: 0.0 for idx in front_ids}
        for k in range(m):
            key = lambda i: objs[i][k]
            sorted_ids = sorted(front_ids, key=key)
            distance[sorted_ids[0]] = distance[sorted_ids[-1]] = float('inf')
            f_min, f_max = objs[sorted_ids[0]][k], objs[sorted_ids[-1]][k]
            if f_max == f_min:
                continue
            for i in range(1, len(sorted_ids) - 1):
                prev_f = objs[sorted_ids[i - 1]][k]
                next_f = objs[sorted_ids[i + 1]][k]
                # normalised
                distance[sorted_ids[i]] += (next_f - prev_f) / (f_max - f_min)
        return [distance[i] for i in front_ids]

    # ----------------------------------------------------- evolution
    def evolve(self) -> List[Tuple]:
        pop = self._initial_population()
        objs = self._eval_objs(pop)

        for _ in range(self.generations):
            # -------- produce offspring
            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = random.sample(pop, 2)
                child = (self.problem.crossover(p1, p2)
                         if random.random() < self.crossover_rate else p1.copy())
                child = self.problem.mutate(child, self.mutation_rate)
                offspring.append(child)

            # combine & sort
            combined = pop + offspring
            combined_objs = objs + self._eval_objs(offspring)
            fronts = self._fast_nondominated_sort(combined, combined_objs)

            # build next population
            new_pop, new_objs = [], []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    # take whole front
                    new_pop.extend(combined[i] for i in front)
                    new_objs.extend(combined_objs[i] for i in front)
                else:
                    # sort by crowding distance to fill the remaining slots
                    cd = self._crowding_distance(front, combined_objs)
                    sorted_front = [i for _, i in sorted(zip(cd, front), reverse=True)]
                    slots = self.pop_size - len(new_pop)
                    chosen = sorted_front[:slots]
                    new_pop.extend(combined[i] for i in chosen)
                    new_objs.extend(combined_objs[i] for i in chosen)
                    break
            pop, objs = new_pop, new_objs

        # return the final Pareto front (non‑dominated set)
        final_front_ids = self._fast_nondominated_sort(pop, objs)[0]
        return [(pop[i], objs[i]) for i in final_front_ids]
