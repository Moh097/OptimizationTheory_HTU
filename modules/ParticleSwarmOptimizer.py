# modules/ParticleSwarmOptimizer.py
import numpy as np
import random


class ParticleSwarmOptimizer:
    """
    Very small PSO good enough for coursework.
    Works with any OptimizationProblem that exposes:
        param_ranges  (dict)
        random_solution()
        fitness()
    """

    def __init__(
        self,
        problem,
        pop_size=30,
        w=0.5,
        c1=1.0,
        c2=1.0,
        iterations=20,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.w, self.c1, self.c2 = w, c1, c2
        self.iterations = iterations

    # ---------- main ----------
    def optimize(self):
        self._init_swarm()
        for _ in range(self.iterations):
            for i in range(self.pop_size):
                self._update_particle(i)
        return self.gbest_pos.copy(), self.gbest_score

    # ---------- helpers ----------
    def _init_swarm(self):
        self.positions = [np.array(self.problem.random_solution(), dtype=float)
                          for _ in range(self.pop_size)]
        self.velocities = [np.zeros_like(p) for p in self.positions]
        self.pbest_pos = [p.copy() for p in self.positions]
        self.pbest_score = [self.problem.fitness(p) for p in self.positions]
        best_idx = int(np.argmax(self.pbest_score))
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_score = self.pbest_score[best_idx]

    def _update_particle(self, i):
        r1, r2 = random.random(), random.random()
        v = (
            self.w * self.velocities[i]
            + self.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
            + self.c2 * r2 * (self.gbest_pos - self.positions[i])
        )
        self.velocities[i] = v
        # new position
        new_pos = self.positions[i] + v

        # clamp/round into legal space
        clamped = []
        for k, key in enumerate(self.problem.param_ranges):
            low, high = self.problem.param_ranges[key]
            val = int(round(new_pos[k]))
            clamped.append(np.clip(val, low, high))
        self.positions[i] = np.array(clamped, dtype=float)

        # evaluate
        score = self.problem.fitness(self.positions[i])
        if score > self.pbest_score[i]:
            self.pbest_pos[i] = self.positions[i].copy()
            self.pbest_score[i] = score
            if score > self.gbest_score:
                self.gbest_pos = self.positions[i].copy()
                self.gbest_score = score
