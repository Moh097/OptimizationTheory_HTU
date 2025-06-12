# modules/RandomForestTuning.py
"""
Single‑ & Multi‑objective fitness for RandomForest hyper‑parameters.

Gene layout (fixed):
    [n_estimators, max_depth, min_samples_split, min_samples_leaf]
"""
from __future__ import annotations
from typing import List, Tuple, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from .OptimizationProblem import OptimizationProblem


class RandomForestTuning(OptimizationProblem):
    # ------------------------------------------------ constructor
    def __init__(self, X, y, cv: int = 5) -> None:
        self.X, self.y, self.cv = X, y, cv

        self._param_info: List[Tuple[str, int, int]] = [
            ("n_estimators",      10, 200),
            ("max_depth",          3, 20),
            ("min_samples_split",  2, 20),
            ("min_samples_leaf",   1, 10),
        ]
        # convenient dict access
        self.param_ranges: Dict[str, Tuple[int, int]] = {
            name: (low, high) for name, low, high in self._param_info
        }

    # ---------------------------------------- required interface (single obj.)
    def random_solution(self) -> List[int]:
        return [np.random.randint(low, high + 1)
                for _, low, high in self._param_info]

    def fitness(self, solution: List[int]) -> float:
        """Return CV accuracy (the *single* objective)."""
        acc, _ = self.objectives(solution)
        return acc

    def crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        return [p1[i] if np.random.rand() < 0.5 else p2[i]
                for i in range(len(p1))]

    def mutate(self, sol: List[int], rate: float) -> List[int]:
        out = sol.copy()
        for i, (_, low, high) in enumerate(self._param_info):
            if np.random.rand() < rate:
                out[i] = int(np.clip(out[i] + np.random.randint(-2, 3), low, high))
        return out

    # ---------------------------------------- multi‑objective helper
    def objectives(self, sol: List[int]) -> Tuple[float, int]:
        """
        (1) accuracy  – maximise  
        (2) size      – minimise (proxy: n_estimators)
        """
        n_estimators, max_depth, min_split, min_leaf = map(int, sol)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=42,
        )
        acc = cross_val_score(model, self.X, self.y,
                              cv=self.cv, scoring="accuracy").mean()
        size = n_estimators
        return acc, size
