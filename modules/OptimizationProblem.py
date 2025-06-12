import pandas as pd
import random
import numpy as np
from abc import ABC, abstractmethod


class OptimizationProblem(ABC):
    """
    Abstract base for optimization problems.
    """
    @abstractmethod
    def fitness(self, solution: np.ndarray) -> float:
        pass

    @abstractmethod
    def random_solution(self) -> np.ndarray:
        pass

    @abstractmethod
    def mutate(self, solution: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        pass
