"""
Baseline Models Module.

Provides benchmark models (e.g. Discrete First-Order Hidden Markov Models / HMMs)
to contrast against continuous-time Neural SDE trajectory modeling.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseDiscreteBaseline(ABC):
    """
    Abstract Base Class for Discrete First-Order Baseline Models (HMMs / Markov chains).
    """

    def __init__(self, num_states: int):
        self.num_states = num_states

    @abstractmethod
    def fit(self, observations: np.ndarray, time_steps: np.ndarray) -> None:
        """
        Fit baseline model parameters on discretized observation sequences.
        """
        pass

    @abstractmethod
    def sample(self, num_sequences: int, max_length: int) -> np.ndarray:
        """
        Sample synthetic discrete trajectories.
        """
        pass
