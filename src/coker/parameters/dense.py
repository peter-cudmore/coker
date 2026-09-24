"""Dense and vector parameter declarations."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .scalar import ParameterMixin


@dataclass
class DenseTensorVariable(ParameterMixin):
    """Finite dense decision block reconstructed with its declared shape."""

    name: str
    guess: np.ndarray | Sequence[float]

    def __post_init__(self):
        guess = np.asarray(self.guess, dtype=float)
        if guess.ndim == 0 or not np.all(np.isfinite(guess)):
            raise ValueError("DenseTensorVariable guess must be finite array")
        self.guess = guess

    @property
    def shape(self) -> tuple[int, ...]:
        return self.guess.shape

    @property
    def size(self) -> int:
        return self.guess.size

    def degrees_of_freedom(self, *interval):
        return self.size


@dataclass
class BoundVector(ParameterMixin):
    """One-dimensional finite decision block with component-wise bounds."""

    name: str
    lower_bound: np.ndarray | Sequence[float]
    upper_bound: np.ndarray | Sequence[float]
    guess: np.ndarray | Sequence[float]

    def __post_init__(self):
        self.guess = np.asarray(self.guess, dtype=float)
        self.lower_bound = np.asarray(self.lower_bound, dtype=float)
        self.upper_bound = np.asarray(self.upper_bound, dtype=float)
        if (
            self.guess.ndim != 1
            or self.lower_bound.shape != self.guess.shape
            or self.upper_bound.shape != self.guess.shape
            or not np.all(np.isfinite(self.guess))
            or np.any(self.lower_bound > self.upper_bound)
            or np.any(self.guess < self.lower_bound)
            or np.any(self.guess > self.upper_bound)
        ):
            raise ValueError("BoundVector bounds and guess are invalid")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.guess.shape

    @property
    def size(self) -> int:
        return self.guess.size

    def degrees_of_freedom(self, *interval):
        return self.size
