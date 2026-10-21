"""Dense and vector parameter declarations."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .scalar import ParameterMixin


@dataclass
class DenseTensorVariable(ParameterMixin):
    """Declare a finite dense block reconstructed with its initial shape.

    Args:
        name: Solver parameter name.
        guess: Finite non-scalar array or sequence defining shape and initial
            values. Array conversion may retain the caller's storage.

    Raises:
        ValueError: The guess is scalar or contains a non-finite value.
    """

    name: str
    guess: np.ndarray | Sequence[float]

    def __post_init__(self):
        guess = np.asarray(self.guess, dtype=float)
        if guess.ndim == 0 or not np.all(np.isfinite(guess)):
            raise ValueError("DenseTensorVariable guess must be finite array")
        self.guess = guess

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dense shape inferred from the guess."""
        return self.guess.shape

    @property
    def size(self) -> int:
        """Return the number of scalar decisions in this block."""
        return self.guess.size

    def degrees_of_freedom(self, *interval):
        """Return the block size, independently of the interval."""
        return self.size


@dataclass
class BoundVector(ParameterMixin):
    """Declare a vector decision with inclusive component-wise bounds.

    Args:
        name: Solver parameter name.
        lower_bound: Lower-bound vector with the same shape as ``guess``.
        upper_bound: Upper-bound vector with the same shape as ``guess``.
        guess: Finite one-dimensional feasible initial vector.

    Raises:
        ValueError: Shapes differ, bounds are reversed, or the guess is
            non-finite or outside the bounds.
    """

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
        """Return the one-dimensional shape of this block."""
        return self.guess.shape

    @property
    def size(self) -> int:
        """Return the number of scalar decisions in this block."""
        return self.guess.size

    def degrees_of_freedom(self, *interval):
        """Return the vector size, independently of the interval."""
        return self.size
