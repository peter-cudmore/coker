"""Dense and vector parameter declarations."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .scalar import ParameterMixin


@dataclass
class DenseTensorVariable(ParameterMixin):
    """Declare an unbounded dense decision block with a retained array shape.

    Args:
        name: Key used to identify the fitted array in named parameter results.
        guess: Finite numeric array or sequence with at least one dimension.
            Converted to a floating-point NumPy array; its shape determines the
            fitted block's shape and its size determines the decision count.

    Raises:
        TypeError: The guess cannot be converted to floating-point values.
        ValueError: The guess is scalar, non-finite or not numerically convertible.

    No physical units are attached or converted. Supply values in the units
    expected by the model. There are no component-wise bounds.

    Examples:
        >>> from coker.parameters import DenseTensorVariable
        >>> matrix = DenseTensorVariable("matrix", [[1, 0], [0, 1]])
        >>> (matrix.shape, matrix.size, matrix.degrees_of_freedom())
        ((2, 2), 4, 4)
        >>> matrix.guess.tolist()
        [[1.0, 0.0], [0.0, 1.0]]
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
        """tuple[int, ...]: Array shape retained in the fitted value."""
        return self.guess.shape

    @property
    def size(self) -> int:
        """int: Number of independent scalar entries in the array."""
        return self.guess.size

    def degrees_of_freedom(self, *interval):
        """Return the number of scalar entries in the decision block.

        Args:
            *interval: Accepted for the decision-count interface; ignored.

        Returns:
            int: The array size, independent of the interval.
        """
        return self.size


@dataclass
class BoundVector(ParameterMixin):
    """Declare a vector decision with inclusive component-wise bounds.

    Args:
        name: Key used to identify the fitted vector in named parameter results.
        lower_bound: Lower bounds with exactly the guess's one-dimensional shape.
        upper_bound: Upper bounds with exactly the guess's one-dimensional shape.
        guess: Finite initial vector. Bounds and guess are converted to
            floating-point NumPy arrays; bounds are not broadcast.

    Raises:
        TypeError: An input cannot be converted to floating-point values.
        ValueError: Shapes differ, the guess is not a finite vector, bounds are
            reversed, the guess lies outside them, or conversion fails.

    Infinite bounds are permitted. Supply ordered non-NaN bounds: construction
    does not explicitly reject NaN bounds. All entries must use units consistent
    with the model; this declaration performs no unit conversion.

    Examples:
        >>> from coker.parameters import BoundVector
        >>> gains = BoundVector("gains", [0, 0], [2, 3], [1, 2])
        >>> (gains.shape, gains.size, gains.degrees_of_freedom())
        ((2,), 2, 2)
        >>> gains.guess.tolist()
        [1.0, 2.0]
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
        """tuple[int, ...]: One-dimensional shape retained in the fitted value."""
        return self.guess.shape

    @property
    def size(self) -> int:
        """int: Number of scalar components in the vector."""
        return self.guess.size

    def degrees_of_freedom(self, *interval):
        """Return the number of scalar components in the vector.

        Args:
            *interval: Accepted for the decision-count interface; ignored.

        Returns:
            int: The vector size, independent of the interval.
        """
        return self.size
