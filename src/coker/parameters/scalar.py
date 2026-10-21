"""Scalar parameter and decision declarations."""

import abc
from dataclasses import dataclass
from typing import Union

import numpy as np

from coker.algebra.dimensions import Scalar, VectorSpace


class ParameterMixin(abc.ABC):
    """Define the decision-count interface for parameter declarations."""

    @abc.abstractmethod
    def degrees_of_freedom(self, *interval) -> int:
        """Return the number of scalar decisions over an optional interval."""
        pass


@dataclass
class BoundedVariable(ParameterMixin):
    """Declare one scalar decision with lower and upper bounds.

    Args:
        name: Solver parameter name.
        lower_bound: Inclusive lower bound; may be negative infinity.
        upper_bound: Inclusive upper bound; may be positive infinity.
        guess: Initial scalar value; choose a feasible value within the bounds.

    Note:
        Construction stores these values without validating bounds or guesses.
    """

    name: str
    lower_bound: float
    upper_bound: float
    guess: float = 0

    def degrees_of_freedom(self, *interval):
        """Return one scalar decision, independently of the interval."""
        return 1


@dataclass
class UnboundedVariable(ParameterMixin):
    """Declare a scalar decision without bounds.

    Args:
        name: Solver parameter name.
        guess: Initial scalar value, stored without constructor validation.
    """

    name: str
    guess: float = 0

    def degrees_of_freedom(self, *interval):
        """Return one scalar decision, independently of the interval."""
        return 1

    @property
    def upper_bound(self):
        return np.inf

    @property
    def lower_bound(self):
        return -np.inf


Constant = Union[float, int]
ValueType = Scalar | VectorSpace
