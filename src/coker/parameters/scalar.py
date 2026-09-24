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
    """Declare one scalar decision with finite lower and upper bounds."""

    name: str
    lower_bound: float
    upper_bound: float
    guess: float = 0

    def degrees_of_freedom(self, *interval):
        return 1


@dataclass
class UnboundedVariable(ParameterMixin):
    """Scalar decision variable with a finite initial guess and no bounds."""

    name: str
    guess: float = 0

    def degrees_of_freedom(self, *interval):
        return 1

    @property
    def upper_bound(self):
        return np.inf

    @property
    def lower_bound(self):
        return -np.inf


Constant = Union[float, int]
ValueType = Scalar | VectorSpace
