"""Scalar parameter and decision declarations."""

import abc
from dataclasses import dataclass
from typing import Union

import numpy as np

from coker.algebra.dimensions import Scalar, VectorSpace


class ParameterMixin(abc.ABC):
    @abc.abstractmethod
    def degrees_of_freedom(self, *interval) -> int:
        pass


@dataclass
class BoundedVariable(ParameterMixin):
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
