from dataclasses import dataclass
from typing import Callable

import numpy as np

from coker.algebra.dimensions import Scalar
from coker.parameters import (
    ParameterMixin as _ParameterMixin,
    ValueType as _ValueType,
)


@dataclass
class PiecewiseConstantVariable(_ParameterMixin):
    name: str
    sample_rate: float
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        start, end = interval
        return int(np.ceil((end - start) * self.sample_rate))

    def to_solution(self, value):
        return PiecewiseControlSolution(self.name, self.sample_rate, value)


@dataclass
class SpikeVariable(_ParameterMixin):
    name: str
    time: float
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        return 1

    def to_solution(self, value):
        return SpikeControlSolution(self.name, self.time, value)


@dataclass
class ConstantControlVariable(_ParameterMixin):
    name: str
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        return 1

    def to_solution(self, value):
        return ConstantControlSolution(self.name, value)


@dataclass
class ConstantControlSolution:
    name: str
    value: float

    def __call__(self, t):
        return self.value


@dataclass
class SpikeControlSolution:
    name: str
    time: float
    value: float
    tolerance: float = 1e-9

    def __call__(self, t):
        return self.value if abs(t - self.time) < self.tolerance else 0.0


@dataclass
class PiecewiseControlSolution:
    name: str
    sample_rate: float
    value: np.ndarray

    def __call__(self, t):
        idx = int(t * self.sample_rate)
        assert idx < len(self.value), f"Time {t} is outside of the interval"
        return self.value[idx]


ControlSolution = (
    SpikeControlSolution | PiecewiseControlSolution | ConstantControlSolution
)
ControlLaw = Callable[[Scalar], _ValueType]
ControlVariable = (
    ConstantControlVariable | PiecewiseConstantVariable | SpikeVariable
)

Solution = (
    "DynamicalSystem" | Callable[[Scalar, ControlLaw, _ValueType], Scalar]
)
LossFunction = Callable[[Solution, ControlLaw, _ValueType], Scalar]
