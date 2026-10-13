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
    """Declare one bounded scalar, sampled piecewise-constant control.

    Args:
        name: Solution/control channel name.
        sample_rate: Samples per unit of physical time.
        upper_bound: Upper bound for every sample.
        lower_bound: Lower bound for every sample.

    The number of samples is ``ceil(duration * sample_rate)``. Evaluation
    uses ``int(t * sample_rate)``; the terminal boundary is not a new sample.
    """

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
    """Declare a bounded scalar control concentrated at one physical time.

    Args:
        name: Solution/control channel name.
        time: Physical time of the spike.
        upper_bound: Upper bound on spike amplitude.
        lower_bound: Lower bound on spike amplitude.
    """

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
    """Declare one scalar control constant over the full horizon.

    Args:
        name: Solution/control channel name.
        upper_bound: Upper bound on the constant value.
        lower_bound: Lower bound on the constant value.
    """

    name: str
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        return 1

    def to_solution(self, value):
        return ConstantControlSolution(self.name, value)


@dataclass
class ConstantControlSolution:
    """Evaluate a fitted constant scalar control at any scalar time."""

    name: str
    value: float

    def __call__(self, t):
        return self.value


@dataclass
class SpikeControlSolution:
    """Evaluate the amplitude within ``tolerance`` of the declared spike."""

    name: str
    time: float
    value: float
    tolerance: float = 1e-9

    def __call__(self, t):
        return self.value if abs(t - self.time) < self.tolerance else 0.0


@dataclass
class PiecewiseControlSolution:
    """Evaluate fitted scalar samples in half-open physical-time bins.

    Times outside the stored sample range are not supported, including a
    terminal endpoint whose bin index equals the number of samples.
    """

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
