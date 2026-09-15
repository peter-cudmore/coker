import abc
from dataclasses import dataclass
from typing import Callable, Sequence, Union

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
class DenseTensorVariable(ParameterMixin):
    """Finite dense decision block reconstructed with its declared shape."""

    name: str
    guess: np.ndarray | Sequence[float]
    lower_bound: float | np.ndarray = -np.inf
    upper_bound: float | np.ndarray = np.inf

    def __post_init__(self):
        guess = np.asarray(self.guess, dtype=float)
        if guess.ndim == 0:
            raise TypeError("DenseTensorVariable guess must be an array")
        lower = np.broadcast_to(self.lower_bound, guess.shape).astype(float)
        upper = np.broadcast_to(self.upper_bound, guess.shape).astype(float)
        if not np.all(np.isfinite(guess)) or np.any(lower > upper):
            raise ValueError("tensor bounds and guess are invalid")
        if np.any(guess < lower) or np.any(guess > upper):
            raise ValueError("tensor guess must be within bounds")
        self.guess = guess
        self.lower_bound = lower
        self.upper_bound = upper

    @property
    def shape(self) -> tuple[int, ...]:
        return self.guess.shape

    @property
    def size(self) -> int:
        return self.guess.size

    def degrees_of_freedom(self, *interval):
        return self.size


@dataclass
class BoundVector(DenseTensorVariable):
    """One-dimensional bounded dense decision block."""

    def __post_init__(self):
        super().__post_init__()
        if self.guess.ndim != 1:
            raise ValueError("BoundVector guess must be one-dimensional")


@dataclass
class PiecewiseConstantVariable(ParameterMixin):
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
class SpikeVariable(ParameterMixin):
    name: str
    time: float
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        return 1

    def to_solution(self, value):
        return SpikeControlSolution(self.name, self.time, value)


@dataclass
class ConstantControlVariable(ParameterMixin):
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

Constant = Union[float, int]
ValueType = Scalar | VectorSpace
ControlLaw = Callable[[Scalar], ValueType]
ControlVariable = (
    ConstantControlVariable | PiecewiseConstantVariable | SpikeVariable
)
ParameterVariable = BoundedVariable | DenseTensorVariable | Constant
Solution = (
    "DynamicalSystem" | Callable[[Scalar, ControlLaw, ValueType], Scalar]
)
LossFunction = Callable[[Solution, ControlLaw, ValueType], Scalar]
