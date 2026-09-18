import abc
from dataclasses import dataclass, field
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
class UnboundedVariable(ParameterMixin):
    """Scalar decision variable with a finite initial guess and no bounds."""

    name: str
    guess: float = 0
    lower_bound: float = field(default=-np.inf, init=False)
    upper_bound: float = field(default=np.inf, init=False)

    def degrees_of_freedom(self, *interval):
        return 1


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
ParameterVariable = (
    BoundedVariable
    | UnboundedVariable
    | BoundVector
    | DenseTensorVariable
    | Constant
)
Solution = (
    "DynamicalSystem" | Callable[[Scalar, ControlLaw, ValueType], Scalar]
)
LossFunction = Callable[[Solution, ControlLaw, ValueType], Scalar]
