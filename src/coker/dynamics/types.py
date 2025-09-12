import abc
from typing import List, Callable, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field

from coker import Dimension
from coker.algebra.kernel import (
    FunctionSpace,
    Scalar,
    VectorSpace,
    Function,
    Noop,
)
import numpy as np
from typing import Dict


@dataclass
class DynamicsSpec:
    inputs: FunctionSpace
    parameters: Scalar | VectorSpace
    algebraic: Optional[VectorSpace]

    initial_conditions: Callable[
        [VectorSpace, VectorSpace], Tuple[np.ndarray, np.ndarray]
    ]
    """ [x, z] = initial_conditions(t_0, p) """

    dynamics: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    """dx = dynamics(t, x, z, u, p)"""

    constraints: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    """ g(t, x, z, u, p) = constraints(t, x, z, u, p) = 0."""

    outputs: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ]
    """ y(t) = outputs(t, x, z, u, p, q) """

    quadratures: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    """ dq/dt = quadratures(t, x, u, p) """


@dataclass
class DynamicalSystem:
    inputs: FunctionSpace
    parameters: VectorSpace | Scalar
    x0: Function
    dxdt: Function
    g: Optional[Function]
    dqdt: Optional[Function]
    y: Function

    def get_state_dimensions(self) -> Tuple[Dimension, Dimension, Dimension]:
        _t, x_dim, z_dim, _u, _p, q_dim = self.y.input_shape()
        return x_dim, z_dim, q_dim

    def backend(self):
        return self.dxdt.backend

    def _map_arguments(self, *args):
        arg_stack = list(reversed(args))
        t = arg_stack.pop()
        try:
            u = arg_stack.pop() if self.inputs is not Noop() else None
            p = arg_stack.pop() if self.parameters is not None else None
        except IndexError as ex:
            raise ValueError(
                f"Invalid number of arguments: Expected 2 - 3, received: {len(args)}"
            ) from ex

        return t, u, p

    def __call__(self, *args):
        from coker.backends import get_backend_by_name

        t, u, p = self._map_arguments(*args)

        x0, z0 = self.x0(0, u, p)

        # solve ODE
        # x' = dxdt(...)
        # 0  = g(...)
        # to get x,z over the interval

        if self.dqdt is not Noop():
            # zeros, the same size a q
            raise NotImplementedError
        else:
            q0 = None

        backend = get_backend_by_name(self.dxdt.backend)
        x, z, q = backend.evaluate_integrals(
            [self.dxdt, self.g, self.dqdt], [x0, z0, q0], t, [u, p]
        )

        if isinstance(t, (float, int)):
            return self.y(t, x, z, u, p, q)

        def map_args(i):
            t_i = t[i]
            x_i = x[:, i]
            z_i = z[:, i] if z is not None else None
            q_i = q[:, i] if q is not None else None
            return t_i, x_i, z_i, u, p, q_i

        return np.concatenate([self.y(*map_args(i)) for i, _ in enumerate(t)])


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


Constant = Union[float, int]
ValueType = Scalar | VectorSpace
ControlLaw = Callable[[Scalar], ValueType]
ControlVariable = (
    ConstantControlVariable | PiecewiseConstantVariable | SpikeVariable
)
ParameterVariable = BoundedVariable | Constant
Solution = Union[
    DynamicalSystem, Callable[[Scalar, ControlLaw, ValueType], Scalar]
]
LossFunction = Callable[[Solution, ControlLaw, ValueType], Scalar]


@dataclass
class TranscriptionOptions:
    minimum_n_intervals: int = 4
    minimum_degree: int = 7
    absolute_tolerance: float = 1e-12
    verbose: bool = False
    optimiser_options: dict = field(default_factory=dict)
    initialise_near_guess: bool = False


@dataclass
class VariationalProblem:
    loss: LossFunction
    system: DynamicalSystem
    t_final: float
    control: Optional[List[ControlVariable]] = None
    parameters: Optional[List[ParameterVariable]] = None
    constraints: Optional[List] = None
    transcription_options: TranscriptionOptions = field(
        default_factory=TranscriptionOptions
    )
    backend: Optional[str] = "casadi"

    def __call__(self) -> "VariationalSolution":
        from coker.backends import get_backend_by_name

        if self.system.parameters is not None:
            n = self.system.parameters.dimension
            assert len(self.parameters) == n
        if self.control is not None:
            assert self.system.inputs is not Noop()

        backend = get_backend_by_name(self.backend)
        return backend.create_variational_solver(self)


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


ControlSolution = Union[
    SpikeControlSolution, PiecewiseControlSolution, ConstantControlSolution
]


class InterpolatingPolyCollection:
    def __init__(self, polys: List["InterpolatingPoly"]):
        self.polys = polys
        self._size = sum(p.size() for p in polys)
        self.intervals = [p.interval for p in polys]

    def size(self):
        return self._size

    def __call__(self, t):
        for i, (start, end) in enumerate(self.intervals):
            if start <= t <= end:
                return self.polys[i](t)
        raise ValueError(f"Value {t} is not in any interval")

    def interval_starts(self):
        for p in self.polys:
            yield p.start_point()

    def interval_ends(self):
        for p in self.polys:
            yield p.end_point()

    def knot_points(self) -> Iterator[Tuple[float, np.ndarray, np.ndarray]]:
        for p in self.polys:
            for point in p.knot_points():
                yield point

    def map(self, func: Callable[[float, np.ndarray], np.ndarray]):
        new_polys = [p.map(func) for p in self.polys]
        return InterpolatingPolyCollection(new_polys)


@dataclass
class VariationalSolution:
    cost: float
    path: InterpolatingPolyCollection
    projectors: Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]
    control_solutions: List[ControlSolution]
    parameter_solutions: Dict[str, float]
    parameters: np.ndarray
    output: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ]

    def as_raw(self) -> np.ndarray:
        points = [
            np.vstack([np.array([t]), x])
            for t, x, _ in self.path.knot_points()
        ]
        return np.hstack(points)

    def state(self, t):
        v = self.path(t)
        proj = self.projectors[0]
        return proj @ v

    def algebraic(self, t):
        if self.projectors[1] is None:
            return None
        return self.projectors[1] @ self.path(t)

    def quadratures(self, t):
        if self.projectors[2] is None:
            return None
        return self.projectors[2] @ self.path(t)

    def control_law(self, t):
        if not self.control_solutions:
            return None
        return np.array([c(t) for c in self.control_solutions])

    def __call__(self, t) -> np.ndarray:
        x = self.state(t)
        q = self.quadratures(t)
        u = self.control_law(t)
        z = self.algebraic(t)
        return self.output(t, x, z, u, self.parameters, q)

    def to_poly(self) -> InterpolatingPolyCollection:

        def f(t, v):
            x = self.projectors[0] @ v
            z = (
                self.projectors[1] @ v
                if self.projectors[1] is not None
                else None
            )
            q = (
                self.projectors[2] @ v
                if self.projectors[2] is not None
                else None
            )
            u = self.control_law(t)
            return self.output(t, x, z, u, self.parameters, q)

        return self.path.map(f)
