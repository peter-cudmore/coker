import abc
from typing import List, Callable, Optional, Tuple, Union
from dataclasses import dataclass, field

from coker import Dimension
from coker.algebra.kernel import FunctionSpace, Scalar, VectorSpace, Function
import numpy as np


@dataclass
class DynamicsSpec:
    inputs: FunctionSpace
    parameters: Scalar | VectorSpace

    algebraic: Optional[VectorSpace]

    initial_conditions: Callable[
        [float, VectorSpace, VectorSpace], Tuple[np.ndarray, np.ndarray]
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

    quadratures: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    ] = None
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

    def __call__(self, *args):
        from coker.backends import get_backend_by_name

        if len(args) == 2:
            if self.inputs is not None:
                raise ValueError(
                    f"Invalid number of arguments: Expected 3, received: {len(args)}"
                )
            t, p = args
            u = lambda _: None  # No input so treat it as a noop
        elif len(args) == 3:
            t, u, p = args
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")

        if self.g is not None:
            # solve z s.t. 0 = g(0, x0, z, u(0), p)
            raise NotImplementedError
        else:
            z0 = None

        x0 = self.x0(0, z0, u, p)

        # solve ODE
        # x' = dxdt(...)
        # 0  = g(...)
        # to get x,z over the interval

        if self.dqdt is not None:
            # zeros, the same size a q
            raise NotImplementedError
        else:
            q0 = None

        backend = get_backend_by_name(self.dxdt.backend)
        x, z, q = backend.evaluate_integrals(
            [self.dxdt, self.g, self.dqdt], [x0, z0, q0], t, [u, p]
        )

        out = self.y(t, x, z, u, p, q)

        return out

class ParameterMixin(abc.ABC):
    @abc.abstractmethod
    def degrees_of_freedom(self, *interval) -> int:
        pass

@dataclass
class BoundedVariable(ParameterMixin):
    name: str
    lower_bound: float
    upper_bound: float

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


@dataclass
class SpikeVariable(ParameterMixin):
    name: str
    time: float
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        return 1

@dataclass
class ConstantControlVariable(ParameterMixin):
    name: str
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        return 1


Constant = Union[float, int]
ValueType = Scalar | VectorSpace
ControlLaw = Callable[[Scalar], ValueType]
ControlVariable = (
    ConstantControlVariable | PiecewiseConstantVariable | SpikeVariable
)
ParameterVariable = BoundedVariable | Constant
Solution = Union[DynamicalSystem, Callable[[Scalar, ControlLaw, ValueType], Scalar]]
LossFunction = Callable[[Solution, ControlLaw, ValueType], Scalar]


@dataclass
class TranscriptionOptions:
    minimum_n_intervals: int = 4
    minimum_degree: int = 4


@dataclass
class VariationalProblem:
    loss: LossFunction
    system: DynamicalSystem
    arguments: Tuple[List[ControlVariable], List[ParameterVariable]]
    t_final: float
    constraints: Optional[List] = None
    transcription_options: TranscriptionOptions = field(
        default_factory=TranscriptionOptions
    )

#    def __post_init__(self):
#        if self.constraints is None:
#            self.constraints = []
#
#        if not isinstance(self.loss, Function):
#            output_space, = self.system.y.output_shape()
#
#            parameter_space = VectorSpace('p', len(self.arguments[1]))
#
#            control_space = FunctionSpace('u', [Scalar('t')], [VectorSpace('u', len(self.arguments[0]))]),
#            solution_space = FunctionSpace('x', [Scalar('t'), control_space, parameter_space], [output_space])
#            loss_input_space = [
#                solution_space,
#                control_space,
#                parameter_space
#            ]
#            self.loss = Function(loss_input_space, self.loss, backend=self.system.backend())


    def lower(self, backend: Optional[str] = None):
        from coker.backends import get_backend_by_name

        backend = get_backend_by_name(backend or self.system.backend())
        return backend.create_variational_solver(self)

    def call(self):
        solver = self.lower()
        return solver()
