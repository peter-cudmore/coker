import dataclasses
import enum
from typing import Optional, List, Tuple, Callable

import numpy as np

from coker import Dimension
from coker.algebra.kernel import Tape, Tracer, VectorSpace, Scalar
from coker.backends import get_backend_by_name, get_current_backend


@dataclasses.dataclass
class SolverOptions:
    warm_start: bool = False


class Minimise:
    def __init__(self, expression: Tracer):
        self.expression = expression


class MathematicalProgram:
    def __init__(
        self,
        input_shape: Tuple[Dimension, ...],
        output_shape: Tuple[Dimension, ...],
        impl: Callable,
    ):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.impl = impl

    def __call__(self, *args):
        assert len(args) == len(self.input_shape)

        return [
            np.reshape(o, dim.shape)
            for o, dim in zip(self.impl(args), self.output_shape)
        ]


class ProblemBuilder:
    def __init__(self, arguments: Optional[List[VectorSpace | Scalar]] = None):

        self.arguments = (
            [self.tape.input(a) for a in arguments] if arguments else []
        )
        self.objective = None
        self.tape: Optional[Tape] = None
        self.constraints = []
        self.outputs = []
        self.initial_conditions = {}
        self.warm_start = False

    def new_variable(self, name, shape=None, initial_value=None):
        assert self.tape is not None
        if shape is None:
            v = self.tape.input(Scalar(name))
            initial_value = 0 if initial_value is None else initial_value
        else:
            v = self.tape.input(VectorSpace(name, shape))
            initial_value = (
                np.zeros(shape=shape)
                if initial_value is None
                else initial_value
            )

        self.initial_conditions[v.index] = initial_value
        return v

    @property
    def input_shape(self) -> Tuple[Dimension, ...]:
        return tuple(i.dim for i in self.arguments)

    @property
    def output_shape(self) -> Tuple[Dimension, ...]:
        return tuple(o.dim for o in self.outputs)

    def build(self, backend: Optional[str] = None) -> MathematicalProgram:
        assert isinstance(self.objective, Minimise)
        assert self.tape is not None
        assert self.outputs

        backend = (
            get_backend_by_name(backend)
            if backend is not None
            else get_current_backend()
        )

        impl = backend.build_optimisation_problem(
            self.objective.expression,  # cost
            self.constraints,
            self.arguments,
            self.outputs,
            self.initial_conditions,
        )

        return MathematicalProgram(self.input_shape, self.output_shape, impl)

    def __enter__(self):
        self.tape = Tape()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tape = None
        pass


class VariationalProblem:
    """

    A Variational problem is of the form


        min_[u(t), p]  c(1)

        s.t.
            dx/dt = f(x(t), p)                              [Dynamics]
            dq/dt = g(x(t), p)                              [Integral Constraints]
            0 < h(t, x, p)                                  [Path Constraints]
            c = C(t, x, q, p)                               [Cost]
            (t, x, q, u, p) in  [0, 1] * X * Q * U * P      [Set Constraints]


    A solution is the
        - value c,
        - vector function u(t)
        - parameters p

    that (locally) minimise c.


    This is solved by:
        1. Choosing a discritisation scheme for t, [x,q] and u
        2. transcribing the problem into a nonlinear program by evaluating the constraints at knot points
           and adding additional constraints to enforce continuity.
        3. Classifying the NLP and passing it into an appropriate solver.

    """

    def __init__(self, intervals: List[float]):
        self.intervals = intervals
        self.signals = []
        self.variables = []


def norm(arg, order=2):
    pass
