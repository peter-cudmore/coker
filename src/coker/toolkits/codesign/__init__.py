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

        return self.impl(args)


class ProblemBuilder:
    def __init__(self, arguments: Optional[List[VectorSpace | Scalar]] = None):

        self.arguments = [self.tape.input(a) for a in arguments] if arguments else []
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
                np.zeros(shape=shape) if initial_value is None else initial_value
            )

        self.initial_conditions[v] = initial_value
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
        )

        return MathematicalProgram(self.input_shape, self.output_shape, impl)

    def __enter__(self):
        self.tape = Tape()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tape = None
        pass


def norm(arg, order=2):
    pass
