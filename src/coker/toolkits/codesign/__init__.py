import dataclasses
from collections.abc import Mapping, Sequence
from typing import Optional, List, Tuple, Callable, Any

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
        self.solve_info = None

    def __call__(self, *args):
        if len(args) != len(self.input_shape):
            raise ValueError(
                f"Expected {len(self.input_shape)} runtime arguments, got {len(args)}"
            )

        try:
            result = self.impl(*args)
        finally:
            self.solve_info = getattr(self.impl, "last_solve_info", None)
        if not isinstance(result, (list, tuple)):
            result = [result]
        if len(result) != len(self.output_shape):
            raise ValueError(
                f"Backend returned {len(result)} outputs for "
                f"{len(self.output_shape)} requested outputs"
            )

        return [
            np.reshape(np.asarray(o), dim.shape)
            for o, dim in zip(result, self.output_shape)
        ]


class ProblemBuilder:
    def __init__(self, arguments: Optional[List[VectorSpace | Scalar]] = None):
        self.tape: Optional[Tape] = Tape()
        self.arguments = (
            [self.tape.input(a) for a in arguments] if arguments else []
        )
        self.objective = None
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

    def _normalise_initial_conditions(self) -> dict[int, Any]:
        if isinstance(self.initial_conditions, Mapping):
            return dict(self.initial_conditions)
        if not isinstance(self.initial_conditions, Sequence):
            raise TypeError(
                "initial_conditions must be a mapping from tracer index to value "
                "or a sequence aligned with decision-variable declaration order"
            )
        assert self.tape is not None
        parameter_indicies = {argument.index for argument in self.arguments}
        decision_input_indicies = [
            index
            for index in self.tape.input_indicies
            if index not in parameter_indicies
        ]
        if len(self.initial_conditions) != len(decision_input_indicies):
            raise ValueError(
                "initial_conditions sequence length does not match the number of "
                "decision variables"
            )
        return dict(zip(decision_input_indicies, self.initial_conditions))

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
            self._normalise_initial_conditions(),
        )

        return MathematicalProgram(self.input_shape, self.output_shape, impl)

    def __enter__(self):
        assert self.tape is not None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def norm(arg, order=2):
    return np.linalg.norm(arg, ord=order)
