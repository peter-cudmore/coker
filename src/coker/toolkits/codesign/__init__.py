import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Callable, List, Optional, Tuple, cast

import numpy as np

from coker.algebra.dimensions import Dimension, FunctionSpace
from coker.algebra.kernel import (
    OP,
    SymbolicCallable,
    Tape,
    TraceContext,
    Tracer,
    VectorSpace,
    Scalar,
    function,
)
from .optimisation import (
    BoundedConstraint,
    SolveFailure,
    SolveInfo,
    WeightedNorm,
    bounded,
    weighted_norm,
)


@dataclasses.dataclass
class SolverOptions:
    warm_start: bool = False


class Minimise:
    def __init__(self, expression: Tracer):
        self.expression = expression


class MathematicalProgram(SymbolicCallable):
    """An optimisation module that maps parameters to an objective and outputs.

    ``backend`` is the solver backend selected by :class:`ProblemBuilder`.
    ``lower()`` always lowers with that same backend; a program cannot be
    embedded in a graph lowered by a different backend.
    """

    def __init__(
        self,
        input_shape: Tuple[Dimension, ...],
        output_shape: Tuple[Dimension, ...],
        implementation: Callable,
        backend: Optional[str] = None,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._impl = implementation
        self.backend = backend
        self.solve_info = None

    @property
    def result_shape(self) -> Tuple[Dimension, ...]:
        """Return the objective-first shapes produced by this program."""
        return (Dimension(None), *self.output_shape)

    def _validate_arguments(self, args) -> None:
        if len(args) != len(self.input_shape):
            raise ValueError(
                f"Expected {len(self.input_shape)} arguments, got {len(args)}"
            )
        for index, (arg, expected) in enumerate(zip(args, self.input_shape)):
            if isinstance(arg, Tracer) and arg.dim != expected:
                raise ValueError(
                    f"Argument {index} has shape {arg.dim}, expected {expected}"
                )

    def _call_numeric(self, *args):
        """Solve with concrete arguments and return objective followed by outputs."""
        self._validate_arguments(args)

        try:
            result = self._impl(*args)
        finally:
            self.solve_info = getattr(self._impl, "last_solve_info", None)
        if not isinstance(result, (list, tuple)):
            result = [result]
        if len(result) != len(self.result_shape):
            raise ValueError(
                f"Backend returned {len(result)} results for "
                f"{len(self.result_shape)} requested objective and outputs"
            )

        objective, *outputs = result
        objective_array = np.asarray(objective)
        if objective_array.size != 1:
            raise TypeError(
                "Backend returned a non-scalar optimisation objective with "
                f"shape {objective_array.shape}"
            )
        return (
            float(objective_array.reshape(-1)[0]),
            *(
                np.reshape(np.asarray(output), dim.shape)
                for output, dim in zip(outputs, self.output_shape)
            ),
        )

    def _call_symbolic(self, *args):
        """Emit objective and output evaluations on the active symbolic tape."""
        self._validate_arguments(args)
        tape = TraceContext.get_local_tape()
        if tape is None:
            raise RuntimeError(
                "symbolic program calls require an active Coker tracing context"
            )
        if self.backend is not None and tape.backend != self.backend:
            raise ValueError(
                f"MathematicalProgram uses backend {self.backend!r}, "
                f"but the enclosing graph uses {tape.backend!r}"
            )
        call = self
        arguments = [
            dim.to_space(f"input_{i}")
            for i, dim in enumerate(self.input_shape)
        ]
        results = []
        for index, output_dim in enumerate(self.result_shape):
            output = output_dim.to_space(f"output_{index}")
            space = FunctionSpace("program_output", arguments, [output])
            reference = tape._callable_reference(call, space, index)
            results.append(
                Tracer(tape, tape.append(OP.EVALUATE, reference, *args))
            )
        return tuple(results)

    def __call__(self, *args):
        """Call symbolically for tracer arguments and numerically otherwise."""
        if any(isinstance(arg, Tracer) for arg in args):
            return self._call_symbolic(*args)
        return self._call_numeric(*args)

    def lower(self):
        """Lower this program using its configured solver backend."""
        backend_name = self.backend or "numpy"
        from coker.backends import get_backend_by_name

        # Resolve early so unavailable backends fail before graph execution.
        get_backend_by_name(backend_name, set_current=False)
        return function(
            arguments=[
                dim.to_space(f"input_{i}")
                for i, dim in enumerate(self.input_shape)
            ],
            implementation=cast(Callable, self._call_symbolic),
            backend=backend_name,
        )

    def export_payload(self) -> dict[str, object]:
        """Return the deterministic artifact payload when supported."""
        exporter = getattr(self._impl, "export_payload", None)
        if exporter is None:
            raise NotImplementedError(
                "this mathematical program backend has no "
                "Coker artifact payload"
            )
        return exporter()


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
                "initial_conditions must be a mapping from tracer "
                "index to value or a sequence aligned with "
                "decision-variable declaration order"
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
                "initial_conditions sequence length does not match "
                "the number of decision variables"
            )
        return dict(zip(decision_input_indicies, self.initial_conditions))

    def build(self, backend: Optional[str] = None) -> MathematicalProgram:
        assert isinstance(self.objective, Minimise)
        assert self.tape is not None
        assert self.outputs
        from coker.backends import get_backend_by_name, get_current_backend

        backend_name = backend
        if backend_name is None:
            current = get_current_backend()
            backend_name = getattr(current, "name", None)
            if backend_name is None:
                raise RuntimeError(
                    "Current backend does not expose a stable backend name"
                )
        backend_impl = get_backend_by_name(backend_name)

        implementation = backend_impl.build_optimisation_problem(
            self.objective.expression,
            self.constraints,
            self.arguments,
            [self.objective.expression, *self.outputs],
            self._normalise_initial_conditions(),
        )
        impl = backend_impl.make_optimisation_module(implementation)

        return MathematicalProgram(
            self.input_shape, self.output_shape, impl, backend=backend_name
        )

    def __enter__(self):
        assert self.tape is not None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def norm(arg, order=2):
    return np.linalg.norm(arg, ord=order)


__all__ = [
    "BoundedConstraint",
    "MathematicalProgram",
    "Minimise",
    "ProblemBuilder",
    "SolveFailure",
    "SolveInfo",
    "SolverOptions",
    "WeightedNorm",
    "bounded",
    "norm",
    "weighted_norm",
]
