import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Callable, List, Optional, Tuple, cast

import numpy as np

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.function import BoundCallable, SymbolicCallable, function
from coker.algebra.graph import Tape, TraceContext, Tracer
from coker.algebra.ops import OP
from .optimisation import (
    BoundedConstraint,
    SolveFailure,
    SolveInfo,
    WeightedNorm,
    bounded,
    weighted_norm,
)


@dataclasses.dataclass(frozen=True)
class SolverOptions:
    """Backend-independent settings for nonlinear-program solvers.

    Backend implementations define subclasses for their algorithm, precision,
    and device settings. ODE initial-value settings use the separate
    :class:`coker.interfaces.SolverParameters` hierarchy; variational solvers
    expose a dedicated options type only when they require configuration beyond
    their problem definition.
    """

    warm_start: bool = False

    def __post_init__(self):
        if not isinstance(self.warm_start, bool):
            raise TypeError("warm_start must be a bool")


class Minimise:
    def __init__(self, expression: Tracer):
        self.expression = expression


@dataclasses.dataclass(frozen=True)
class _ParameterCapture:
    """Private metadata for reconstructing a solved decision."""

    name: str
    target: Scalar | VectorSpace | FunctionSpace
    declaration: Any
    capture: Tracer


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
        self._parameter_captures: tuple[_ParameterCapture, ...] = ()
        self.parameters: dict[str, Any] = {}

    @classmethod
    def _from_optimisation(
        cls,
        input_shape: Tuple[Dimension, ...],
        output_shape: Tuple[Dimension, ...],
        implementation: Callable,
        backend: str,
        captures: Sequence[_ParameterCapture],
    ) -> "MathematicalProgram":
        program = cls(input_shape, output_shape, implementation, backend)
        program._parameter_captures = tuple(captures)
        return program

    @property
    def result_shape(self) -> Tuple[Dimension, ...]:
        """Return the objective-first shapes produced by this program."""
        return (Dimension.scalar(), *self.output_shape)

    def _validate_arguments(self, args) -> None:
        if len(args) != len(self.input_shape):
            raise ValueError(
                f"Expected {len(self.input_shape)} arguments, got {len(args)}"
            )
        for index, (arg, expected) in enumerate(zip(args, self.input_shape)):
            if isinstance(arg, Tracer) and arg.dim != expected:
                raise ValueError(
                    f"Argument {index} has shape {arg.dim}, "
                    f"expected {expected}"
                )

    def _call_numeric(self, *args):
        """Solve with concrete arguments; return objective and outputs."""
        self._validate_arguments(args)

        try:
            result = self._impl(*args)
        finally:
            self.solve_info = getattr(self._impl, "last_solve_info", None)
        if not isinstance(result, (list, tuple)):
            result = [result]
        expected_results = len(self.result_shape) + len(
            self._parameter_captures
        )
        if len(result) != expected_results:
            raise ValueError(
                f"Backend returned {len(result)} results for "
                f"{expected_results} requested objective, outputs, and "
                "private parameter captures"
            )

        public_result_count = len(self.result_shape)
        objective, *outputs = result[:public_result_count]
        captured_values = result[public_result_count:]
        self.parameters = self._reconstruct_parameters(captured_values)
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

    def _reconstruct_parameters(
        self, captured_values: Sequence[Any]
    ) -> dict[str, Any]:
        """Rebuild public decision values from private solver captures."""
        from coker.backends import get_backend_by_name

        backend = get_backend_by_name(
            self.backend or "numpy", set_current=False
        )
        result = {}
        for metadata, value in zip(self._parameter_captures, captured_values):
            if isinstance(metadata.target, FunctionSpace):
                result[metadata.name] = backend.fit_function_parameter(
                    metadata.declaration, metadata.target, value
                )
            elif isinstance(metadata.target, Scalar):
                result[metadata.name] = float(np.asarray(value).reshape(-1)[0])
            else:
                result[metadata.name] = np.asarray(value).reshape(
                    metadata.target.dimension
                )
        return result

    def _call_symbolic(self, *args):
        """Emit objective and output evaluations on the symbolic tape."""
        self._validate_arguments(args)
        tape = TraceContext.get_local_tape()
        if tape is None:
            raise RuntimeError(
                "symbolic program calls require an active Coker "
                "tracing context"
            )
        if self.backend is not None and tape.backend != self.backend:
            raise ValueError(
                f"MathematicalProgram uses backend {self.backend!r}, "
                f"but the enclosing graph uses {tape.backend!r}"
            )
        arguments = [
            dim.to_space(f"input_{i}")
            for i, dim in enumerate(self.input_shape)
        ]
        result_dimensions = self.result_shape
        result_space = VectorSpace(
            "program_result", sum(dim.flat() for dim in result_dimensions)
        )
        function_space = FunctionSpace("program", arguments, [result_space])
        reference = tape._create_callable_reference(
            self,
            function_space,
            function_space.output_dimensions()[0],
        )
        packed = Tracer(tape, tape.append(OP.EVALUATE, reference, *args))
        offset = 0
        results = []
        for dim in result_dimensions:
            if dim.is_scalar():
                results.append(packed[offset])
            else:
                results.append(
                    np.reshape(packed[offset : offset + dim.flat()], dim.dim)
                )
            offset += dim.flat()
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


class ProblemBuilder:
    def __init__(
        self,
        arguments: Optional[List[VectorSpace | Scalar]] = None,
        *,
        solver_options: SolverOptions | None = None,
    ):
        self.tape: Optional[Tape] = Tape()
        self.arguments = (
            [self.tape.input(a) for a in arguments] if arguments else []
        )
        self.objective = None
        self.constraints = []
        self.outputs = []
        self.initial_conditions = {}
        self.solver_options = solver_options
        self._parameter_captures: list[_ParameterCapture] = []

    def _add_decision(self, name, shape=None, initial_value=None):
        assert self.tape is not None
        if shape is None:
            variable = self.tape.input(Scalar(name))
            initial_value = 0 if initial_value is None else initial_value
        else:
            variable = self.tape.input(VectorSpace(name, shape))
            initial_value = (
                np.zeros(shape=shape)
                if initial_value is None
                else initial_value
            )
        self.initial_conditions[variable.index] = initial_value
        return variable

    def new_variable(self, name, shape=None, initial_value=None):
        variable = self._add_decision(name, shape, initial_value)
        space = variable.dim.to_space(name)
        self._parameter_captures.append(
            _ParameterCapture(name, space, space, variable)
        )
        return variable

    def new_function_parameter(self, target, declaration):
        """Add a finite function parameter as optimisation decisions."""
        from coker.dynamics.function_parameters import FunctionParameter
        from coker.dynamics.variables import (
            BoundedVariable,
            BoundVector,
            DenseTensorVariable,
            UnboundedVariable,
        )

        if not isinstance(target, FunctionSpace):
            raise TypeError("target must be a FunctionSpace")
        if not isinstance(declaration, FunctionParameter):
            raise TypeError("declaration must implement FunctionParameter")
        if not isinstance(declaration.name, str) or not declaration.name:
            raise ValueError(
                "function parameter name must be a non-empty string"
            )
        target = declaration.validate_target(target)
        concrete_declarations = declaration.list_concrete_parameters()
        concrete_values = []
        for concrete in concrete_declarations:
            if not isinstance(concrete.name, str) or not concrete.name:
                raise ValueError(
                    "concrete parameter declaration name must be non-empty"
                )
            if isinstance(concrete, (BoundedVariable, UnboundedVariable)):
                value = self._add_decision(
                    concrete.name, initial_value=concrete.guess
                )
            elif isinstance(concrete, (BoundVector, DenseTensorVariable)):
                value = self._add_decision(
                    concrete.name,
                    shape=concrete.shape,
                    initial_value=concrete.guess,
                )
            else:
                raise TypeError(
                    "function parameter concrete declarations must be scalar, "
                    "vector, or dense tensor variables"
                )
            if isinstance(concrete, (BoundedVariable, BoundVector)):
                self.constraints.append(
                    bounded(value, concrete.lower_bound, concrete.upper_bound)
                )
            concrete_values.append(value)
        capture = np.concatenate(
            [np.reshape(value, (-1,)) for value in concrete_values]
        )
        self._parameter_captures.append(
            _ParameterCapture(declaration.name, target, declaration, capture)
        )
        return BoundCallable(
            declaration.build_function(target, None),
            target,
            tuple(concrete_values),
        )

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
            [
                self.objective.expression,
                *self.outputs,
                *(capture.capture for capture in self._parameter_captures),
            ],
            self._normalise_initial_conditions(),
            options=self.solver_options,
        )
        impl = backend_impl.make_optimisation_module(implementation)

        return MathematicalProgram._from_optimisation(
            self.input_shape,
            self.output_shape,
            impl,
            backend_name,
            self._parameter_captures,
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
