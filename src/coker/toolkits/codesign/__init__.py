import dataclasses
from abc import ABC, abstractmethod
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
from coker.algebra.ops import OP, SelectOP
from coker.backends.backend import Backend, create_native_symbol_entry

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

    Args:
        warm_start: Whether a supporting backend may reuse its preceding solve.
            Defaults to ``False``. Backend-specific subclasses may add options.

    Raises:
        TypeError: If ``warm_start`` is not a bool.

    Examples:
        >>> from coker.toolkits.codesign import SolverOptions
        >>> SolverOptions().warm_start
        False
    """

    warm_start: bool = False

    def __post_init__(self):
        if not isinstance(self.warm_start, bool):
            raise TypeError("warm_start must be a bool")


class Minimise:
    """Mark a symbolic scalar expression as an objective to minimise.

    This wrapper records an expression; it neither builds nor solves a problem.
    Assign it to :attr:`ProblemBuilder.objective`. Shape support and lowering
    are determined by the selected solver backend.

    Args:
        expression: Scalar symbolic objective on the builder's tape.

    Attributes:
        expression: The supplied symbolic objective.

    Examples:
        >>> from coker.toolkits.codesign import Minimise, ProblemBuilder
        >>> builder = ProblemBuilder()
        >>> x = builder.new_variable("x")
        >>> objective = Minimise(x ** 2)
        >>> objective.expression.dim.is_scalar()
        True
    """

    def __init__(self, expression: Tracer):
        self.expression = expression


@dataclasses.dataclass(frozen=True)
class _ParameterCapture:
    """Private metadata for reconstructing a solved decision."""

    name: str
    target: Scalar | VectorSpace | FunctionSpace
    declaration: Any
    blocks: tuple[Tracer, ...]


class OptimisationModule(ABC):
    """Extension interface for numerical solvers with solve-result metadata.

    Implementations set ``last_solve_info`` before returning from a solve and
    before raising :class:`SolveFailure`. Builders supply a backend-created
    implementation to :class:`MathematicalProgram`; user models normally use
    :class:`ProblemBuilder` rather than subclassing this interface.

    Attributes:
        last_solve_info: Most recent :class:`SolveInfo`, or ``None`` before a
            solve. Failed solves must also publish their available status.

    Examples:
        >>> from coker.toolkits.codesign import OptimisationModule, SolveInfo
        >>> class ConstantSolver(OptimisationModule):
        ...     last_solve_info = None
        ...     def __call__(self, target):
        ...         self.last_solve_info = SolveInfo(
        ...             "example", "constant", True, "complete")
        ...         return (0.0, target)
        >>> solver = ConstantSolver()
        >>> solver(2.0)
        (0.0, 2.0)
        >>> solver.last_solve_info.success
        True
    """

    last_solve_info: SolveInfo | None

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        """Solve once for concrete runtime arguments.

        Args:
            *args: Runtime parameters in the builder's declaration order.

        Returns:
            Objective-first backend values, followed by requested outputs and
            any private decision-capture blocks requested by the builder.
            MathematicalProgram removes the private blocks from public results.

        Raises:
            SolveFailure: If the backend reports an unsuccessful solve.
        """


class MathematicalProgram(SymbolicCallable):
    """Callable optimisation program with objective-first public results.

    Numeric calls return ``(objective, *outputs)``. The objective is a Python
    float and outputs are NumPy arrays reshaped to each output dimension's
    physical shape. Scalar outputs have logical dimension metadata ``None``
    but physical array shape ``(1,)``. Symbolic calls emit the same ordered
    results onto the active tape.
    Programs produced by :meth:`ProblemBuilder.build` also reconstruct named
    decisions after successful numeric calls.

    Args:
        input_shape: Runtime argument dimensions in declaration order.
        output_shape: Requested output dimensions, excluding the objective.
        implementation: Numerical callable returning objective-first values.
            Backend extensions additionally supply private decision captures
            through the builder's internal construction path.
        backend: Solver backend name, or ``None`` for NumPy lowering. A
            configured program cannot be embedded in another backend's graph.

    Attributes:
        backend: Solver backend retained for symbolic calls and lowering.
        parameters: Named reconstructed decisions from the last successful
            numeric solve, initially empty. Values may be scalars, arrays or
            fitted callable parameters. Failed solves preserve previous values.
        solve_info: Latest backend solve status, initially ``None``. Builder
            programs refresh it even when a solve raises ``SolveFailure``.
            Directly supplied callables do not automatically publish status.

    Examples:
        >>> from coker import Dimension
        >>> from coker.toolkits.codesign import MathematicalProgram
        >>> program = MathematicalProgram(
        ...     (Dimension.scalar(),), (Dimension.scalar(),),
        ...     lambda value: (value ** 2, value + 1), backend="numpy")
        >>> objective, output = program(3.0)
        >>> objective, output.item()
        (9.0, 4.0)
        >>> assert output.shape == (1,)
        >>> assert output[0] == 4.0
        >>> len(program.result_shape)
        2
    """

    def __init__(
        self,
        input_shape: Tuple[Dimension, ...],
        output_shape: Tuple[Dimension, ...],
        implementation: Callable[..., Any],
        backend: Optional[str] = None,
    ):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._impl = implementation
        self.backend = backend
        self._optimisation_module: OptimisationModule | None = None
        self.solve_info = None
        self._parameter_captures: tuple[_ParameterCapture, ...] = ()
        self.parameters: dict[str, Any] = {}

    @classmethod
    def _from_optimisation(
        cls,
        input_shape: Tuple[Dimension, ...],
        output_shape: Tuple[Dimension, ...],
        implementation: OptimisationModule,
        backend: str,
        captures: Sequence[_ParameterCapture],
    ) -> "MathematicalProgram":
        program = cls(input_shape, output_shape, implementation, backend)
        program._optimisation_module = implementation
        program._parameter_captures = tuple(captures)
        return program

    def input_shape(self) -> Tuple[Dimension, ...]:
        """Return runtime argument dimensions in declaration order.

        Returns:
            Tuple of dimensions, excluding optimisation decisions.
        """
        return self._input_shape

    def output_shape(self) -> Tuple[Dimension, ...]:
        """Return requested output dimensions.

        Returns:
            Tuple of dimensions, excluding the objective and private captures.
        """
        return self._output_shape

    @property
    def result_shape(self) -> Tuple[Dimension, ...]:
        """Tuple of public result dimensions with the scalar objective first."""
        return (Dimension.scalar(), *self.output_shape())

    def _validate_arguments(self, args) -> None:
        input_shape = self.input_shape()
        if len(args) != len(input_shape):
            raise ValueError(f"Expected {len(input_shape)} arguments, got {len(args)}")
        for index, (arg, expected) in enumerate(zip(args, input_shape)):
            if isinstance(arg, Tracer) and arg.dim != expected:
                raise ValueError(
                    f"Argument {index} has shape {arg.dim}, " f"expected {expected}"
                )

    def _call_numeric(self, *args):
        """Solve with concrete arguments; return objective and outputs."""
        self._validate_arguments(args)

        try:
            result = self._impl(*args)
        finally:
            if self._optimisation_module is not None:
                self.solve_info = self._optimisation_module.last_solve_info
        if not isinstance(result, (list, tuple)):
            result = [result]
        capture_sizes = tuple(
            len(capture.blocks) for capture in self._parameter_captures
        )
        expected_results = len(self.result_shape) + sum(capture_sizes)
        if len(result) != expected_results:
            raise ValueError(
                f"Backend returned {len(result)} results for "
                f"{expected_results} requested objective, outputs, and "
                "private parameter captures"
            )

        public_result_count = len(self.result_shape)
        objective, *outputs = result[:public_result_count]
        captured_values = result[public_result_count:]
        captured_blocks = []
        offset = 0
        for size in capture_sizes:
            captured_blocks.append(tuple(captured_values[offset : offset + size]))
            offset += size
        self.parameters = self._reconstruct_parameters(captured_blocks)
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
                for output, dim in zip(outputs, self.output_shape())
            ),
        )

    def _reconstruct_parameters(
        self, captured_blocks: Sequence[tuple[Any, ...]]
    ) -> dict[str, Any]:
        """Rebuild public decision values from private solver captures."""
        from coker.backends import get_backend_by_name

        backend = get_backend_by_name(self.backend or "numpy", set_current=False)
        return {
            metadata.name: backend._materialize_parameter(
                metadata.target, metadata.declaration, blocks
            )
            for metadata, blocks in zip(self._parameter_captures, captured_blocks)
        }

    def _call_symbolic(self, *args):
        """Emit objective and output evaluations on the symbolic tape."""
        self._validate_arguments(args)
        tape = TraceContext.get_local_tape()
        if tape is None:
            raise RuntimeError(
                "symbolic program calls require an active Coker " "tracing context"
            )
        if self.backend is not None and tape.backend != self.backend:
            raise ValueError(
                f"MathematicalProgram uses backend {self.backend!r}, "
                f"but the enclosing graph uses {tape.backend!r}"
            )
        arguments = [
            dim.to_space(f"input_{i}") for i, dim in enumerate(self.input_shape())
        ]
        result_dimensions = self.result_shape
        function_space = FunctionSpace(
            "program",
            arguments,
            [
                dimension.to_space(f"result_{index}")
                for index, dimension in enumerate(result_dimensions)
            ],
        )
        symbol = tape.intern_symbol(
            create_native_symbol_entry(
                self,
                function_space,
            )
        )
        result = Tracer(tape, tape.append(OP.EVALUATE, symbol, *args))
        if len(function_space.output) == 1:
            return (result,)
        return tuple(
            Tracer(tape, tape.append(SelectOP(index), result))
            for index in range(len(result_dimensions))
        )

    def __call__(self, *args):
        """Evaluate numerically or emit a symbolic solve.

        Args:
            *args: Runtime arguments in declaration order. Any tracer argument
                selects symbolic evaluation on the active tracing context.

        Returns:
            Tuple ``(objective, *outputs)``. Numeric results contain a float
            objective and NumPy output arrays; symbolic results contain tracers.
            Private decision captures never appear in this tuple.

        Raises:
            ValueError: If argument arity or symbolic dimensions are wrong,
                symbolic backends differ, or the backend result count is wrong.
            TypeError: If the backend returns a non-scalar objective.
            RuntimeError: If symbolic evaluation has no active tracing context.
            SolveFailure: If the backend reports an unsuccessful solve.
        """
        if any(isinstance(arg, Tracer) for arg in args):
            return self._call_symbolic(*args)
        return self._call_numeric(*args)

    def lower(self):
        """Wrap the solve in a symbolic Function pinned to this backend.

        Returns:
            A :class:`coker.Function`, not a lowered execution handle. Calling
            it solves and returns only the objective and requested outputs.
            The current backend selection is unchanged.
        """
        backend_name = self.backend or "numpy"
        from coker.backends import get_backend_by_name

        # Resolve early so unavailable backends fail before graph execution.
        get_backend_by_name(backend_name, set_current=False)
        return function(
            arguments=[
                dim.to_space(f"input_{i}") for i, dim in enumerate(self.input_shape())
            ],
            implementation=cast(Callable, self._call_symbolic),
            backend=backend_name,
        )


class ProblemBuilder:
    """Construct an ordinary mathematical program on one symbolic tape.

    Runtime ``arguments`` are supplied anew to each solve; decisions declared
    with ``new_variable`` or ``new_function_parameter`` are chosen by the
    optimiser. A context manager is optional: entry returns this builder, and
    exit neither builds nor solves, clears the tape, nor suppresses exceptions.
    Configuration remains available after leaving the context.

    Args:
        arguments: Ordered scalar or vector spaces for runtime parameters.
            Defaults to no runtime parameters.
        solver_options: Backend-specific solver configuration, or ``None``.
            Each backend determines which option types it supports.

    Attributes:
        arguments: Symbolic runtime parameters in declaration order.
        objective: A :class:`Minimise` wrapping a scalar symbolic expression.
        constraints: Symbolic comparisons or :class:`BoundedConstraint` objects.
            Defaults to no constraints.
        outputs: Non-empty list of symbolic expressions returned after the
            objective. All problem expressions must use this builder's tape.
        initial_conditions: Initial guesses, normally a mapping from decision
            tracer indices to values. May be replaced with a sequence aligned
            with concrete decision declaration order, excluding runtime inputs.
        solver_options: Configuration forwarded to the selected backend.

    Examples:
        >>> from coker import Scalar
        >>> from coker.toolkits.codesign import Minimise, ProblemBuilder
        >>> with ProblemBuilder(arguments=[Scalar("target")]) as builder:
        ...     target = builder.arguments[0]
        ...     x = builder.new_variable("x", initial_value=0.0)
        ...     builder.objective = Minimise((x - target) ** 2)
        ...     builder.outputs = [x]
        ...     program = builder.build("numpy")
        >>> objective, solution = program(2.0)
        >>> abs(objective) < 1e-10 and abs(solution.item() - 2.0) < 1e-5
        True
        >>> bool(abs(program.parameters["x"] - 2.0) < 1e-5)
        True
        >>> program.solve_info.success
        True
    """

    def __init__(
        self,
        arguments: Optional[List[VectorSpace | Scalar]] = None,
        *,
        solver_options: SolverOptions | None = None,
    ):
        self.tape: Optional[Tape] = Tape()
        self.arguments = [self.tape.input(a) for a in arguments] if arguments else []
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
                np.zeros(shape=shape) if initial_value is None else initial_value
            )
        self.initial_conditions[variable.index] = initial_value
        return variable

    def new_variable(self, name, shape=None, initial_value=None):
        """Declare a scalar or array decision and record its initial guess.

        Args:
            name: Decision name used in the solved program's ``parameters``.
            shape: Array shape, or ``None`` to declare a scalar.
            initial_value: Initial guess. Defaults to scalar zero or an array
                of zeros matching ``shape``.

        Returns:
            Symbolic decision tracer owned by this builder's tape.
        """
        variable = self._add_decision(name, shape, initial_value)
        space = variable.dim.to_space(name)
        self._parameter_captures.append(
            _ParameterCapture(name, space, space, (variable,))
        )
        return variable

    def new_function_parameter(self, target, declaration):
        """Declare a finite function parameter as optimisation decisions.

        Concrete blocks, guesses and any bounds come from the declaration.
        After a successful numeric solve, the program's ``parameters`` contains
        the reconstructed callable under the declaration's name.

        Args:
            target: FunctionSpace describing the callable's inputs and outputs.
            declaration: FunctionParameter providing finite concrete decisions.

        Returns:
            BoundCallable usable symbolically with the declared target.

        Raises:
            TypeError: If target or declaration has the wrong type, or concrete
                declarations are not supported scalar, vector or tensor types.
            ValueError: If a declaration name is empty or its target is
                incompatible with the supplied function space.

        Examples:
            >>> from coker import FunctionSpace, Scalar
            >>> from coker.parameters.function_parameters import MonotonePiecewiseLinear
            >>> from coker.toolkits.codesign import ProblemBuilder
            >>> target = FunctionSpace(
            ...     "response", arguments=[Scalar("x")], output=[Scalar("y")])
            >>> declaration = MonotonePiecewiseLinear(
            ...     domain_knots=[0.0, 1.0], lower_bound=0.0,
            ...     upper_bound=1.0, name="response")
            >>> builder = ProblemBuilder()
            >>> response = builder.new_function_parameter(target, declaration)
            >>> response(0.5).dim.is_scalar()
            True
        """
        from coker.parameters.function_parameters import FunctionParameter
        from coker.parameters import (
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
            raise ValueError("function parameter name must be a non-empty string")
        target = declaration.validate_target(target)
        concrete_declarations = declaration.list_concrete_parameters()
        concrete_values = []
        for concrete in concrete_declarations:
            if not isinstance(concrete.name, str) or not concrete.name:
                raise ValueError(
                    "concrete parameter declaration name must be non-empty"
                )
            if isinstance(concrete, (BoundedVariable, UnboundedVariable)):
                value = self._add_decision(concrete.name, initial_value=concrete.guess)
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
        self._parameter_captures.append(
            _ParameterCapture(
                declaration.name,
                target,
                declaration,
                tuple(concrete_values),
            )
        )
        return BoundCallable(
            declaration.build_function(target, None),
            target,
            tuple(concrete_values),
        )

    @property
    def input_shape(self) -> Tuple[Dimension, ...]:
        """Tuple of runtime argument dimensions, excluding solver decisions."""
        return tuple(i.dim for i in self.arguments)

    @property
    def output_shape(self) -> Tuple[Dimension, ...]:
        """Tuple of requested output dimensions, excluding the objective."""
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
        """Build a solver-backed callable without performing a solve.

        Args:
            backend: Explicit backend name, or ``None`` to use the current
                backend. Building does not change the current selection.

        Returns:
            MathematicalProgram returning ``(objective, *outputs)`` per solve.

        Raises:
            TypeError: If a present objective is not Minimise, or
                ``initial_conditions`` is neither a mapping nor a sequence.
                Backends also reject unsupported solver option types.
            ValueError: If the objective is missing, outputs are empty, or an
                initial-condition sequence has the wrong number of decisions.

        Backend-specific construction errors propagate without a solve.
        """
        if self.objective is None:
            raise ValueError("An objective is required to build a problem")
        if not isinstance(self.objective, Minimise):
            raise TypeError("The objective must be a Minimise expression")
        assert self.tape is not None
        if not self.outputs:
            raise ValueError("At least one output is required to build a problem")
        from coker.backends import get_backend_by_name, get_current_backend

        if backend is None:
            current: Backend = get_current_backend()
            backend_name = current.name
        else:
            backend_name = backend
        backend_impl = get_backend_by_name(backend_name, set_current=False)

        implementation = backend_impl.build_optimisation_problem(
            self.objective.expression,
            self.constraints,
            self.arguments,
            [
                self.objective.expression,
                *self.outputs,
                *(
                    block
                    for capture in self._parameter_captures
                    for block in capture.blocks
                ),
            ],
            self._normalise_initial_conditions(),
            options=self.solver_options,
        )
        impl = cast(
            OptimisationModule,
            backend_impl.make_optimisation_module(implementation),
        )

        return MathematicalProgram._from_optimisation(
            self.input_shape,
            self.output_shape,
            impl,
            backend_name,
            self._parameter_captures,
        )

    def __enter__(self):
        """Enter the optional configuration context.

        Returns:
            This builder, without activating a global tracing context.
        """
        assert self.tape is not None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave configuration intact and allow exceptions to propagate.

        Args:
            exc_type: Exception type supplied by the context protocol, if any.
            exc_val: Exception instance, if any.
            exc_tb: Exception traceback, if any.

        Returns:
            None, so exceptions are not suppressed.
        """
        pass


def norm(arg, order=2):
    """Return an unsquared NumPy-style norm of a numeric or symbolic value.

    Args:
        arg: Numeric array or symbolic vector.
        order: NumPy norm order, defaulting to ``2``. Symbolic vectors support
            orders ``1`` and ``2``; other cases depend on backend support.

    Returns:
        Numeric norm or symbolic scalar expression from ``numpy.linalg.norm``.

    Examples:
        >>> from coker.toolkits.codesign import norm
        >>> float(norm([3.0, 4.0]))
        5.0
    """
    return np.linalg.norm(arg, ord=order)


__all__ = [
    "BoundedConstraint",
    "MathematicalProgram",
    "OptimisationModule",
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
