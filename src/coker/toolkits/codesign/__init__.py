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
        warm_start: Request reuse of solver state on later solves. Support is
            backend-specific; the current CasADi builder always uses its
            original initial guess.

    Raises:
        TypeError: If ``warm_start`` is not a bool.
    """

    warm_start: bool = False

    def __post_init__(self):
        if not isinstance(self.warm_start, bool):
            raise TypeError("warm_start must be a bool")


class Minimise:
    """Declare a scalar objective to minimise with :class:`ProblemBuilder`.

    Args:
        expression: Scalar symbolic expression on the builder's tape. A vector
            residual must first be reduced, for example with ``np.dot(r, r)``.
            This wrapper records the expression; it does not validate it or
            run a solver.

    Attributes:
        expression: The supplied symbolic objective.
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
    """Advanced extension interface for numerical optimisation implementations.

    Ordinary users obtain :class:`MathematicalProgram` from
    :class:`ProblemBuilder`; they do not need to subclass this interface.
    Backend extensions must set ``last_solve_info`` before returning from a
    solve and before raising :class:`SolveFailure`.

    Attributes:
        last_solve_info: Most recent normalised solve outcome, or None before
            any outcome is available.
    """

    last_solve_info: SolveInfo | None

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        """Solve with concrete runtime arguments.

        Args:
            *args: Runtime inputs in the declared order.

        Returns:
            Objective-first results, followed by requested outputs and any
            private decision captures required by the builder.

        Raises:
            SolveFailure: If the solver reports failure. The exception must
                carry the same metadata as ``last_solve_info``.
        """


class MathematicalProgram(SymbolicCallable):
    """Callable mathematical program with objective-first public results.

    Prefer :meth:`ProblemBuilder.build` for solver-backed construction. Direct
    construction wraps a supplied callable and does not create an optimiser.
    Concrete calls return ``(objective, *outputs)`` even for one output; the
    objective is a float and each output is a NumPy array with its declared
    shape (including shape ``(1,)`` for a scalar output).

    Args:
        input_shape: Runtime argument dimensions in positional order.
        output_shape: Requested output dimensions, excluding the objective.
        implementation: Concrete callable returning objective then outputs.
        backend: Backend name used for symbolic composition and ``lower()``.
            Directly constructed programs default to NumPy when lowered;
            builder-created programs retain their selected solver backend.

    Attributes:
        backend: Selected solver backend, or None for an unspecified direct
            wrapper.
        parameters: Dictionary of named decisions reconstructed after a
            successful concrete call. Initially empty; replaced on each
            success, but retained on solver failure. NumPy and CasADi restore
            scalars as floats and tensors as shaped NumPy arrays; PyTorch
            preserves native tensors. Function decisions are
            :class:`coker.parameters.function_parameters.FittedFunction`
            instances. Private decision captures never enter the public tuple.
        solve_info: Latest backend outcome, initially None. Updated on success
            and :class:`SolveFailure` for builder-created programs. Argument
            errors or exceptions raised before the backend records an outcome
            may leave the previous value intact. Direct callable wrappers
            have no solver metadata.

    Notes:
        Result arrays and the mutable ``parameters`` mapping are returned
        without an isolation/copy guarantee. Copy values if retaining a
        snapshot while mutating results. The program owns its current result
        metadata; it is not a per-call history or a concurrent-solve interface.
        Symbolic calls create graph nodes rather than immediately solving.
        Backend support for executing such nested solves is separate from
        support for constructing a standalone optimisation problem.
        ``input_shape()`` and ``output_shape()`` return dimension tuples;
        unlike the builder's corresponding properties, they are methods.
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
        """Return runtime argument shapes in declaration order."""
        return self._input_shape

    def output_shape(self) -> Tuple[Dimension, ...]:
        """Return requested output shapes, excluding the objective."""
        return self._output_shape

    @property
    def result_shape(self) -> Tuple[Dimension, ...]:
        """Return the objective-first shapes produced by this program."""
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
        """Solve numerically or compose symbolically for tracer arguments.

        Args:
            *args: Positional runtime inputs matching ``input_shape()``.

        Returns:
            Tuple ``(objective, *outputs)``. Concrete calls return a float
            objective and NumPy arrays; symbolic calls return tracers.

        Raises:
            ValueError: If argument count or symbolic dimensions do not match,
                or symbolic composition uses a different backend.
            RuntimeError: If a symbolic call has no active tracing context.
            SolveFailure: If a solver reports an unsuccessful numeric solve.
                ``solve_info`` records the failure while ``parameters`` keeps
                the preceding successful values (or remains empty).
            TypeError: If the implementation returns a non-scalar objective.
        """
        if any(isinstance(arg, Tracer) for arg in args):
            return self._call_symbolic(*args)
        return self._call_numeric(*args)

    def lower(self):
        """Wrap this program in a Coker :class:`~coker.Function`.

        Unlike ``Function.lower()``, this method takes no lowering options and
        returns a Function, not a backend ``LoweredFunction`` handle. The
        wrapper uses this program's backend, or NumPy for a direct wrapper
        with no backend. It cannot switch the solver to another backend.

        Returns:
            Function with objective-first outputs. Backend-native support for
            calling an optimiser from a graph is required for execution.
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
    """Build a host-side mathematical program over symbolic decisions.

    Declare decisions, assign a scalar ``Minimise`` objective, optionally
    assign constraints, and assign at least one symbolic output before
    calling ``build``. Construction creates the tape immediately. The context
    manager is a convenience: entering returns this builder and exiting does
    not solve, build, close the tape, or freeze the configuration. ``build``
    does not consume the builder; later builds use its current configuration.
    Construct symbolic expressions from the builder's own tracers.

    Args:
        arguments: Scalar or vector/tensor spaces for runtime inputs, in
            positional call order. None creates a zero-input program.
        solver_options: Optional settings accepted by the chosen backend.
            NumPy accepts only None; supplied CasADi and PyTorch settings must
            use their respective NLP options classes, not ODE SolverParameters.

    Attributes:
        arguments: Symbolic runtime inputs, not optimisation decisions.
        objective: Required :class:`Minimise` wrapping a scalar expression.
        constraints: Sequence of symbolic comparisons or
            :class:`BoundedConstraint` objects. Empty by default. Solvers treat
            strict symbolic inequalities as non-strict numerical bounds.
        outputs: Required nonempty sequence of symbolic result expressions.
        initial_conditions: Initial guesses, defaulting to the mapping
            populated by decision declarations. May instead be a sequence in
            decision declaration order, excluding runtime arguments.
        solver_options: Backend-specific solver construction settings.
        tape: Tape owning the builder's expressions; must exist when building.

    Examples:
        >>> import numpy as np
        >>> from coker.toolkits.codesign import Minimise, ProblemBuilder
        >>> with ProblemBuilder() as builder:
        ...     x = builder.new_variable("x")
        ...     builder.objective = Minimise((x - 2.0) ** 2)
        ...     builder.outputs = [x]
        ...     program = builder.build("numpy")
        >>> objective, value = program()
        >>> np.testing.assert_allclose(value, [2.0], atol=1e-6)
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
        """Declare a scalar, vector, or tensor optimisation decision.

        Args:
            name: Decision name used as a key in ``program.parameters``.
            shape: Tuple of dimensions, or None for a scalar.
            initial_value: Initial guess, defaulting to scalar zero or a
                zero-filled array with ``shape``.

        Returns:
            Symbolic decision tracer for objectives, constraints, and outputs.
        """
        variable = self._add_decision(name, shape, initial_value)
        space = variable.dim.to_space(name)
        self._parameter_captures.append(
            _ParameterCapture(name, space, space, (variable,))
        )
        return variable

    def new_function_parameter(self, target, declaration):
        """Declare a function realised by finite optimisation decisions.

        Args:
            target: FunctionSpace specifying argument and output spaces.
            declaration: FunctionParameter defining the finite realisation,
                its nonempty name, concrete decisions, and initial guesses.
                Bounded concrete declarations also add builder constraints.

        Returns:
            Bound symbolic callable. After a successful numeric solve,
            ``program.parameters[declaration.name]`` is a FittedFunction.

        Raises:
            TypeError: If target or declaration has the wrong type, or its
                concrete declarations are unsupported.
            ValueError: If names are empty or the target is incompatible with
                the declaration.
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
        """Runtime input dimensions in argument declaration order."""
        return tuple(i.dim for i in self.arguments)

    @property
    def output_shape(self) -> Tuple[Dimension, ...]:
        """Requested output dimensions, excluding the scalar objective."""
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
        """Construct a solver-backed program from the current configuration.

        No solve occurs here. The result retains its solver backend; all
        expressions must belong to this builder's tape. The objective must be
        scalar and outputs must be nonempty. Backend-specific shape and
        operation restrictions still apply.

        Args:
            backend: Registered backend name, or None to use the current
                backend. Current standalone NLP implementations are CasADi
                (IPOPT), NumPy (SciPy trust-constr), and PyTorch.

        Returns:
            MathematicalProgram accepting ``arguments`` in declaration order
            and returning ``(objective, *outputs)`` on a concrete solve.

        Raises:
            ValueError: If no objective or no outputs are configured, or an
                initial-condition sequence has the wrong length.
            TypeError: If the objective is not Minimise, initial conditions
                have an unsupported type, or solver options are incompatible.
            RuntimeError: If the builder has no tape.
            NotImplementedError: If the backend does not support optimisation.
                JAX and SymPy do not currently provide standalone NLP solves.
        """
        if self.objective is None:
            raise ValueError("An objective is required before building")
        if not isinstance(self.objective, Minimise):
            raise TypeError("objective must be a Minimise instance")
        if self.tape is None:
            raise RuntimeError("Cannot build a problem without a tape")
        if not self.outputs:
            raise ValueError("At least one output is required before building")
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
        """Return this builder; entering does not start a solver."""
        assert self.tape is not None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave the convenience context without consuming the builder."""
        pass


def norm(arg, order=2):
    """Return a symbolic or numeric norm using NumPy's norm operation.

    Args:
        arg: Residual array or symbolic tracer.
        order: Norm order passed to ``np.linalg.norm``. Supported symbolic
            orders and array ranks depend on the chosen backend.

    Returns:
        Norm value or tracer. This is not a squared norm; use
        :func:`weighted_norm` for a structured squared residual.
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
