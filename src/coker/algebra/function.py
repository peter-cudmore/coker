from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple


from coker.interfaces import FunctionSignatureValue, SymbolicCallable
import numpy as np

from coker.algebra.dimensions import (
    Dimension,
    Element,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.graph import (
    DanglingTracerError,
    SymbolEntry,
    Tape,
    TraceContext,
    Tracer,
    _find_closure_tracers,
    get_dim_by_class,
    strip_symbols_from_array,
)
from coker.algebra.ops import OP, Noop
from coker.algebra.tensor import SymbolicVector

from coker.backends.backend import (
    _restore_output_slots,
    get_backend_by_name,
    get_current_backend,
)
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
    LoweredFunction,
    LoweringOptions,
)


class Function(SymbolicCallable, FunctionSignatureValue):
    """A traced callable with backend-specific concrete evaluation.

    Prefer :func:`function` to trace a Python implementation. A direct
    constructor wraps an existing tape without tracing it again. Concrete
    calls evaluate through the selected backend; calls containing tracers
    compose with the enclosing tape.

    Attributes:
        tape: Recorded computation graph.
        backend: Backend name used for concrete evaluation.
        output: Ordered output tracers, including absent ``None`` slots.
        signature: Named input and output metadata.
        name: Optional human-readable function name.

    Examples:
        >>> from coker import Scalar, function
        >>> square = function([Scalar("x")], lambda x: x * x)
        >>> float(square(3.0))
        9.0
    """

    INLINE_SIZE = 10

    def __init__(
        self,
        tape: Tape,
        outputs: Tracer | None | Sequence[Tracer | None],
        backend: str = "numpy",
        name: str | None = None,
        signature: FunctionSignature | None = None,
    ) -> None:
        """Wrap a recorded computation and its public output slots.

        Args:
            tape: Tape containing the input declarations and computation.
            outputs: One output tracer or ``None``, or a sequence of output
                slots. A sequence preserves tuple-return semantics, even for
                one output; a single slot produces a bare return value.
            backend: Registered backend name for evaluation.
            name: Optional display name.
            signature: Explicit signature metadata, inferred from the tape
                and outputs when omitted.
        """
        self.name = name
        self.tape = tape
        self.backend = backend
        self._lowered_cache: dict[tuple[int, LoweringOptions], LoweredFunction] = {}
        self.output: list[Tracer | None]
        if isinstance(outputs, Tracer) or outputs is None:
            self.output = [outputs]
            self.is_single = True
        else:
            self.output = list(outputs)
            self.is_single = False
        self._native_callable: Callable[..., Any] | None = None
        if signature is None:
            signature = FunctionSignature(
                inputs=tuple(
                    FunctionInputSpec(name, space)
                    for name, space in zip(self.arguments, self.input_spaces())
                ),
                outputs=tuple(
                    FunctionOutputSpec(f"output_{index}", shape)
                    for index, shape in enumerate(self.output_shape())
                ),
            )
        self.signature = signature

    @property
    def arguments(self) -> list[str]:
        """list[str]: Copy of input names in declaration order."""
        return self.tape.input_names.copy()

    def __repr__(self):
        """Return the display name and input/output shape metadata."""
        name = self.name if self.name else "<unknown>"
        try:
            out_shape = f"{self.output_shape()}"
        except AttributeError:
            out_shape = f"{self.output}"
        return f"{name}:{self.input_shape()} -> {out_shape}"

    def input_spaces(self) -> list[Scalar | VectorSpace | FunctionSpace]:
        """Return the argument spaces of this function as a list.

        Returns:
            A list of :class:`~coker.algebra.dimensions.Scalar`,
            :class:`~coker.algebra.dimensions.VectorSpace`, or
            :class:`~coker.algebra.dimensions.FunctionSpace` objects in
            argument order.
        """
        return list(self.tape.list_inputs())

    def input_shape(self) -> tuple[Dimension | FunctionSpace | None, ...]:
        """Return input shapes, preserving absent and function-valued slots.

        Returns:
            Tuple in declaration order. Numeric entries are dimensions;
            function-valued entries are function spaces and absent entries
            retain their ``None`` or no-op declaration.
        """
        special_inputs = {
            Tape.NONE: None,
            Tape.MAP_TO_NONE: Noop().cast_to_function_space(None),
        }

        return tuple(
            self.tape.dim[i] if i >= 0 else special_inputs[i]
            for i in self.tape.input_indicies
        )

    def output_shape(self) -> tuple[Dimension | FunctionSpace | None, ...]:
        """Return output shapes in declaration order.

        Returns:
            Tuple of dimensions or function spaces, with ``None`` for each
            absent output. This is always a tuple, even for a bare output.
        """
        return tuple(o.dim if o is not None else None for o in self.output)

    def _symbol_entry(self) -> SymbolEntry:
        function_space = FunctionSpace(
            self.name or "function",
            arguments=[
                input_spec.space
                for input_spec in self.signature.inputs
                if input_spec.space is not None
                and not isinstance(input_spec.space, Noop)
            ],
            output=[
                (
                    output_spec.shape
                    if isinstance(
                        output_spec.shape,
                        (Scalar, VectorSpace, FunctionSpace),
                    )
                    else output_spec.shape.to_space(output_spec.name)
                )
                for output_spec in self.signature.outputs
                if output_spec.shape is not None
            ],
        )
        return SymbolEntry(
            self,
            function_space,
            self.name,
            ("coker", id(self)),
        )

    def _emit_symbol_value(self, tape: Tape) -> Tracer:
        """Emit this unbound target as a symbol value on ``tape``."""
        return tape.insert_symbol_value(tape.intern_symbol(self._symbol_entry()))

    def _append_symbolic_call(
        self, tape: Tape, inputs: Sequence[Any]
    ) -> list[Tracer | None]:
        """Record a call to this Coker target on ``tape``."""
        if len(inputs) != len(self.tape.input_indicies):
            raise TypeError(
                f"Expected {len(self.tape.input_indicies)} inputs, got "
                f"{len(inputs)}"
            )
        arguments = [
            value
            for value, spec in zip(inputs, self.signature.inputs)
            if spec.space is not None and not isinstance(spec.space, Noop)
        ]
        symbol = tape.intern_symbol(self._symbol_entry())
        bundle = Tracer(
            tape,
            tape.append(OP.EVALUATE, symbol, *arguments),
        )
        return _restore_output_slots(tape, bundle, self.signature.outputs)

    def _prepare_argument(self, arg, index):
        if index == Tape.MAP_TO_NONE:
            return Noop()
        elif index == Tape.NONE:
            return None

        elif isinstance(self.tape.dim[index], FunctionSpace):
            if isinstance(arg, SymbolicCallable):
                return arg

            try:
                return function(self.tape.dim[index].arguments, arg, self.backend)
            except DanglingTracerError as ex:
                return self._lift_closure(
                    arg, self.tape.dim[index], ex
                )  # arg is a Python callable

        return arg

    def _lift_closure(
        self, fn, space: FunctionSpace, ex: DanglingTracerError
    ) -> BoundCallable:
        """Re-trace ``fn`` with captured outer-tape tracers as inputs.

        The resulting target accepts both its public arguments and the
        captured values explicitly. ``BoundCallable`` then records the
        public ``FunctionSpace`` and supplies those bound tail arguments.
        """
        captured = _find_closure_tracers(fn)
        if not captured:
            raise NotImplementedError from ex

        unique_captured: List[Tracer] = list(captured.values())

        extra_spaces = []
        for i, t in enumerate(unique_captured):
            dim = t.dim
            if not isinstance(dim, Dimension):
                raise NotImplementedError(
                    "Capturing FunctionSpace-typed tracers is not supported"
                ) from ex
            extra_spaces.append(dim.to_space(f"_cap_{i}"))

        all_spaces = list(space.arguments) + extra_spaces
        inner_tape = Tape()
        all_inner_args = [inner_tape.input(v) for v in all_spaces]
        orig_inner_args = all_inner_args[: len(space.arguments)]
        cap_inner_args = all_inner_args[len(space.arguments) :]

        for outer_t, inner_t in zip(unique_captured, cap_inner_args):
            inner_tape.add_substitution(outer_t, inner_t)

        with TraceContext(inner_tape):
            output = fn(*orig_inner_args)
            result = _normalise_result(output, inner_tape)

        inner_fn = Function(inner_tape, result, self.backend)
        return BoundCallable(inner_fn, space, tuple(unique_captured))

    def call_inline(self, *args) -> Tuple[Tracer]:
        """Evaluate explicitly through the NumPy tracing interpreter.

        Expands the computation for symbolic higher-order evaluation; ordinary
        traced calls leave composition to the enclosing backend.

        Args:
            *args: Concrete or symbolic values in input declaration order.

        Returns:
            The bare output for a single-slot construction, otherwise the
            ordered output collection, including absent slots.

        Examples:
            >>> from coker import Scalar, function
            >>> increment = function([Scalar("x")], lambda x: x + 1)
            >>> float(increment.call_inline(2.0))
            3.0
        """
        backend = get_backend_by_name("numpy", set_current=False)
        output = backend.evaluate(self, args)
        if self.is_single:
            return output[0]
        return output

    def __call__(self, *args: Any) -> Any:
        """Evaluate numerically or record a call on an enclosing tape.

        Args:
            *args: Values in input declaration order. Function-space inputs
                accept Coker functions or traceable Python callables.

        Returns:
            A bare value when constructed with one bare output, or a tuple
            for a sequence of outputs. Absent output slots remain ``None``.

        Raises:
            ValueError: The argument count differs from the declared arity.

        Examples:
            >>> from coker import Scalar, function
            >>> pair = function([Scalar("x")], lambda x: (x, x + 1))
            >>> tuple(float(value) for value in pair(2.0))
            (2.0, 3.0)
        """
        if len(args) != len(self.tape.input_indicies):
            raise ValueError(
                f"Expected {len(self.tape.input_indicies)} arguments but got "
                f"{len(args)}"
            )

        args = [
            (self._prepare_argument(arg, idx))
            for idx, arg in zip(self.tape.input_indicies, args)
        ]

        if any(isinstance(a, Tracer) for a in args):
            outer_tape = TraceContext.get_local_tape()
            if outer_tape is None:
                outer_tape = next(arg.tape for arg in args if isinstance(arg, Tracer))
            backend_name = (
                outer_tape.backend or self.backend or get_current_backend().name
            )
            backend = get_backend_by_name(backend_name, set_current=False)
            output = backend.append_native_call(self, args, outer_tape)
            if output is None:
                output = self._append_symbolic_call(outer_tape, args)
        else:
            # Concrete evaluation: lower once per backend/options combination.
            lowered = self.lower()
            output = lowered.restore_public_outputs(lowered.execute(args))

        if self.is_single:
            return output[0]
        return tuple(output)

    def lower(self, options: LoweringOptions | None = None) -> LoweredFunction:
        """Return a cached backend-specific executable handle.

        Args:
            options: Immutable lowering options, or ``None`` for defaults.
                An explicit backend overrides this function's backend.

        Returns:
            A lowering handle cached by backend identity and options.

        Raises:
            TypeError: ``options`` is not a ``LoweringOptions`` instance.
            RuntimeError: A native callable is lowered for a different backend.

        Examples:
            >>> from coker import Scalar, function
            >>> increment = function([Scalar("x")], lambda x: x + 1)
            >>> increment.lower() is increment.lower()
            True
        """

        options = LoweringOptions() if options is None else options
        if not isinstance(options, LoweringOptions):
            raise TypeError("options must be a LoweringOptions instance")
        backend = get_backend_by_name(
            options.backend or self.backend, set_current=False
        )
        if self._native_callable is not None and backend.name != self.backend:
            raise RuntimeError(
                "Cannot lower native callable for backend "
                f"{self.backend!r} with backend {backend.name!r}"
            )
        cache_key = (id(backend), options)
        try:
            return self._lowered_cache[cache_key]
        except KeyError:
            lowered = backend.lower(self, options)
            self._lowered_cache[cache_key] = lowered
            return lowered

    def _validate_bound(self, other):
        if len(self.output) != 1:
            raise ValueError("Cannot compare tensors: expected exactly one output")
        (dim,) = self.output_shape()
        if dim != get_dim_by_class(other):
            raise ValueError("Arguments have different shapes")

    def __le__(self, other: np.ndarray):
        """Construct an inclusive upper bound for one numeric output.

        Args:
            other: Bound with exactly the output's shape; no broadcasting.

        Returns:
            An inequality expression with an unbounded lower side.

        Raises:
            ValueError: There is not exactly one output or shapes differ.
        """
        self._validate_bound(other)
        ones = np.ones_like(other)
        return InequalityExpression(self, -np.inf * ones, other, is_equal=True)

    def __ge__(self, other):
        """Construct an inclusive lower bound for one numeric output.

        Args:
            other: Bound with exactly the output's shape; no broadcasting.

        Returns:
            An inequality expression with an unbounded upper side.

        Raises:
            ValueError: There is not exactly one output or shapes differ.
        """
        self._validate_bound(other)
        ones = np.ones_like(other)
        return InequalityExpression(self, other, ones * np.inf, is_equal=True)

    def __lt__(self, other):
        """Construct a strict upper-bound expression.

        Args:
            other: Bound with exactly the single numeric output's shape.

        Returns:
            An inequality expression with an unbounded lower side.

        Raises:
            ValueError: Output count or bound shape does not match.
        """
        self._validate_bound(other)
        ones = np.ones_like(other)
        return InequalityExpression(self, -np.inf * ones, other, is_equal=False)

    def __gt__(self, other):
        """Construct a strict lower-bound expression.

        Args:
            other: Bound with exactly the single numeric output's shape.

        Returns:
            An inequality expression with an unbounded upper side.

        Raises:
            ValueError: Output count or bound shape does not match.
        """
        self._validate_bound(other)
        ones = np.ones_like(other)
        return InequalityExpression(self, other, ones * np.inf, is_equal=False)


class BoundCallable(SymbolicCallable, FunctionSignatureValue):
    """A callable view that supplies captured tracer arguments to a target.

    Args:
        target: Function whose trailing present inputs are captured.
        public_space: Signature exposed to callers, excluding captures.
        bound_arguments: Captured tracer dependencies in target input order.

    Raises:
        ValueError: Public inputs and captures do not match target arity.

    Examples:
        >>> from coker import FunctionSpace, Scalar, function
        >>> from coker.algebra.function import BoundCallable
        >>> target = function([Scalar("x")], lambda x: x + 1)
        >>> view = BoundCallable(
        ...     target, FunctionSpace("f", [Scalar("x")], [Scalar("y")]), ()
        ... )
        >>> float(view(2.0))
        3.0
    """

    def __init__(
        self,
        target: Function,
        public_space: FunctionSpace,
        bound_arguments: tuple[Tracer, ...],
    ) -> None:
        present_input_count = sum(
            input_spec.space is not None and not isinstance(input_spec.space, Noop)
            for input_spec in target.signature.inputs
        )
        if present_input_count != (len(public_space.arguments) + len(bound_arguments)):
            raise ValueError(
                "Bound callable target signature does not match public and "
                "bound arguments"
            )
        self.target = target
        self.public_space = public_space
        self.bound_arguments = bound_arguments

    def _emit_symbol_value(self, tape: Tape) -> Tracer:
        """Emit the target symbol followed by one BIND node per capture."""
        captures = self.bound_arguments
        if not all(isinstance(capture, Tracer) for capture in captures):
            raise TypeError("BoundCallable arguments must be tracer dependencies")
        invalid_captures = [capture for capture in captures if capture.tape is not tape]
        if invalid_captures:
            raise DanglingTracerError(tracers=invalid_captures)
        value = self.target._emit_symbol_value(tape)
        position = len(self.public_space.arguments)
        for capture in captures:
            value = Tracer(tape, tape.append(OP.BIND, value, capture, position))
        return value

    def __call__(self, *arguments: Any) -> Any:
        """Call the target with public inputs followed by captured inputs.

        Args:
            *arguments: Values matching the public function space.

        Returns:
            The target's public result, with its original output structure.

        Raises:
            TypeError: The number of public arguments is incorrect.
        """
        if len(arguments) != len(self.public_space.arguments):
            raise TypeError(
                f"Expected {len(self.public_space.arguments)} arguments, "
                f"got {len(arguments)}"
            )
        return self.target(*arguments, *self.bound_arguments)

    def lower(self, options=None) -> LoweredFunction:
        """Lower the complete target, including its captured input slots.

        Args:
            options: Optional target lowering options.

        Returns:
            The target's cached lowering handle; captures are not executed
            or supplied by this handle automatically.
        """
        return self.target.lower(options)

    def input_shape(self):
        """Return the public input dimensions, excluding captured inputs.

        Returns:
            Tuple of dimensions in public argument order.
        """
        return tuple(self.public_space.input_dimensions())

    def output_shape(self):
        """Return the public output dimensions.

        Returns:
            Tuple of dimensions in public result order.
        """
        return tuple(self.public_space.output_dimensions())


class InequalityExpression:
    """Store a function and elementwise lower and upper bounds.

    Args:
        value: Function whose output is constrained.
        lower: Lower-bound array, possibly containing negative infinity.
        upper: Upper-bound array, possibly containing positive infinity.
        is_equal: Comparison construction flag. The expression currently
            stores only ``value``, ``lower`` and ``upper``, not strictness.

    Attributes:
        value: Constrained function.
        lower: Lower-bound values.
        upper: Upper-bound values.

    Examples:
        >>> import numpy as np
        >>> from coker import VectorSpace, function
        >>> identity = function([VectorSpace("x", 2)], lambda x: x)
        >>> constraint = identity > np.zeros(2)
        >>> constraint.lower.tolist()
        [0.0, 0.0]
    """

    def __init__(
        self,
        value: Function,
        lower: np.ndarray,
        upper: np.ndarray,
        is_equal: bool = False,
    ):
        self.value = value
        self.lower = lower
        self.upper = upper


def _normalise_result(result, tape: Tape):
    """Normalise an implementation's return value into Tracer(s) on tape."""

    def wrap(v):
        if isinstance(v, (list, tuple)):
            v = np.asarray(v, dtype=object)
        if isinstance(v, np.ndarray):
            v = strip_symbols_from_array(v)
        if isinstance(v, SymbolicVector):
            v = v.collapse()
        return v if isinstance(v, Tracer) else tape.insert_value(v)

    if isinstance(result, (list, tuple)):
        result = [wrap(r) for r in result]
    else:
        result = wrap(result)

    if isinstance(result, Tracer) and result.index == tape.NONE:
        return None
    return result


def function(
    arguments: List[Scalar | VectorSpace | FunctionSpace],
    implementation: Callable[[Element, ...], Element],
    backend: str = "numpy",
    name: Optional[str] = None,
) -> Function:
    """Compile a Python callable into a Coker :class:`Function`.

    Traces ``implementation`` by calling it with symbolic arguments derived
    from ``arguments``, records the resulting computation graph, and returns
    a :class:`Function` that evaluates it via the chosen backend.

    Args:
        arguments: Ordered list of argument spaces describing the domain of
            the function.  Each entry must be a :class:`Scalar`,
            :class:`VectorSpace`, or :class:`FunctionSpace`.
        implementation: A Python callable that defines the computation.  It
            will be called once during tracing with symbolic
            :class:`~coker.algebra.graph.Tracer` arguments.
        backend: Registered plugin name used for concrete evaluation.
            Defaults to ``"numpy"`` independently of the current backend.
        name: Optional human-readable name attached to the returned
            :class:`Function`.

    Returns:
        A compiled :class:`Function` that can be called with concrete numpy
        arrays or symbolic tracers.

    Raises:
        TypeError: ``implementation`` is not callable.

    Example:
        >>> import numpy as np
        >>> from coker import function, VectorSpace
        >>> A = np.eye(3)
        >>> f = function(
        ...     [VectorSpace("x", 3)],
        ...     lambda x: A @ x,
        ...     backend="numpy",
        ... )
        >>> f(np.array([1.0, 0.0, 0.0]))
        array([1., 0., 0.])
    """
    with TraceContext(backend=backend) as tape:
        args = [tape.input(v) for v in arguments]
        output = implementation(*args)
        result = _normalise_result(output, tape)
        return Function(tape, result, backend, name)
