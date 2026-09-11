from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from coker.algebra.dimensions import (
    Dimension,
    Element,
    FunctionSpace,
    ResultBundleDimension,
    Scalar,
    VectorSpace,
)
from coker.algebra.graph import (
    DanglingTracerError,
    Tape,
    TraceContext,
    Tracer,
    _find_closure_tracers,
    get_dim_by_class,
    strip_symbols_from_array,
)
from coker.algebra.ops import OP, Noop, SelectOP
from coker.algebra.tensor import SymbolicVector

from coker.backends.backend import get_backend_by_name
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
    LoweredFunction,
    LoweringOptions,
    OutputShape,
)


class SymbolicCallable(ABC):
    """Common call/lowering contract for symbolic callable graph values.

    Implementations accept concrete values or tracers and expose ``lower``
    for backend-specific execution.  Variational problems intentionally do
    not participate because their solver interface is different.
    """

    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError

    @abstractmethod
    def lower(self):
        raise NotImplementedError


class Function(SymbolicCallable):
    """A compiled Coker function.

    Created by :func:`function`.  Holds the traced computation graph and
    dispatches to a backend for evaluation.  Calling a ``Function`` with
    concrete numpy arrays returns the result; calling it inside a tracing
    context records the operations on the outer tape.

    Attributes:
        tape: The recorded computation graph.
        backend: Name of the backend used for concrete evaluation.
        output: List of output :class:`~coker.algebra.kernel.Tracer` nodes.
    """

    INLINE_SIZE = 10

    def __init__(
        self,
        tape: Tape,
        outputs: Tracer | None | Sequence[Tracer | None],
        backend: str = "coker",
        name: str | None = None,
        signature: "FunctionSignature | None" = None,
    ) -> None:
        self.name = name
        self.tape = tape
        self.backend = backend
        self._lowered_cache: dict[
            tuple[int, "LoweringOptions"], "LoweredFunction"
        ] = {}
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
        return self.tape.input_names.copy()

    def __repr__(self):
        name = self.name if self.name else "<unknown>"
        try:
            out_shape = f"{self.output_shape()}"
        except AttributeError:
            out_shape = f"{self.output}"
        return f"{name}:{self.input_shape()} -> {out_shape}"

    def to_space(self, name):
        return FunctionSpace(
            name,
            arguments=[
                dim.to_space("input_{i}") if dim else None
                for i, dim in enumerate(self.input_shape())
            ],
            output=[
                dim.to_space("output_{i}") if dim else None
                for i, dim in enumerate(self.input_shape())
            ],
        )

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
        """Return each input argument shape in declaration order."""
        special_inputs = {
            Tape.NONE: None,
            Tape.MAP_TO_NONE: Noop().cast_to_function_space(None),
        }

        return tuple(
            self.tape.dim[i] if i >= 0 else special_inputs[i]
            for i in self.tape.input_indicies
        )

    def output_shape(self) -> tuple[Dimension | FunctionSpace | None, ...]:
        """Return each output shape in declaration order."""
        return tuple(o.dim if o is not None else None for o in self.output)

    @staticmethod
    def _append_native_outputs(
        tape: Tape,
        native: Callable[..., Any],
        backend: str,
        input_spaces: Sequence[Scalar | VectorSpace | FunctionSpace],
        output_specs: Sequence["FunctionOutputSpec"],
        args: Sequence[Tracer],
    ) -> list[Tracer | None]:
        def result_output_dimension(
            shape: "OutputShape",
        ) -> Dimension | FunctionSpace | None:
            if shape is None or isinstance(shape, (Dimension, FunctionSpace)):
                return shape
            if isinstance(shape, Scalar):
                return Dimension(None)
            if isinstance(shape, VectorSpace):
                return Dimension(shape.dimension)
            raise TypeError(f"Unsupported native output shape {shape!r}")

        result_dimension = ResultBundleDimension(
            tuple(
                result_output_dimension(output_spec.shape)
                for output_spec in output_specs
            )
        )
        output_spaces = [
            (
                output_spec.shape.to_space(output_spec.name)
                if isinstance(output_spec.shape, Dimension)
                else output_spec.shape
            )
            for output_spec in output_specs
            if output_spec.shape is not None
        ]
        function_space = FunctionSpace(
            f"{backend}_native",
            arguments=list(input_spaces),
            output=output_spaces,
        )
        native_ref = tape._create_callable_reference(
            native,
            function_space,
            result_dimension,
        )
        bundle = Tracer(
            tape,
            tape.append(
                OP.EVALUATE,
                native_ref,
                *args,
            ),
        )
        return [
            (
                None
                if output_spec.shape is None
                else Tracer(
                    tape,
                    tape.append(SelectOP(output_index), bundle),
                )
            )
            for output_index, output_spec in enumerate(output_specs)
        ]

    def _call_native_in_trace(
        self, args: Sequence[Tracer], outer_tape: Tape
    ) -> Tracer | tuple[Tracer | None, ...]:
        if outer_tape.backend != self.backend:
            raise RuntimeError(
                "Cannot compose native callable for backend "
                f"{self.backend!r} into {outer_tape.backend!r} trace"
            )
        native = self._native_callable
        assert native is not None
        outputs = self._append_native_outputs(
            outer_tape,
            native,
            self.backend,
            [spec.space for spec in self.signature.inputs],
            self.signature.outputs,
            args,
        )
        return outputs[0] if self.is_single else tuple(outputs)

    def _prepare_argument(self, arg, index):
        if index == Tape.MAP_TO_NONE:
            return Noop()
        elif index == Tape.NONE:
            return None

        elif isinstance(self.tape.dim[index], FunctionSpace):
            if isinstance(arg, SymbolicCallable):
                return arg

            try:
                return function(
                    self.tape.dim[index].arguments, arg, self.backend
                )
            except DanglingTracerError as ex:
                return self._lift_closure(
                    arg, self.tape.dim[index], ex
                )  # arg is a Python callable

        return arg

    def _lift_closure(
        self, fn, space: FunctionSpace, ex: DanglingTracerError
    ) -> "BoundCallable":
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
        """Evaluate this function symbolically inside an active trace.

        Unlike ``__call__``, which compiles to the configured backend,
        this always routes through the numpy interpreter so the result
        is a :class:`~coker.algebra.kernel.Tracer` recorded on the
        enclosing tape. Use this when composing functions inside an
        ``implementation`` passed to :func:`function`.
        """

        backend = get_backend_by_name("numpy", set_current=False)
        output = backend.evaluate(self, args)
        if self.is_single:
            return output[0]
        return output

    def __call__(self, *args: Any) -> Any:
        assert len(args) == len(self.tape.input_indicies), (
            f"Expected {len(self.tape.input_indicies)} arguments but got "
            f"{len(args)}"
        )

        args = [
            (self._prepare_argument(arg, idx))
            for idx, arg in zip(self.tape.input_indicies, args)
        ]

        if any(isinstance(a, Tracer) for a in args):
            if self._native_callable is not None:
                outer_tape = TraceContext.get_local_tape()
                if outer_tape is None:
                    outer_tape = next(
                        a.tape for a in args if isinstance(a, Tracer)
                    )
                return self._call_native_in_trace(args, outer_tape)
            # Tracing context: interpret through numpy so ops are recorded on
            # the outer tape rather than evaluated numerically.
            backend = get_backend_by_name("numpy", set_current=False)
            output = backend.evaluate(self, args)
        else:
            # Concrete evaluation: lower once per backend/options combination.
            lowered = self.lower()
            outputs = lowered.execute(args)
            backend = get_backend_by_name(
                lowered.backend_name, set_current=False
            )
            output = backend.restore_public_outputs(self, outputs)

        if self.is_single:
            return output[0]
        return tuple(output)

    def lower(
        self, options: "LoweringOptions | None" = None
    ) -> "LoweredFunction":
        """Return a cached backend-specific executable lowering handle."""

        options = LoweringOptions() if options is None else options
        if not isinstance(options, LoweringOptions):
            raise TypeError("options must be a LoweringOptions instance")
        backend = get_backend_by_name(options.backend or self.backend)
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

    def __le__(self, other: np.ndarray):
        # self < other
        assert len(self.output_shape()) == 1, "Cannot compare tensors"
        dim = self.output_shape()[0]
        assert dim.shape == get_dim_by_class(
            other
        ), "Arguments have different shapes"
        ones = np.ones_like(other)
        return InequalityExpression(self, -np.inf * ones, other, is_equal=True)

    def __ge__(self, other):
        # self => other
        assert len(self.output_shape()) == 1, "Cannot compare tensors"
        (dim,) = self.output_shape()
        assert dim == get_dim_by_class(
            other
        ), "Arguments have different shapes"
        ones = np.ones_like(other)
        return InequalityExpression(self, other, ones * np.inf, is_equal=True)

    def __lt__(self, other):
        # self <= other
        assert len(self.output_shape()) == 1, "Cannot compare tensors"
        (dim,) = self.output_shape()
        assert dim == get_dim_by_class(
            other
        ), "Arguments have different shapes"
        ones = np.ones_like(other)
        return InequalityExpression(
            self, -np.inf * ones, other, is_equal=False
        )

    def __gt__(self, other):
        # self > other
        assert len(self.output_shape()) == 1, "Cannot compare tensors"
        dim = self.output_shape()[0]
        assert dim.shape == other.shape, "Arguments have different shapes"
        ones = np.ones_like(other)
        return InequalityExpression(self, other, ones * np.inf, is_equal=False)


class BoundCallable(SymbolicCallable):
    """A public callable view over a target with explicit bound arguments."""

    def __init__(
        self,
        target: Function,
        public_space: FunctionSpace,
        bound_arguments: tuple[Tracer, ...],
    ) -> None:
        if len(target.signature.inputs) != (
            len(public_space.arguments) + len(bound_arguments)
        ):
            raise ValueError(
                "Bound callable target signature does not match public and "
                "bound arguments"
            )
        self.target = target
        self.public_space = public_space
        self.bound_arguments = bound_arguments

    def expand_call(self, *arguments: Any) -> tuple[Function, tuple[Any, ...]]:
        if len(arguments) != len(self.public_space.arguments):
            raise TypeError(
                f"Expected {len(self.public_space.arguments)} arguments, "
                f"got {len(arguments)}"
            )
        return self.target, (*arguments, *self.bound_arguments)

    def __call__(self, *arguments: Any) -> Any:
        target, expanded_arguments = self.expand_call(*arguments)
        return target(*expanded_arguments)

    def lower(self, options=None) -> "LoweredFunction":
        return self.target.lower(options)


class InequalityExpression:
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
    if isinstance(result, np.ndarray):
        result = strip_symbols_from_array(result)
    if isinstance(result, SymbolicVector):
        result = result.collapse()

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
    backend: str = "coker",
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
            :class:`~coker.algebra.kernel.Tracer` arguments.
        backend: Name of the backend used for concrete evaluation.
            Built-in options are ``"numpy"`` (default for tracing),
            ``"casadi"``, ``"sympy"``, and ``"coker"``.
        name: Optional human-readable name attached to the returned
            :class:`Function`.

    Returns:
        A compiled :class:`Function` that can be called with concrete numpy
        arrays or symbolic tracers.

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


def create_function_from_native(
    native: Callable[..., Any],
    signature: "FunctionSignature",
    *,
    backend: str,
    name: str | None = None,
) -> Function:
    """Build a traceable Coker function around a backend-native callable."""

    if not isinstance(signature, FunctionSignature):
        raise TypeError("signature must be a FunctionSignature")

    input_spaces = [spec.space for spec in signature.inputs]
    with TraceContext(backend=backend) as tape:
        args = [tape.input(space) for space in input_spaces]
        outputs = Function._append_native_outputs(
            tape, native, backend, input_spaces, signature.outputs, args
        )

    result = Function(
        tape,
        outputs[0] if len(outputs) == 1 else outputs,
        backend=backend,
        name=name,
        signature=signature,
    )
    result._native_callable = native
    return result
