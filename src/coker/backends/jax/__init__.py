from __future__ import annotations

from collections.abc import Sequence
from typing import Any, List

import numpy as np
import jax.numpy as jnp

from coker.algebra.function import Function, create_function_from_native
from coker.algebra import Dimension, OP
from coker.algebra.dimensions import ResultBundleDimension
from coker.algebra.graph import CallableReference, Tracer
from coker.algebra.ops import (
    ConcatenateOP,
    Noop,
    NormOP,
    ReshapeOP,
    SelectOP,
    invoke_callable,
    normalize_evaluate_result,
)
from coker.backends.evaluator import _cast_outputs
from coker.interfaces import SymbolicCallable

from coker.backends.backend import (
    ArrayLike,
    Backend,
    Evaluator,
    register_backend,
    split_function_parameter_values,
)

from coker.backends.lowered import (
    FunctionSignature,
    LoweredFunction,
    LoweringCapabilities,
)


scalar_types = (
    jnp.float32,
    jnp.float64,
    np.float64,
    np.float32,
    np.int32,
    np.int64,
    jnp.int32,
    jnp.int64,
    float,
    complex,
    int,
    bool,
    jnp.bool_,
)


def div(num, den):
    if (num == 0).all() and (den == 0):
        return num
    else:
        return jnp.divide(num, den)


class _JaxFunctionTableValue:
    """A typed function-table target bound to evaluated capture values."""

    def __init__(self, target: Function, captures, backend) -> None:
        self._target = target
        self._captures = tuple(captures)
        self._backend = backend

    def __call__(self, *public_arguments):
        supplied_arguments = (*public_arguments, *self._captures)
        input_spaces = tuple(
            input_spec.space for input_spec in self._target.signature.inputs
        )
        present_input_count = sum(
            input_space is not None and not isinstance(input_space, Noop)
            for input_space in input_spaces
        )
        if len(supplied_arguments) != present_input_count:
            raise TypeError(
                f"Expected {present_input_count} present inputs, got "
                f"{len(supplied_arguments)}"
            )

        arguments = iter(supplied_arguments)
        target_inputs = tuple(
            (
                None
                if input_space is None
                else (
                    Noop()
                    if isinstance(input_space, Noop)
                    else next(arguments)
                )
            )
            for input_space in input_spaces
        )
        workspace = {}
        _evaluate_jax_tape(
            self._target.tape, target_inputs, self._backend, workspace
        )
        values = tuple(
            workspace[output.index]
            for output, output_spec in zip(
                self._target.output, self._target.signature.outputs
            )
            if output_spec.shape is not None
        )
        if not values:
            return ()
        return values[0] if len(values) == 1 else values


def _is_jax_callable(value) -> bool:
    return isinstance(
        value,
        (SymbolicCallable, CallableReference, _JaxFunctionTableValue),
    )


def _evaluate_jax_tape(tape, inputs, backend, workspace) -> None:
    """Evaluate a tape while resolving typed function-table entries."""
    workspace[-1] = None
    for index, value in zip(tape.input_indicies, inputs):
        workspace[index] = (
            value
            if value is None
            or isinstance(value, Noop)
            or _is_jax_callable(value)
            else backend.to_backend_array(value)
        )

    for index in range(len(tape.nodes)):
        if index in workspace:
            continue

        op, *nodes = tape.nodes[index]
        arguments = []
        for node in nodes:
            if isinstance(node, Tracer):
                arguments.append(
                    workspace[node.index] if node.tape is tape else node
                )
            elif _is_jax_callable(node):
                arguments.append(node)
            else:
                arguments.append(backend.to_backend_array(node))

        if op == OP.VALUE:
            (value,) = arguments
        elif op == OP.FUNCTION_VALUE:
            reference, *captures = arguments
            value = (
                _JaxFunctionTableValue(reference.target, captures, backend)
                if reference.is_function_reference
                else reference
            )
        else:
            value = backend.call(op, *arguments)

        if op == OP.EVALUATE and isinstance(arguments[0], CallableReference):
            value = normalize_evaluate_result(value, tape.dim[index])
        workspace[index] = (
            value
            if op == OP.FUNCTION_VALUE
            or _is_jax_callable(value)
            or isinstance(tape.dim[index], ResultBundleDimension)
            else backend.reshape(value, tape.dim[index])
        )


impls = {
    OP.ADD: jnp.add,
    OP.SUB: jnp.subtract,
    OP.MUL: jnp.multiply,
    OP.DIV: div,
    OP.MATMUL: jnp.matmul,
    OP.SIN: jnp.sin,
    OP.COS: jnp.cos,
    OP.TAN: jnp.tan,
    OP.EXP: jnp.exp,
    OP.PWR: jnp.power,
    OP.INT_PWR: jnp.power,
    OP.ARCCOS: jnp.arccos,
    OP.ARCSIN: jnp.arcsin,
    OP.DOT: jnp.dot,
    OP.CROSS: jnp.cross,
    OP.TRANSPOSE: jnp.transpose,
    OP.NEG: jnp.negative,
    OP.SQRT: jnp.sqrt,
    OP.ABS: jnp.abs,
    OP.ARCTAN2: jnp.arctan2,
    OP.LESS_EQUAL: jnp.less_equal,
    OP.LESS_THAN: jnp.less,
    OP.EQUAL: jnp.equal,
    OP.CASE: lambda c, t, f: t if c else f,
    OP.LOG: jnp.log,
    OP.EVALUATE: lambda callable_value, *args: invoke_callable(
        callable_value, *args
    ),
}

parameterised_impls = {
    ConcatenateOP: lambda op, *x: jnp.concatenate(x, axis=op.axis),
    ReshapeOP: lambda op, x: jnp.reshape(x, shape=op.newshape),
    NormOP: lambda op, x: jnp.linalg.norm(x, ord=op.ord),
    SelectOP: lambda op, value: op.select(value),
}


def call_parameterised_op(op, *args):
    kls = op.__class__
    result = parameterised_impls[kls](op, *args)

    return result


def proj(i, n):
    p = np.zeros((n, n))
    p[i, i] = 1
    return p


def basis(i, n):
    p = np.zeros((n,))
    p[i] = 1
    return p


class JaxLoweredFunction(LoweredFunction):
    def __init__(self, backend: JaxBackend, function) -> None:

        self._backend = backend
        self._function = function

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def signature(self):
        return self._function.signature

    @property
    def capabilities(self) -> LoweringCapabilities:
        return LoweringCapabilities(
            eager_execution=True,
            symbolic_execution=True,
            autograd=True,
        )

    def execute(self, inputs: Sequence[Any]) -> tuple[Any | None, ...]:
        return tuple(self._backend.evaluate(self._function, inputs))


class JaxBackend(Backend):
    def fit_function_parameter(self, declaration, target, values):
        from coker.parameters.function_parameters import FittedFunction

        flat_values = np.asarray(
            self.to_numpy_array(values), dtype=float
        ).reshape(-1)
        parameters = split_function_parameter_values(declaration, flat_values)
        return FittedFunction(
            declaration,
            declaration.validate_target(target),
            lambda argument: declaration.evaluate(parameters, argument),
            parameters,
        )

    def __init__(self, *args, **kwargs):
        super(JaxBackend, self).__init__(*args, **kwargs)

    def to_numpy_array(self, array) -> ArrayLike:
        if array is None:
            return array
        numpy_array = np.array(array)
        if numpy_array.shape == ():
            return numpy_array.item()
        return numpy_array

    def to_backend_array(self, array):
        import scipy.sparse

        if scipy.sparse.issparse(array):
            array = array.toarray()
        return jnp.array(array)

    def reshape(self, arg, dim: Dimension):
        if dim.is_scalar():
            if isinstance(arg, scalar_types) or arg.ndim == 0:
                return arg
            else:
                try:
                    (inner,) = arg
                except ValueError as ex:
                    raise TypeError(f"Expecting a scalar, got {arg}") from ex

                return self.reshape(inner, dim)
        elif isinstance(arg, jnp.ndarray):
            return jnp.reshape(arg, dim.dim)
        elif isinstance(arg, np.ndarray):
            return np.reshape(arg, dim.dim)
        elif arg is None:
            return arg
        raise NotImplementedError(
            f"Don't know how to resize {arg.__class__.__name__}"
        )

    def evaluate(
        self, function: Function, inputs: Sequence[Any]
    ) -> list[Any | None]:
        workspace = {}
        _evaluate_jax_tape(function.tape, inputs, self, workspace)
        return _cast_outputs(function.output, function.tape, workspace, self)

    def call(self, op, *args) -> ArrayLike:
        if op == OP.EVALUATE:
            callable_value, *arguments = args
            if (
                isinstance(callable_value, CallableReference)
                and callable_value.is_function_reference
            ):
                return _JaxFunctionTableValue(callable_value.target, (), self)(
                    *arguments
                )

        try:
            result = impls[op](*args)
            return result
        except KeyError:
            pass

        if isinstance(op, tuple(parameterised_impls.keys())):
            return call_parameterised_op(op, *args)
        raise NotImplementedError(f"{op} is not implemented")

    def lower(self, function, options=None) -> JaxLoweredFunction:
        return JaxLoweredFunction(self, function)

    def get_evaluator(self) -> Evaluator:
        raise NotImplementedError(
            "JAX lowering does not use compiled tape evaluators"
        )

    def import_function(
        self,
        implementation,
        signature: FunctionSignature,
        *,
        name: str | None = None,
    ) -> Function:
        """Import a JAX-compatible callable as a Coker function."""
        return create_function_from_native(
            implementation, signature, backend=self.name, name=name
        )

    def build_optimisation_problem(
        self,
        cost: Tracer,
        constraints: List[Tracer],
        arguments: List[Tracer],
        outputs: List[Tracer],
        **kwargs,
    ):
        raise NotImplementedError


register_backend("jax", JaxBackend)
