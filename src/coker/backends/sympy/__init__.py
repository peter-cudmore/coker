from __future__ import annotations

import sympy as sp
import numpy as np
from coker.algebra.function import Function, create_function_from_native
from coker.interfaces import SymbolicCallable
from coker.backends.backend import (
    ArrayLike,
    Backend,
    Evaluator,
    register_backend,
)
from coker.backends.lowered import (
    FunctionSignature,
    LoweredFunction,
    LoweringCapabilities,
)

from coker.algebra.ops import (
    OP,
    Noop,
    ConcatenateOP,
    NormOP,
    ReshapeOP,
    SelectOP,
    invoke_callable,
    normalize_evaluate_result,
)
from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    ResultBundleDimension,
    Scalar,
    VectorSpace,
)
from coker.algebra.graph import CallableReference, Tracer
from coker.backends.sympy.shape import reshape

MatrixType = (sp.Matrix, sp.ImmutableMatrix)


def sympy_mul(x, y):

    if isinstance(x, MatrixType) and isinstance(y, MatrixType):
        return x.multiply_elementwise(y)
    result = x * y
    return result


def to_matrix(x):
    if isinstance(x, sp.ImmutableMatrix):
        return x
    if isinstance(x, sp.Matrix):
        return sp.ImmutableMatrix(x)
    if isinstance(x, sp.Array):
        return sp.ImmutableMatrix(x.tomatrix())
    if isinstance(x, list):
        array = sp.Array(x)
        if len(array.shape) == 1:
            array.reshape(*array.shape, 1)
            return to_matrix(array)
        if len(array.shape) == 2:
            return to_matrix(array)
        raise ValueError("Cannot convert tensor to matrix")

    raise NotImplementedError(f"Unknown type: {x}")


def sympy_div(x, y):
    if isinstance(x, MatrixType) and isinstance(y, MatrixType):
        y_inv = sp.zeros(*y.shape)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y_inv[i, j] = 1 / y[i, j]
        return sp.ImmutableMatrix(x.multiply_elementwise(y_inv))
    return x / y


def sympy_matmul(x, y):
    if isinstance(x, sp.Matrix) and isinstance(y, sp.Matrix):
        return to_matrix(to_matrix(x) * to_matrix(y))

    idx = len(x.shape)
    assert x.shape[idx - 1] == y.shape[0]
    result = sp.tensorcontraction(sp.tensorproduct(x, y), (idx - 1, idx))
    assert result.shape == (*x.shape[0:-1], *y.shape[1:])

    if len(result.shape) == 2:
        return to_matrix(result)
    return result


def sympy_dot(x, y):
    if isinstance(x, MatrixType) and isinstance(y, MatrixType):
        return to_matrix(x).dot(to_matrix(y))
    try:
        return sp.tensorcontraction(sp.tensorproduct(x, y), (0, 1))
    except (AttributeError, TypeError, ValueError):
        pass
    return x.T @ y


def sympy_norm(x, ord):
    if not hasattr(x, "shape"):
        if ord in (None, 1, 2):
            return sp.Abs(x)
        raise NotImplementedError(f"Scalar norm ord={ord} is not supported")

    try:
        matrix = to_matrix(x)
    except (NotImplementedError, ValueError) as ex:
        raise NotImplementedError(
            "Norm is only implemented for scalars, vectors, and "
            f"matrices, got {x}"
        ) from ex

    rows, cols = matrix.shape
    is_vector = rows == 1 or cols == 1

    if is_vector:
        if ord in (None, 2):
            return matrix.norm()
        if ord == 1:
            return matrix.norm(1)
        raise NotImplementedError(f"Vector norm ord={ord} is not supported")

    if ord in (None, "fro"):
        return matrix.norm()
    if ord == 1:
        return matrix.norm(1)
    raise NotImplementedError(
        f"Matrix norm ord={ord} is not supported by the sympy backend"
    )


impls = {
    OP.ADD: lambda x, y: x + y,
    OP.SUB: lambda x, y: x - y,
    OP.MUL: sympy_mul,
    OP.DIV: sympy_div,
    OP.MATMUL: sympy_matmul,
    OP.SIN: sp.sin,
    OP.COS: sp.cos,
    OP.TAN: sp.tan,
    OP.EXP: sp.exp,
    OP.PWR: lambda x, y: x**y,
    OP.INT_PWR: lambda x, y: x**y,
    OP.ARCCOS: sp.acos,
    OP.ARCSIN: sp.asin,
    OP.DOT: sympy_dot,
    OP.CROSS: lambda x, y: x.cross(y),
    OP.TRANSPOSE: sp.transpose,
    OP.NEG: lambda x: -x,
    OP.SQRT: sp.sqrt,
    OP.ABS: lambda x: sp.Abs(x),
    OP.EQUAL: lambda x, y: x == y,
    OP.CASE: lambda cond, t, f: t if cond else f,
    OP.ARCTAN2: sp.atan2,
    OP.LOG: sp.log,
    OP.EVALUATE: lambda callable_value, *args: invoke_callable(
        callable_value, *args
    ),
}


def sympy_concat(*arrays, axis: int = 0):

    if axis == 0:
        return sp.Matrix.vstack(*arrays)
    if axis == 1:
        return sp.Matrix.hstack(*arrays)
    raise NotImplementedError


parameterised_impls = {
    ConcatenateOP: lambda op, *x: sympy_concat(*x, axis=op.axis),
    ReshapeOP: lambda op, x: reshape(x, dim=Dimension(op.newshape)),
    NormOP: lambda op, x: sympy_norm(x, ord=op.ord),
    SelectOP: lambda op, value: op.select(value),
}


class _SymbolicVectorFunction:
    """Represent a vector-valued undefined SymPy function."""

    def __init__(self, name: str, output: Dimension) -> None:
        self._name = name
        self._output = output

    def __call__(self, *arguments):
        values = [
            sp.Function(
                f"{self._name}_{'_'.join(str(value) for value in index)}"
            )(*arguments)
            for index in self._output.index_iterator(row_major=True)
        ]
        return sp.Array(values, shape=self._output.shape)


class _SympyFunctionTableValue:
    """A typed function-table target bound to evaluated capture values."""

    def __init__(self, target: Function, captures, backend) -> None:
        self._target = target
        self._captures = tuple(captures)
        self._backend = backend

    def __call__(self, *public_arguments):
        supplied_arguments = (*public_arguments, *self._captures)
        present_input_count = sum(
            input_spec.space is not None
            and not isinstance(input_spec.space, Noop)
            for input_spec in self._target.signature.inputs
        )
        if len(supplied_arguments) != present_input_count:
            raise TypeError(
                f"Expected {present_input_count} present inputs, got "
                f"{len(supplied_arguments)}"
            )

        arguments = iter(supplied_arguments)
        target_arguments = tuple(
            (
                None
                if input_spec.space is None
                else (
                    Noop()
                    if isinstance(input_spec.space, Noop)
                    else next(arguments)
                )
            )
            for input_spec in self._target.signature.inputs
        )
        outputs = _evaluate_sympy_tape(
            self._target.tape,
            target_arguments,
            self._target.output,
            self._backend,
            {},
        )
        values = tuple(
            output
            for output, output_spec in zip(
                outputs, self._target.signature.outputs
            )
            if output_spec.shape is not None
        )
        if not values:
            return ()
        return values[0] if len(values) == 1 else values


def _is_sympy_callable(value) -> bool:
    return isinstance(
        value,
        (
            SymbolicCallable,
            CallableReference,
            _SympyFunctionTableValue,
        ),
    )


def _evaluate_sympy_tape(tape, inputs, outputs, backend, workspace):
    """Interpret a tape while resolving typed function-table values."""
    from coker.backends.evaluator import _cast_outputs

    workspace[-1] = None
    for index, value in zip(tape.input_indicies, inputs):
        workspace[index] = (
            value
            if _is_sympy_callable(value)
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
            elif _is_sympy_callable(node):
                arguments.append(node)
            else:
                arguments.append(backend.to_backend_array(node))

        if op == OP.VALUE:
            (value,) = arguments
        elif op == OP.FUNCTION_VALUE:
            reference, *captures = arguments
            value = (
                _SympyFunctionTableValue(reference.target, captures, backend)
                if reference.is_function_reference
                else reference
            )
        else:
            value = backend.call(op, *arguments)

        if op == OP.EVALUATE and isinstance(
            arguments[0], (CallableReference, _SympyFunctionTableValue)
        ):
            value = normalize_evaluate_result(value, tape.dim[index])
        workspace[index] = (
            value
            if op == OP.FUNCTION_VALUE
            or isinstance(value, (Tracer, SymbolicCallable, CallableReference))
            or isinstance(tape.dim[index], ResultBundleDimension)
            else backend.reshape(value, tape.dim[index])
        )

    return _cast_outputs(outputs, tape, workspace, backend)


class SympyLoweredFunction(LoweredFunction):
    """SymPy evaluator-backed lowered execution handle."""

    def __init__(self, backend: SympyBackend, function: Function) -> None:
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
        )

    def execute(self, inputs) -> tuple:
        return tuple(
            _evaluate_sympy_tape(
                self._function.tape,
                inputs,
                self._function.output,
                self._backend,
                {},
            )
        )


class SympyBackend(Backend):
    def materialize_parameter(self, target, declaration, blocks):
        """Reconstruct a public parameter value from SymPy solver blocks."""
        flat_values = self._concatenate_parameter_blocks(blocks)
        if isinstance(target, FunctionSpace):
            return self._fit_function_parameter(
                declaration, target, flat_values
            )
        if isinstance(target, VectorSpace):
            return self._reshape_parameter_values(
                flat_values, target.dimension
            )
        if isinstance(target, Scalar):
            if flat_values.rows != 1:
                raise ValueError("scalar parameter must have one solver value")
            return flat_values[0]
        raise TypeError(
            "parameter target must be a scalar, vector, or function space"
        )

    def fit_function_parameter(self, declaration, target, values):
        """Materialize a fitted function from SymPy decision values."""
        return self._fit_function_parameter(
            declaration,
            target,
            self._concatenate_parameter_blocks((values,)),
        )

    def _fit_function_parameter(self, declaration, target, flat_values):
        from coker.parameters import BoundedVariable, UnboundedVariable
        from coker.parameters.function_parameters import FittedFunction

        parameters = []
        offset = 0
        for concrete in declaration.list_concrete_parameters():
            is_scalar = isinstance(
                concrete, (BoundedVariable, UnboundedVariable)
            )
            size = 1 if is_scalar else concrete.size
            block = flat_values[offset : offset + size, :]
            if block.rows != size:
                raise ValueError(
                    "solver decisions do not match function parameter "
                    "declarations"
                )
            parameters.append(
                block[0]
                if is_scalar
                else self._reshape_parameter_values(block, concrete.shape)
            )
            offset += size
        if offset != flat_values.rows:
            raise ValueError(
                "solver decisions do not match function parameter declarations"
            )

        target = declaration.validate_target(target)
        native = self.lower(declaration.build_function(target, self.name))
        return FittedFunction(
            declaration,
            target,
            lambda argument: native(argument, *parameters),
            tuple(parameters),
        )

    def _concatenate_parameter_blocks(self, blocks):
        if not blocks:
            raise ValueError("parameter blocks must not be empty")
        return sp.ImmutableMatrix(
            sp.Matrix.vstack(
                *(self._as_parameter_column(block) for block in blocks)
            )
        )

    def _as_parameter_column(self, block):
        value = self.to_backend_array(block)
        if isinstance(value, MatrixType):
            return sp.ImmutableMatrix(value.rows * value.cols, 1, list(value))
        if isinstance(
            value, (sp.ImmutableDenseNDimArray, sp.MutableDenseNDimArray)
        ):
            return sp.ImmutableMatrix(len(value), 1, list(value))
        return sp.ImmutableMatrix([value])

    @staticmethod
    def _reshape_parameter_values(values, shape):
        shape = (shape,) if isinstance(shape, int) else shape
        if len(shape) == 1:
            return sp.ImmutableMatrix(shape[0], 1, list(values))
        if len(shape) == 2:
            return sp.ImmutableMatrix(shape[0], shape[1], list(values))
        return sp.ImmutableDenseNDimArray(list(values), shape)

    def to_numpy_array(self, array):

        if isinstance(array, np.ndarray) and array.dtype in {
            float,
            np.float64,
        }:
            return array
        if isinstance(array, (int, float, complex)):
            return array

        try:
            if array.free_symbols or not array.is_constant():
                return array
        except AttributeError:
            pass

        try:
            value = sp.nsimplify(array, tolerance=1e-10)
            out = np.array(
                value.tolist() if hasattr(value, "tolist") else value,
                dtype=float,
            )
            if out.shape == ():
                return out.item()
            return out
        except (AttributeError, TypeError):
            pass

        raise ValueError(f"Cannot convert {array} to a numpy array")

    def to_backend_array(self, array):
        import scipy.sparse

        if scipy.sparse.issparse(array):
            array = array.toarray()
        if isinstance(array, np.ndarray):
            if len(array.shape) == 1:
                return to_matrix(
                    sp.Array(array.tolist(), shape=(array.shape[0], 1))
                )
            elif len(array.shape) == 2:
                return to_matrix(sp.Array(array.tolist(), shape=array.shape))
            return sp.Array(array.tolist(), shape=array.shape)
        if isinstance(array, list):
            try:
                return to_matrix(array)
            except ValueError:
                return sp.Array(array)
        if isinstance(array, np.float64):
            return sp.Float(float(array))
        if isinstance(
            array, (sp.ImmutableDenseNDimArray, sp.MutableDenseNDimArray)
        ):
            if len(array.shape) == 1:
                array = array.reshape(*array.shape, 1)
            if len(array.shape) == 2:
                return to_matrix(array)

        return array

    def reshape(self, array, shape):
        result = reshape(array, shape)
        try:
            if len(result.shape) == 2:
                return to_matrix(result)
        except (AttributeError, ValueError):
            pass
        return self.to_backend_array(result)

    def call(self, op, *args):
        if op == OP.EVALUATE:
            callable_value, *arguments = args
            if (
                isinstance(callable_value, CallableReference)
                and callable_value.is_function_reference
            ):
                return _SympyFunctionTableValue(
                    callable_value.target, (), self
                )(*arguments)
        if op in impls:
            result = impls[op](*args)
            return result

        if isinstance(op, tuple(parameterised_impls.keys())):
            kls = op.__class__

            result = parameterised_impls[kls](op, *args)
            return result

        raise NotImplementedError(f"{op} is not implemented")

    def lower(self, function, options=None) -> SympyLoweredFunction:
        return SympyLoweredFunction(self, function)

    def get_evaluator(self) -> Evaluator:
        raise NotImplementedError(
            "SymPy lowering does not use compiled tape evaluators"
        )

    def import_function(
        self,
        implementation,
        signature: FunctionSignature,
        *,
        name: str | None = None,
    ) -> Function:
        """Import a SymPy-compatible callable as a Coker function."""
        return create_function_from_native(
            implementation, signature, backend=self.name, name=name
        )

    def evaluate(self, function: Function, inputs: ArrayLike):

        results = _evaluate_sympy_tape(
            function.tape, inputs, function.output, self, {}
        )

        def eval(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x
            try:
                if x.is_constant():
                    return self.to_numpy_array(x)
            except (AttributeError, ValueError):
                pass

            try:
                return sp.nsimplify(x, tolerance=1e-10)
            except (TypeError, ValueError):
                pass
            if isinstance(x, sp.ImmutableDenseNDimArray):
                return x.applyfunc(eval)
            return x

        output = []
        for result, shape in zip(results, function.output_shape()):
            if shape is None:
                output.append(result)
                continue
            result = eval(result)

            if (
                result is not None
                and not shape.is_scalar()
                and result.shape != shape.shape
                and isinstance(result, MatrixType)
            ):
                result = sp.Array(result)
                result = result.reshape(*shape)
                output.append(result)

            else:
                output.append(result)

        return output

    def lower_to_symbolic(self, function: Function):
        """Return (args, output) as sympy symbolic expressions.

        This is a sympy-specific utility for inspecting or printing functions
        symbolically; it is separate from lower() which returns a callable.
        """

        tape = function.tape
        args = []
        workspace = {}
        for idx, name in zip(tape.input_indicies, tape.input_names):
            if idx < 0:
                args.append(None)
                continue
            dim = tape.dim[idx]
            if isinstance(dim, FunctionSpace):
                (output,) = dim.output_dimensions()
                if not isinstance(output, Dimension):
                    raise NotImplementedError(
                        "SymPy function parameters must return a finite "
                        "scalar or vector value"
                    )
                sym = (
                    sp.Function(name)
                    if output.is_scalar()
                    else _SymbolicVectorFunction(name, output)
                )
            elif dim.is_scalar():
                sym = sp.Symbol(name)
            else:
                shape = dim.shape
                if len(shape) == 1:
                    sym = sp.Array(
                        [sp.Symbol(f"{name}_{i}") for i in range(shape[0])]
                    )
                else:
                    sym = sp.Array(
                        [
                            [
                                sp.Symbol(f"{name}_{i}_{j}")
                                for j in range(shape[1])
                            ]
                            for i in range(shape[0])
                        ]
                    )
            args.append(sym)
            workspace[idx] = sym
        symbolic_backend = _SymbolicSympyBackend()
        outputs = _evaluate_sympy_tape(
            tape,
            args,
            function.output,
            symbolic_backend,
            workspace,
        )
        if function.is_single:
            return args, outputs[0]
        return args, outputs

    def build_optimisation_problem(*args):
        raise NotImplementedError("not supported on sympy backend")

    def evaluate_integrals(*args):

        raise NotImplementedError("not supported on sympy backend")


class _SymbolicSympyBackend(SympyBackend):
    """Lower external calls to undefined SymPy functions."""

    def call(self, op, *args):
        if (
            op == OP.EVALUATE
            and isinstance(args[0], CallableReference)
            and not args[0].is_function_reference
        ):
            return sp.Function(args[0].symbol_name)(*args[1:])
        return super().call(op, *args)


register_backend("sympy", SympyBackend)
