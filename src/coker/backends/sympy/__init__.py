import sympy as sp
import numpy as np
from coker import Function
from coker.backends.backend import Backend, ArrayLike
from coker.algebra.ops import OP, ConcatenateOP, ReshapeOP, NormOP
from coker.algebra.dimensions import Dimension
from coker.backends.numpy.core import reshape


def sympy_mul(x, y):

    if isinstance(x, sp.Matrix) and isinstance(y, sp.Matrix):
        return x.multiply_elementwise(y)

    try:
        return x * y
    except ValueError as ex:
        return x @ y


def sympy_div(x, y):
    if isinstance(x, sp.Matrix) and isinstance(y, sp.Matrix):
        y_inv = sp.zeros(*y.shape)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y_inv[i, j] = 1 / y[i, j]
        return x.multiply_elementwise(y_inv)
    return x / y


def sympy_matmul(x, y):
    if isinstance(x, sp.Matrix) and isinstance(y, sp.Matrix):
        return x @ y

    idx = len(x.shape)
    assert x.shape[idx - 1] == y.shape[0]
    result = sp.tensorcontraction(sp.tensorproduct(x, y), (idx - 1, idx))
    assert result.shape == (*x.shape[0:-1], *y.shape[1:])
    if len(result.shape) == 2:
        return result.tomatrix()
    return result


def sympy_dot(x, y):
    if isinstance(x, sp.Matrix) and isinstance(y, sp.Matrix):
        return x.dot(y)
    try:
        return sp.tensorcontraction(sp.tensorproduct(x, y), (0, 1))
    except:
        pass
    return x.T @ y


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
    OP.EVALUATE: lambda op, *args: op(*args),
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
    #    NormOP: lambda op, x:
}


class SympyBackend(Backend):

    def to_numpy_array(self, array):
        if isinstance(array, (sp.Symbol, sp.Array, sp.Matrix, sp.Number)):
            return np.array(array, dtype=np.float64)

        return array

    def to_backend_array(self, array):
        if isinstance(array, np.ndarray):
            if len(array.shape) == 1:
                return sp.Array(
                    array.tolist(), shape=(array.shape[0], 1)
                ).tomatrix()
            elif len(array.shape) == 2:
                return sp.Array(array.tolist(), shape=array.shape).tomatrix()
            return sp.Array(array.tolist(), shape=array.shape)
        if isinstance(array, list):
            result = sp.Array(array)
            if len(result.shape) == 1:
                return result.reshape(len(result), 1).tomatrix()
            if len(result.shape) == 2:
                return result.tomatrix()
            return result
        if isinstance(array, np.float64):
            return sp.Float(float(array))
        return array

    def native_types(self):
        return [sp.Array, sp.Symbol, sp.Float, sp.Integer, sp.Rational]

    def reshape(self, array, shape):
        result = reshape(array, shape)
        return self.to_backend_array(result)

    def call(self, op, *args):
        if op in impls:
            return impls[op](*args)

        if isinstance(op, tuple(parameterised_impls.keys())):
            kls = op.__class__

            result = parameterised_impls[kls](op, *args)

            return result

            return call_parameterised_op(op, *args)

        raise NotImplementedError(f"{op} is not implemented")

    def evaluate(self, function: Function, inputs: ArrayLike):
        results = super().evaluate(function, inputs)

        def eval(x):

            if isinstance(x, (sp.Array, sp.Matrix, sp.Number)):
                out = np.array(x)
                return out
            try:
                if x.is_constant():
                    return float(x)
            except AttributeError:
                pass

            return x

        return [eval(result) for result in results]

    def build_optimisation_problem(*args):
        raise NotImplementedError("not supported on sympy backend")

    def evaluate_integrals(*args):

        raise NotImplementedError("not supported on sympy backend")
