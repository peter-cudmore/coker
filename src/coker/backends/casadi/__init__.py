import casadi as ca
import numpy as np
from typing import Tuple, Type, Union

from coker import Tensor, OP, Dimension
from coker.algebra.ops import ConcatenateOP, ReshapeOP, NormOP
from coker.backends.backend import Backend, ArrayLike

scalar_types = (float, int)

impls = {
    OP.ADD: lambda x, y: x + y,
    OP.SUB: lambda x, y: x - y,
    OP.MUL: lambda x, y: x * y,
    OP.DIV: lambda x, y: x / y,
    OP.MATMUL: lambda x, y: x @ y,
    OP.SIN: ca.sin,
    OP.COS: ca.cos,
    OP.TAN: ca.tan,
    OP.EXP: ca.exp,
    OP.PWR: ca.power,
    OP.INT_PWR: ca.power,
    OP.ARCCOS: ca.arccos,
    OP.ARCSIN: ca.arcsin,
    OP.DOT: ca.dot,
    OP.CROSS: ca.cross,
    OP.TRANSPOSE: ca.transpose,
    OP.NEG: lambda x: -x,
    OP.SQRT: ca.sqrt,
    OP.ABS: ca.fabs
}

def concat(a:ca.MX, b:ca.MX, axis=0):
    if not axis:
        return ca.vertcat(a, b)

    if axis == 1:
        return ca.horzcat(a, b)

    # axis = 0 -> vstack
    # axis = 1 -> hstack
    # axis = None -> flatten +
    raise NotImplementedError


def norm(x, ord):
    if ord == 1:
        return ca.norm_1(x)
    if ord == 2:
        return ca.norm_2(x)

    raise NotImplementedError


parameterised_impls = {
    ConcatenateOP: lambda op, x, y: concat(x, y, op.axis),
    NormOP: lambda op, x: norm(x, ord=op.ord)
}


def call_parameterised_op(op, *args):
    kls = op.__class__
    result = parameterised_impls[kls](op, *args)

    return result


class CasadiBackend(Backend):
    def to_native(self, array: Union[ca.MX, ca.DM]) -> ArrayLike:
        if isinstance(array, ca.MX):

            return array.to_DM().toarray(simplify=True)

        elif isinstance(array, ca.DM):
            return array.toarray(simplify=True)

        raise NotImplementedError

    def from_native(self, array) -> ca.MX:
        if isinstance(array, scalar_types):
            return ca.DM(array)
        if array.shape == (1, 1):
            return ca.DM(array[0, 0])
        elif array.shape == (1, ):
            return ca.DM(array[0])
        elif len(array.shape) >= 2:
            result = ca.DM(*array.shape)
            with np.nditer(array, flags=["multi_index"], op_flags=['readonly']) as it:
                for v in it:
                    if v != 0:
                        key = tuple(it.multi_index)
                        result[key] = v

        elif len(array.shape) == 1 and array.shape[0] > 1:
            n, = array.shape
            result = ca.DM(n, 1)
            for i, v in enumerate(array):
                if v != 0:
                    result[i] = v

        else:
            raise NotImplementedError
        return result

    def call(self, op, *args) -> ArrayLike:
        try:
            result: ca.DM = impls[op](*args)
            assert result.is_regular(), f"{op}({args}) =  {result}"

            return result
        except KeyError:
            pass

        if isinstance(op, tuple(parameterised_impls.keys())):
            return call_parameterised_op(op, *args)

        if isinstance(op, ReshapeOP):

            arg, = args
            shape = op.newshape
            if len(shape) == 1:
                shape = (1, *shape)

            return ca.reshape(arg, shape)

        raise NotImplementedError(f"{op} is not implemented")

    def native_types(self) -> Tuple[Type]:
        pass

    def reshape(self, array: ArrayLike, dim: Dimension) -> ArrayLike:
        if dim.is_scalar():
            return array

        if dim.is_vector():
            shape = (*dim, 1)
        else:
            shape = tuple(dim)

        if isinstance(array, (ca.MX, ca.DM)):
            return ca.reshape(array, *shape)
        if isinstance(array, np.ndarray):
            return ca.reshape(array, *shape)
        raise NotImplementedError

    def evaluate(self, kernel, inputs: ArrayLike):
        from coker.backends.evaluator import evaluate_inner
        workspace = {}

        return evaluate_inner(kernel.tape, inputs, kernel.outputs, self, workspace)
