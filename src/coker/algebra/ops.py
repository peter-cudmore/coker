import enum
import numpy as np
from typing import Dict, Callable

from coker.algebra.exceptions import InvalidShape, InvalidArgument
from coker.algebra.dimensions import Dimension


class OP(enum.Enum):
    VALUE = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    MATMUL = 5
    INT_PWR = 6
    PWR = 7
    EXP = 8
    SIN = 9
    COS = 10
    TAN = 11
    ARCSIN = 12
    ARCCOS = 13
    ARCTAN = 14
    SQRT = 15
    DOT = 16
    CROSS = 17
    TRANSPOSE = 18


compute_shape: Dict[OP, Callable[[Dimension, Dimension], Dimension]] = {}


def register_shape(*ops: OP):
    def inner(func):
        for op in ops:
            compute_shape[op] = func

        return func

    return inner

@register_shape(OP.VALUE)
def dimension_identity(dim: Dimension):
    return dim

@register_shape(OP.ADD, OP.SUB)
def same_dimension(d_1: Dimension, d_2: Dimension) -> Dimension:
    if d_1 != d_2:
        raise InvalidShape("Arguments are of different dimensions")

    return d_1


@register_shape(OP.MUL)
def shape_mul(d_1: Dimension, d_2: Dimension):
    if d_1.is_scalar() or d_1.dim == 1:
        return d_2

    if d_2.is_scalar() or d_2.dim == 1:
        return d_1

    raise InvalidArgument("Multiplication is not define between two non-scalars. "
                          "Consider using other operations")


@register_shape(OP.MATMUL)
def shape_matmul(d_1: Dimension, d_2: Dimension):
    if d_1.is_scalar() or d_2.is_scalar():
        raise InvalidArgument("Matrix multiplication is not defined for scalars")

    if d_1.is_vector():
        raise InvalidArgument("Cannot multiply vectors")

    if d_2.is_covector():
        raise InvalidArgument("Cannot right-multiply covectors")

    c = d_1.dim[-1]
    out_dims = []
    if d_1.is_matrix() or d_2.is_multilinear_map():
        out_dims.append(d_1.dim[:-1])

    r = d_2.dim[0]

    if d_2.is_matrix() or d_2.is_multilinear_map():
        out_dims.append(d_1.dim[1:])

    if c != r:
        raise InvalidArgument("Cannot multiply: product axis has different shape")

    if out_dims:
        return Dimension(tuple(*out_dims))
    else:
        return Dimension(None)


@register_shape(OP.SIN, OP.COS, OP.TAN, OP.ARCCOS, OP.ARCSIN, OP.EXP, OP.SQRT)
def scalar_shape(d: Dimension):
    if d.is_scalar():
        return d
    raise InvalidArgument("Can only operate on scalar dimensions.")


@register_shape(OP.CROSS)
def cross_shape(d_1: Dimension, d_2: Dimension):
    if d_1.dim != d_2.dim != (3,):
        raise InvalidArgument("Cross product is only defined for (3,) vectors")
    return Dimension((3,))


@register_shape(OP.DOT)
def dot_shape(d_1: Dimension, d_2: Dimension):
    if d_1.dim == d_2.dim and (d_1.is_vector() or d_1.is_covector()):
        return Dimension(None)

    raise InvalidArgument("Dot product only defined for vectors from the same space.")


@register_shape(OP.TRANSPOSE)
def transpose_shape(d: Dimension):
    if d.is_scalar():
        return d

    if d.is_vector():
        return Dimension((1, d.dim[0]))
    if d.is_covector():
        return Dimension((d.dim[1],))

    return Dimension(tuple(reversed(d.dim)))


numpy_atomics = {
    np.cross: OP.CROSS,
    np.dot: OP.DOT,
    np.matmul: OP.MATMUL,
    np.add: OP.ADD,
    np.multiply: OP.MUL,
    np.subtract: OP.SUB,
    np.sin: OP.SIN,
    np.cos: OP.COS,
    np.tan: OP.TAN,
    np.exp: OP.EXP,
    np.power: OP.PWR,
    np.transpose: OP.TRANSPOSE
}
