import dataclasses
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
    NEG = 19

    def compute_shape(self, *dims: Dimension) -> Dimension:
        return compute_shape[self](*dims)


class Operator:
    def pre_process(self, *args):
        return args

    def compute_shape(self, *dims: Dimension) -> Dimension:
        raise NotImplementedError


class ConcatenateOP(Operator):
    __slots__ = ('axis', )

    def __init__(self, axis=0):
        self.axis = axis

    def pre_process(self, *args):

        assert len(args) == 1
        return args[0]

    def compute_shape(self, *dims: Dimension) -> Dimension:

        out_dims = list(dims[0].dim)
        for d in dims[1:]:
            assert all(
                d.dim[i] == out_dims[i] for i in range(len(out_dims))
                if i != self.axis
            )
            out_dims[self.axis] += d.dim[self.axis]

        return Dimension(tuple(out_dims))


class ReshapeOP(Operator):
    def __init__(self, newshape):
        self.newshape = newshape

    def compute_shape(self, dim: Dimension) -> Dimension:
        return Dimension(self.newshape)


class NormOP(Operator):
    def __init__(self, ord=2):
        self.ord = 2

    def compute_shape(self, *dims: Dimension) -> Dimension:
        return Dimension(None)


compute_shape: Dict[OP, Callable[[Dimension, Dimension], Dimension]] = {}


def register_shape(*ops: OP):
    def inner(func):
        for op in ops:
            compute_shape[op] = func

        return func

    return inner


@register_shape(OP.VALUE, OP.NEG)
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


@register_shape(OP.DIV)
def scalar_binary(d1: Dimension, d2: Dimension):
    assert d2.is_scalar()
    return d1


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
    np.transpose: OP.TRANSPOSE,
    np.divide: OP.DIV,
    np.arccos: OP.ARCCOS,
    np.arcsin: OP.ARCSIN,
    np.sqrt: OP.SQRT,
    np.negative: OP.NEG
}




numpy_composites = {
    np.concatenate: ConcatenateOP,
    np.reshape: ReshapeOP,
    np.linalg.norm: NormOP
}

