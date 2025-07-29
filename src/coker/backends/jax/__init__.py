from typing import Type, Tuple, List

import numpy as np
import jax.numpy as jnp

from coker.algebra import Dimension, OP
from coker.algebra.kernel import Tracer
from coker.algebra.ops import ConcatenateOP, ReshapeOP, NormOP

from coker.backends.backend import Backend, ArrayLike


def to_array(value, shape):

    if isinstance(value, np.ndarray) and value.shape == shape:
        return jnp.array(value)

    raise plementedError


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
    OP.EVALUATE: lambda op, *args: op(*args),
}

parameterised_impls = {
    ConcatenateOP: lambda op, *x: jnp.concatenate(x, axis=op.axis),
    ReshapeOP: lambda op, x: jnp.reshape(x, shape=op.newshape),
    NormOP: lambda op, x: jnp.linalg.norm(x, ord=op.ord),
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


class JaxBackend(Backend):
    def __init__(self, *args, **kwargs):
        super(JaxBackend, self).__init__(*args, **kwargs)

    def native_types(self) -> Tuple[Type]:
        return (
            np.ndarray,
            np.int32,
            np.int64,
            np.float64,
            np.float32,
            float,
            complex,
            int,
        )

    def to_numpy_array(self, array) -> ArrayLike:

        return np.array(array)

    def to_backend_array(self, array):
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
        raise NotImplementedError(
            f"Don't know how to resize {arg.__class__.__name__}"
        )

    def call(self, op, *args) -> ArrayLike:

        try:
            result = impls[op](*args)
            return result
        except KeyError:
            pass

        if isinstance(op, tuple(parameterised_impls.keys())):
            return call_parameterised_op(op, *args)
        raise NotImplementedError(f"{op} is not implemented")

    def build_optimisation_problem(
        self,
        cost: Tracer,
        constraints: List[Tracer],
        arguments: List[Tracer],
        outputs: List[Tracer],
        **kwargs,
    ):
        raise NotImplementedError
