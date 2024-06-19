from typing import Type, Tuple, List
from functools import reduce
from operator import mul
import numpy as np
import jax.numpy as jnp

from coker.algebra import Tensor, Dimension, OP
from coker.algebra.kernel import Expression, Tracer, VectorSpace
from coker.algebra.ops import ConcatenateOP, ReshapeOP, NormOP

from coker.backends.backend import Backend, ArrayLike
from coker.backends.evaluator import evaluate_inner

def to_array(value, shape):

    if isinstance(value, np.ndarray) and value.shape == shape:
        return jnp.array(value)

    raise NotImplementedError


scalar_types = (
    jnp.float32, jnp.float64, np.float64, np.float32,
    np.int32, np.int64,
                jnp.int32, jnp.int64, float, complex, int)


def div(num, den):
    if (num == 0).all() and (den == 0):
        return num
    else:
        return jnp.divide(num, den)


impls = {
    OP.ADD: jnp.add,
    OP.SUB: jnp.subtract,
    OP.MUL:jnp.multiply,
    OP.DIV:div,
    OP.MATMUL:jnp.matmul,
    OP.SIN:jnp.sin,
    OP.COS:jnp.cos,
    OP.TAN:jnp.tan,
    OP.EXP:jnp.exp,
    OP.PWR:jnp.power,
    OP.INT_PWR:jnp.power,
    OP.ARCCOS:jnp.arccos,
    OP.ARCSIN:jnp.arcsin,
    OP.DOT:jnp.dot,
    OP.CROSS:jnp.cross,
    OP.TRANSPOSE:jnp.transpose,
    OP.NEG:jnp.negative,
    OP.SQRT:jnp.sqrt,
    OP.ABS:jnp.abs
}

parameterised_impls = {
    ConcatenateOP: lambda op, x, y:jnp.concatenate((x, y), axis=op.axis),
    ReshapeOP: lambda op, x:jnp.reshape(x, newshape=op.newshape),
    NormOP: lambda op, x:jnp.linalg.norm(x, ord=op.ord)
}


def call_parameterised_op(op, *args):
    kls = op.__class__
    result = parameterised_impls[kls](op, *args)

    return result


def proj(i, n):
    p = np.zeros((n, n))
    p[i, i] = 1
    return p

def basis(i,n):
    p = np.zeros((n, ))
    p[i] = 1
    return p

class JaxBackend(Backend):
    def __init__(self, *args, **kwargs):
        super(JaxBackend, self).__init__(*args, **kwargs)

    def native_types(self) -> Tuple[Type]:
        return np.ndarray, np.int32, np.int64, np.float64, np.float32, float, complex, int

    def to_native(self, array: Tensor) -> ArrayLike:
        return array

    def from_native(self, array: ArrayLike) -> Tensor:
        return array

    def reshape(self, arg, dim: Dimension):
        if dim.is_scalar():
            if isinstance(arg, scalar_types) or arg.ndim == 0:
                return arg
            else:
                try:
                    inner, = arg
                except ValueError as ex:
                    raise TypeError(f'Expecting a scalar, got {arg}') from ex
                return self.reshape(inner, dim)
        elif isinstance(arg, jnp.ndarray):
            return jnp.reshape(arg, dim.dim)
        elif isinstance(arg, np.ndarray):
            return np.reshape(arg, dim.dim)
        raise NotImplementedError(f"Don't know how to resize {arg.__class__.__name__}")

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
        cost: Tracer,  # cost
        constraints: List[Expression],
        arguments: List[Tracer],
        outputs: List[Tracer]
    ):

        n_constraints = len(constraints)
        c = np.zeros((n_constraints,))
        # halfspace = {x: h(x)_i >=0 for all i}
        for i, constraint in enumerate(constraints):
            c_i = -basis(i, n_constraints) * constraint.as_halfplane_bound()
            c += c_i

        # arguments = x
        # cost = f(x)
        # constraints = c(x) >= 0
        # outputs => y = g(x)
        #
        # solve x^* = argmin f(x) s.t. c(x) >=0
        #
        # return g(x^*)


        assert False




#
# OP v_1, v_2, v_3
#
# v is either
# a) a symbol
# b) a constant
# c) another node in the graph
#
# if OP is linear
# -

