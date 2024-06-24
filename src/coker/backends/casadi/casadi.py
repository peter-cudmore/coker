import casadi as ca
import numpy as np

from coker import OP, ExprOp, Tape, Tracer
from coker.algebra.ops import ConcatenateOP, ReshapeOP, NormOP
from typing import List

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
    OP.ABS: ca.fabs,
}


def concat(a: ca.MX, b: ca.MX, axis=0):
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


def reshape(x, *shape):
    if len(shape) == 1:
        return ca.reshape(x, *shape, 1)
    else:
        return ca.reshape(x, *shape)


parameterised_impls = {
    ConcatenateOP: lambda op, x, y: concat(x, y, op.axis),
    NormOP: lambda op, x: norm(x, ord=op.ord),
    ReshapeOP: lambda op, x: reshape(x, *op.newshape),
}


def call_parameterised_op(op, *args):
    kls = op.__class__
    result = parameterised_impls[kls](op, *args)

    return result


def to_casadi(value):
    if isinstance(value, np.ndarray):
        return ca.DM(value)

    return value


def lower(tape: Tape, output: List[Tracer]):
    workspace = {}

    for i in tape.input_indicies:
        workspace[i] = ca.MX.sym(f"x_i", *tape.dim[i].shape)

    def get_node(node: Tracer):
        if node.index in workspace:
            return workspace[node.index]

        if node.is_constant():
            v = to_casadi(node.value())
            if not node.dim.is_scalar():
                shape = node.shape if not node.dim.is_vector() else (*node.dim.shape, 1)
                v = ca.reshape(v, shape)
        else:
            op, *args = node.value()
            args = [get_node(a) for a in args]

            if op in impls:
                v = impls[op](*args)
            else:
                v = call_parameterised_op(op, *args)

        workspace[node.index] = v
        return v

    result = [get_node(o) for o in output]

    return ca.Function("f", [workspace[i] for i in tape.input_indicies], result)
