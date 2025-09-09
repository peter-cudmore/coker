import casadi as ca
import numpy as np

import coker
from coker import OP, Tape, Tracer, FunctionSpace, Noop
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
    OP.ARCTAN2: ca.atan2,
    OP.EQUAL: ca.eq,
    OP.LESS_EQUAL: ca.le,
    OP.LESS_THAN: ca.lt,
    OP.CASE: lambda c, t, f: ca.if_else(c, t, f),
    OP.EVALUATE: lambda op, *args: casadi_eval(op, *args),
}


def casadi_eval(op, *args):
    result = op(*args)
    return to_casadi(result)


def concat(*args: ca.MX, axis=0):
    if not axis:
        return ca.vertcat(*args)

    if axis == 1:
        return ca.horzcat(*args)

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
    ConcatenateOP: lambda op, *args: concat(*args, axis=op.axis),
    NormOP: lambda op, x: norm(x, ord=op.ord),
    ReshapeOP: lambda op, x: reshape(x, *op.newshape),
}


def call_parameterised_op(op, *args):
    kls = op.__class__
    try:
        result = parameterised_impls[kls](op, *args)
    except KeyError as ex:
        raise KeyError(f"Operation {op} not implemented.") from ex

    return result


class CasadiTensor:
    def __init__(self, *shape):
        self.shape = shape
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return 0

    def __mul__(self, other):
        self.data = {k: d * other for k, d in self.data.items()}

    def __matmul__(self, other):
        assert (
            len(self.shape) == 3
        ), f"Higher order tensors not yet implemented"
        assert len(other.shape) == 2
        assert other.shape[0] == self.shape[-1]
        assert other.shape[1] == 1

        out = ca.MX(self.shape[0], self.shape[1])

        for (i, j, k), v in self.data.items():
            out[i, j] += v * other[k, 0]

        return out

    def reshape(self, shape):
        assert self.shape == shape
        return self


def to_casadi(value):
    if isinstance(value, np.ndarray):
        if len(value.shape) == 1:
            value = value.reshape(-1, 1)

        if len(value.shape) > 2:
            v = CasadiTensor(*value.shape)
        else:
            v = ca.DM.zeros(*value.shape)
        it = np.nditer(value, op_flags=["readonly"], flags=["multi_index"])
        for x in it:
            if x != 0:
                k = it.multi_index
                v[k] = x
        return v

    if value == np.inf:
        return ca.inf
    if value == -np.inf:
        return -ca.inf

    return value


def extract_symbols(arg: ca.MX):
    if isinstance(arg, (ca.Function, coker.Function)):
        return set()
    v = {arg.dep(i) for i in range(arg.n_dep()) if arg.dep(i).is_symbolic()}
    return v


def substitute(output: List[Tracer], workspace):
    def get_node(node: Tracer):
        if node is None:
            return None
        if node is Noop():
            return node
        if node.index in workspace:
            return workspace[node.index]

        if node.is_constant():
            v = to_casadi(node.value())
        else:
            op, *args = node.value()
            args = [get_node(a) for a in args]

            if op in impls:
                try:
                    v = impls[op](*args)
                except RuntimeError as e:
                    raise e
            else:
                v = call_parameterised_op(op, *args)
        try:
            if not node.dim.is_scalar():
                shape = (
                    node.shape
                    if not node.dim.is_vector()
                    else (*node.dim.shape, 1)
                )
                v = v.reshape(shape)
        except AttributeError:
            pass

        workspace[node.index] = v
        return v

    return [get_node(o) for o in output]


def lower(tape: Tape, output: List[Tracer], workspace=None):
    workspace = {} if not workspace else workspace
    inputs = dict()
    for i in tape.input_indicies:
        if i in workspace:
            s = extract_symbols(workspace[i])
            inputs.update({s_i.__hash__(): s_i for s_i in s})
            continue

        assert not isinstance(
            tape.dim[i], FunctionSpace
        ), "Cannot lower a partially evaluated function."

        v = ca.MX.sym(f"x_{i}", *tape.dim[i].shape)
        workspace[i] = v
        inputs[v.__hash__()] = v

    result = substitute(output, workspace)
    return list(inputs.values()), result
