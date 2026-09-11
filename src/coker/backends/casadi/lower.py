from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np

import coker
from coker.algebra.kernel import (
    BoundCallable,
    CallableReference,
    Tape,
    Tracer,
)
from coker.algebra.ops import (
    OP,
    ConcatenateOP,
    Noop,
    NormOP,
    ReshapeOP,
    SelectOP,
    normalize_evaluate_result,
)
from coker.algebra.dimensions import FunctionSpace

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
    OP.LOG: ca.log,
    OP.EVALUATE: lambda callable_value, *args: casadi_eval(
        callable_value, *args
    ),
}


def casadi_eval(function, *args):
    if isinstance(function, BoundCallable):
        target, expanded_arguments = function.expand_call(*args)
        return casadi_eval(target, *expanded_arguments)
    # Native references and solver proxies are invoked directly.
    if not isinstance(function, coker.Function):
        return function(*args)
    if function.backend != "casadi":
        raise RuntimeError(
            "Cannot lower CasADi OP.EVALUATE node for "
            f"backend {function.backend!r}; node callable "
            f"{function!r} belongs to backend {function.backend!r}"
        )
    # coker.Function: evaluate symbolically via substitute, keeping CasADi
    # types (MX/DM) throughout rather than converting to Python scalars.
    workspace = {
        idx: arg
        for idx, arg in zip(function.tape.input_indicies, args)
        if idx >= 0
    }
    result = substitute(function.output, workspace)
    if function.is_single:
        return result[0]
    return result


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
    rows, cols = x.shape
    is_vector = rows == 1 or cols == 1

    if is_vector:
        if ord in (None, 2):
            return ca.norm_2(x)
        if ord == 1:
            return ca.norm_1(x)
        raise NotImplementedError(f"Vector norm ord={ord} is not supported")

    if ord is None:
        return ca.norm_fro(x)
    if ord == 1:
        return ca.mmax(ca.sum1(ca.fabs(x)))
    raise NotImplementedError(
        f"Matrix norm ord={ord} is not supported by the CasADi backend"
    )


def reshape(x, *shape):
    if len(shape) == 1:
        return ca.reshape(x, *shape, 1)
    else:
        return ca.reshape(x, *shape)


parameterised_impls = {
    ConcatenateOP: lambda op, *args: concat(*args, axis=op.axis),
    NormOP: lambda op, x: norm(x, ord=op.ord),
    ReshapeOP: lambda op, x: reshape(x, *op.newshape),
    SelectOP: lambda op, value: op.select(value),
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
        assert len(self.shape) == 3, "Higher order tensors not yet implemented"
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
    import scipy.sparse

    if scipy.sparse.issparse(value):
        return ca.DM(scipy.sparse.csc_matrix(value))
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
    try:
        if value == np.inf:
            return ca.inf
        if value == -np.inf:
            return -ca.inf
    except RuntimeError:
        pass

    return value


def extract_symbols(arg: ca.MX):
    if isinstance(arg, (ca.Function, coker.Function)):
        return set()
    if arg.is_symbolic():
        return {arg}
    return {
        arg.dep(index)
        for index in range(arg.n_dep())
        if arg.dep(index).is_symbolic()
    }


def substitute(
    output: Sequence[Tracer | None], workspace: dict[int, Any]
) -> list[Any | None]:
    def get_node(node: Any) -> Any:
        if isinstance(node, CallableReference):
            return node
        if node is None or node.index == Tape.NONE:
            return None
        if node is Noop():
            return node
        if node.index in workspace:
            return workspace[node.index]

        if node.is_constant():
            v = to_casadi(node.value())
        else:
            op, *args = node.value()
            args = [get_node(arg) for arg in args]
            if op == OP.FUNCTION_VALUE:
                (v,) = args
            elif op in impls:
                try:
                    v = impls[op](*args)
                except RuntimeError as e:
                    raise e
            else:
                v = call_parameterised_op(op, *args)
            if op == OP.EVALUATE and isinstance(args[0], CallableReference):
                v = normalize_evaluate_result(v, node.dim)
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


def lower(
    tape: Tape,
    output: Sequence[Tracer | None],
    workspace: dict[int, Any] | None = None,
) -> tuple[list[ca.MX], list[Any | None]]:
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
