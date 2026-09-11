"""NumPy operation implementations and compiled-plan specialization."""

from typing import Any

import numpy as np

from coker.algebra import OP
from coker.algebra.graph import Tracer
from coker.algebra.ops import (
    ConcatenateOP,
    NormOP,
    ReshapeOP,
    SelectOP,
    invoke_callable,
)
from coker.backends.evaluator import GenericEvaluator


def div(num, den):
    if isinstance(den, Tracer):
        return num / den
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(num, den)
    try:
        if np.isscalar(den) and den == 0:
            if np.isscalar(num):
                return float("nan")
            return np.full_like(result, np.nan, dtype=float)
        zero_mask = den == 0
    except ValueError:
        return result
    if np.isscalar(zero_mask):
        return result
    if np.any(zero_mask):
        result = np.asarray(result, dtype=float)
        result[zero_mask] = np.nan
    return result


impls = {
    OP.ADD: np.add,
    OP.SUB: np.subtract,
    OP.MUL: np.multiply,
    OP.DIV: div,
    OP.MATMUL: np.matmul,
    OP.SIN: np.sin,
    OP.COS: np.cos,
    OP.TAN: np.tan,
    OP.EXP: np.exp,
    OP.PWR: np.power,
    OP.INT_PWR: np.power,
    OP.ARCCOS: np.arccos,
    OP.ARCSIN: np.arcsin,
    OP.DOT: np.dot,
    OP.CROSS: np.cross,
    OP.TRANSPOSE: np.transpose,
    OP.NEG: np.negative,
    OP.SQRT: np.sqrt,
    OP.ABS: np.abs,
    OP.ARCTAN2: np.arctan2,
    OP.EQUAL: np.equal,
    OP.LESS_EQUAL: np.less_equal,
    OP.LESS_THAN: np.less,
    OP.CASE: lambda cond, t, f: t if cond else f,
    OP.LOG: np.log,
    OP.EVALUATE: lambda callable_value, *args: invoke_callable(
        callable_value, *args
    ),
}

parameterised_impls = {
    ConcatenateOP: lambda op, *values: np.concatenate(values, axis=op.axis),
    ReshapeOP: lambda op, x: np.reshape(x, shape=op.newshape),
    NormOP: lambda op, x: np.linalg.norm(x, ord=op.ord),
    SelectOP: lambda op, value: op.select(value),
}


def call_parameterised_op(op, *args):
    return parameterised_impls[type(op)](op, *args)


class NumpyEvaluator(GenericEvaluator):
    """Compile plans using NumPy's native operation implementations."""

    def _resolve_operation(self, op):
        if op in impls:
            return impls[op]
        if isinstance(op, tuple(parameterised_impls.keys())):
            operation_type = type(op)
            return lambda *args: parameterised_impls[operation_type](op, *args)
        raise NotImplementedError(f"{op} is not implemented")

    def _resolve_post(self, dim):
        if not dim.is_scalar():
            return lambda value: value
        reshape = self.backend.reshape

        def scalar_post(value: Any):
            if isinstance(value, Tracer):
                return value
            return reshape(value, dim)

        return scalar_post
