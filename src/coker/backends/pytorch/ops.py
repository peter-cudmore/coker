"""PyTorch implementations of Coker algebra operations."""

import numpy as np
import torch

from coker.algebra import OP
from coker.algebra.ops import (
    ConcatenateOP,
    EvaluateOP,
    NormOP,
    ReshapeOP,
    SelectOP,
    invoke_callable,
)


scalar_types = (float, complex, int, bool, np.number)


def div(num, den):
    """Divide while retaining Coker's explicit all-zero 0/0 result."""
    if isinstance(num, torch.Tensor) or isinstance(den, torch.Tensor):
        num_tensor = (
            num if isinstance(num, torch.Tensor) else torch.as_tensor(num)
        )
        den_tensor = (
            den
            if isinstance(den, torch.Tensor)
            else torch.as_tensor(den, device=num_tensor.device)
        )
        if bool(torch.all(den_tensor == 0)) and bool(
            torch.all(num_tensor == 0)
        ):
            return num
        return torch.divide(num, den)
    if num == 0 and den == 0:
        return num
    return torch.divide(torch.as_tensor(num), torch.as_tensor(den))


def _promoted_linear(fn, *args):
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    if len(tensors) > 1:
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        args = tuple(
            arg.to(dtype=dtype) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        )
    return fn(*args)


def _case(condition, true_value, false_value):
    if isinstance(condition, torch.Tensor):
        return torch.where(condition, true_value, false_value)
    return true_value if condition else false_value


def _transpose(value):
    if value.ndim < 2:
        return value
    return value.permute(tuple(reversed(range(value.ndim))))


def _norm(op, value):
    if value.ndim == 1:
        return torch.linalg.vector_norm(value, ord=op.ord)
    if value.ndim == 2:
        return torch.linalg.matrix_norm(value, ord=op.ord)
    return torch.linalg.norm(value, ord=op.ord)


impls = {
    OP.ADD: torch.add,
    OP.SUB: torch.subtract,
    OP.MUL: torch.multiply,
    OP.DIV: div,
    OP.MATMUL: lambda *args: _promoted_linear(torch.matmul, *args),
    OP.INT_PWR: torch.pow,
    OP.PWR: torch.pow,
    OP.EXP: torch.exp,
    OP.SIN: torch.sin,
    OP.COS: torch.cos,
    OP.TAN: torch.tan,
    OP.ARCSIN: torch.arcsin,
    OP.ARCCOS: torch.arccos,
    OP.ARCTAN: torch.arctan,
    OP.DOT: lambda *args: _promoted_linear(torch.dot, *args),
    OP.CROSS: lambda *args: _promoted_linear(torch.linalg.cross, *args),
    OP.NEG: torch.negative,
    OP.SQRT: torch.sqrt,
    OP.ABS: torch.abs,
    OP.CASE: _case,
    OP.EQUAL: torch.eq,
    OP.TRANSPOSE: _transpose,
    OP.ARCTAN2: torch.arctan2,
    OP.LESS_THAN: torch.less,
    OP.LESS_EQUAL: torch.less_equal,
    OP.LOG: torch.log,
}

parameterised_impls = {
    ConcatenateOP: lambda op, *x: torch.cat(x, dim=op.axis),
    ReshapeOP: lambda op, x: torch.reshape(x, shape=op.newshape),
    NormOP: _norm,
    SelectOP: lambda op, value: op.select(value),
    EvaluateOP: lambda op, callable_value, *args: invoke_callable(
        callable_value, *args
    ),
}


def call_parameterised_op(op, *args):
    """Evaluate a Coker operation carrying configuration data."""
    try:
        return parameterised_impls[op.__class__](op, *args)
    except KeyError as ex:
        raise NotImplementedError(f"{op} is not implemented") from ex
