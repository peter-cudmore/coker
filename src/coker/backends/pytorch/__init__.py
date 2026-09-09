"""PyTorch numerical backend.

The backend intentionally implements only Coker's numerical expression
operations. ODE integration and optimisation remain unsupported.
"""

import numpy as np
import torch

from coker.algebra import Dimension, OP
from coker.algebra.kernel import Tracer
from coker.algebra.ops import ConcatenateOP, NormOP, ReshapeOP
from coker.backends.backend import ArrayLike, Backend


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


def case(condition, true_value, false_value):
    if isinstance(condition, torch.Tensor):
        return torch.where(condition, true_value, false_value)
    return true_value if condition else false_value


def _transpose(value):
    if value.ndim < 2:
        return value
    return value.permute(tuple(reversed(range(value.ndim))))


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
    OP.CASE: case,
    OP.EQUAL: torch.eq,
    OP.TRANSPOSE: _transpose,
    OP.LESS_THAN: torch.less,
    OP.LESS_EQUAL: torch.less_equal,
    OP.EVALUATE: lambda op, *args: op(*args),
    OP.LOG: torch.log,
}


def _norm(op, value):
    if value.ndim == 1:
        return torch.linalg.vector_norm(value, ord=op.ord)
    if value.ndim == 2:
        return torch.linalg.matrix_norm(value, ord=op.ord)
    return torch.linalg.norm(value, ord=op.ord)


parameterised_impls = {
    ConcatenateOP: lambda op, *x: torch.cat(x, dim=op.axis),
    ReshapeOP: lambda op, x: torch.reshape(x, shape=op.newshape),
    NormOP: _norm,
}


def call_parameterised_op(op, *args):
    try:
        return parameterised_impls[op.__class__](op, *args)
    except KeyError as ex:
        raise NotImplementedError(f"{op} is not implemented") from ex


class PytorchModule(torch.nn.Module):
    """Expose a lowered Coker function through PyTorch's module interface."""

    def __init__(self, compiled, input_count, is_single):
        super().__init__()
        self._compiled = compiled
        self._input_count = input_count
        self._is_single = is_single

    def forward(self, *inputs):
        if len(inputs) != self._input_count:
            raise TypeError(
                f"Expected {self._input_count} inputs, got {len(inputs)}"
            )
        outputs = self._compiled(inputs)
        return outputs[0] if self._is_single else tuple(outputs)


def _cast_torch_outputs(function, workspace):
    result = []
    for output_ref in function.output:
        if output_ref is None:
            result.append(None)
            continue
        if output_ref.tape is not function.tape:
            result.append(output_ref)
            continue
        value = workspace[output_ref.index]
        if isinstance(value, Tracer):
            result.append(value)
        elif output_ref.dim.is_scalar():
            result.append(value if value.ndim == 0 else value.reshape(()))
        else:
            result.append(value.reshape(output_ref.shape))
    return result


class PytorchBackend(Backend):
    """Evaluate Coker expression graphs using PyTorch tensors."""

    def to_numpy_array(self, array) -> ArrayLike:
        if array is None:
            return None
        if isinstance(array, torch.Tensor):
            result = array.detach().cpu().numpy()
        else:
            result = np.asarray(array)
        return result.item() if result.shape == () else result

    def to_backend_array(self, array):
        if isinstance(array, torch.Tensor):
            return array
        import scipy.sparse

        if scipy.sparse.issparse(array):
            array = array.toarray()
        return torch.as_tensor(array)

    def reshape(self, arg, dim: Dimension):
        if arg is None:
            return arg
        if dim.is_scalar():
            if isinstance(arg, torch.Tensor):
                if arg.ndim == 0:
                    return arg
                if arg.numel() != 1:
                    raise TypeError(f"Expecting a scalar, got {arg}")
                return arg.reshape(())
            if isinstance(arg, scalar_types):
                return arg
            try:
                (inner,) = arg
            except (ValueError, TypeError) as ex:
                raise TypeError(f"Expecting a scalar, got {arg}") from ex
            return self.reshape(inner, dim)
        if isinstance(arg, torch.Tensor):
            return torch.reshape(arg, dim.dim)
        if isinstance(arg, np.ndarray):
            return np.reshape(arg, dim.dim)
        if isinstance(arg, (float, int, complex)):
            return torch.as_tensor([arg]).reshape(dim.dim)
        raise NotImplementedError(
            f"Don't know how to resize {arg.__class__.__name__}"
        )

    def call(self, op, *args) -> ArrayLike:
        if op in impls:
            return impls[op](*args)
        if isinstance(op, tuple(parameterised_impls.keys())):
            return call_parameterised_op(op, *args)
        raise NotImplementedError(f"{op} is not implemented")

    def resolve_fn(self, op):
        if op in impls:
            return impls[op]
        if isinstance(op, tuple(parameterised_impls.keys())):
            operation = op
            return lambda *args: call_parameterised_op(operation, *args)
        raise NotImplementedError(f"{op} is not implemented")

    def resolve_post_fn(self, dim):
        if not dim.is_scalar():
            return lambda value: value

        def scalar_post(value):
            if isinstance(value, Tracer):
                return value
            return self.reshape(value, dim)

        return scalar_post

    def lower(self, function):
        from coker.backends.evaluator import _build_plan, _cast_outputs

        plan = _build_plan(function.tape, self)
        backend = self

        def compiled(inputs):
            workspace = plan.execute(inputs, backend)
            if any(isinstance(arg, torch.Tensor) for arg in inputs):
                return _cast_torch_outputs(function, workspace)
            return _cast_outputs(
                function.output, function.tape, workspace, backend
            )

        return compiled

    def as_module(self, function):
        """Lower a function to an eager ``torch.nn.Module``.

        The module accepts one positional tensor for each Coker input. It
        returns a tensor for a single-output function and a tuple otherwise.
        Coker constants remain ordinary closure state, so this module exposes
        no trainable parameters or registered buffers.
        """
        return PytorchModule(
            self.lower(function),
            len(function.tape.input_indicies),
            function.is_single,
        )

    def build_optimisation_problem(self, *args, **kwargs):
        raise NotImplementedError(
            "optimisation problem construction is not implemented for the "
            "pytorch backend"
        )

    def create_variational_solver(self, problem):
        raise NotImplementedError(
            "variational solving is not implemented for the pytorch backend"
        )


__all__ = ["PytorchBackend", "PytorchModule"]
