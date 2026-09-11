"""PyTorch-specific compiled-plan resolution."""

from typing import Any

from coker.algebra.graph import Tracer
from coker.backends.evaluator import GenericEvaluator
from coker.backends.pytorch.ops import impls, parameterised_impls


class PytorchEvaluator(GenericEvaluator):
    """Compile plans using PyTorch's native operation implementations."""

    def _resolve_operation(self, op):
        if op in impls:
            return impls[op]
        if isinstance(op, tuple(parameterised_impls.keys())):
            return lambda *args: parameterised_impls[type(op)](op, *args)
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
