"""Lowered PyTorch execution adapters for Coker functions."""

import torch

from coker.algebra.kernel import Tracer

from coker.backends.evaluator import _cast_outputs
from coker.backends.lowered import LoweredFunction, LoweringCapabilities


class PytorchLoweredFunction(LoweredFunction):
    """Execute a reusable PyTorch plan while preserving tensor autograd."""

    def __init__(self, backend, function, plan):
        self._backend = backend
        self._function = function
        self._plan = plan

    @property
    def backend_name(self):
        return self._backend.name

    @property
    def signature(self):
        return self._function.signature

    @property
    def capabilities(self):
        return LoweringCapabilities(
            True, True, True, False, False, False, True, True
        )

    def execute(self, inputs):
        workspace = self._plan.execute(inputs, self._backend)
        if any(isinstance(arg, torch.Tensor) for arg in inputs):
            return tuple(cast_torch_outputs(self._function, workspace))
        return tuple(
            _cast_outputs(
                self._function.output,
                self._function.tape,
                workspace,
                self._backend,
            )
        )

    def as_module(self):
        """Expose this handle through an eager ``torch.nn.Module``."""
        return PytorchModule(self)


class PytorchModule(torch.nn.Module):
    """Expose a lowered Coker function through PyTorch's module interface."""

    def __init__(self, lowered):
        super().__init__()
        self._lowered = lowered
        self._input_count = len(lowered.signature.inputs)

    def forward(self, *inputs):
        if len(inputs) != self._input_count:
            raise TypeError(
                f"Expected {self._input_count} inputs, got {len(inputs)}"
            )
        return self._lowered(*inputs)


def cast_torch_outputs(function, workspace):
    """Restore declared output shapes without detaching native tensors."""
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
