"""Lowered PyTorch execution adapters for Coker functions."""

import torch

from coker.algebra.kernel import Tracer


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
