"""Lowered PyTorch execution adapters for Coker functions."""

from typing import TYPE_CHECKING, Any, Sequence

import torch

from coker.algebra.kernel import Tracer
from coker.backends.evaluator import _cast_outputs
from coker.backends.lowered import LoweredFunction, LoweringCapabilities

if TYPE_CHECKING:
    from coker.algebra.kernel import Function
    from coker.backends.backend import Backend
    from coker.backends.evaluator import CompiledPlan
    from coker.backends.lowered import FunctionSignature


class PytorchLoweredFunction(LoweredFunction):
    """Execute a reusable PyTorch plan while preserving tensor autograd."""

    def __init__(
        self,
        backend: "Backend",
        function: "Function",
        plan: "CompiledPlan",
    ) -> None:
        self._backend = backend
        self._function = function
        self._plan = plan

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def signature(self) -> "FunctionSignature":
        return self._function.signature

    @property
    def capabilities(self) -> LoweringCapabilities:
        return LoweringCapabilities(
            True, True, True, False, False, False, True, True
        )

    def execute(self, inputs: Sequence[Any]) -> tuple[Any | None, ...]:
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

    def as_module(self) -> "PytorchModule":
        """Expose this handle through an eager ``torch.nn.Module``."""
        return PytorchModule(self)


class PytorchModule(torch.nn.Module):
    """Expose a lowered Coker function through PyTorch's module interface."""

    def __init__(self, lowered: PytorchLoweredFunction) -> None:
        super().__init__()
        self._lowered = lowered
        self._input_count = len(lowered.signature.inputs)

    def forward(self, *inputs: Any) -> Any:
        if len(inputs) != self._input_count:
            raise TypeError(
                f"Expected {self._input_count} inputs, got {len(inputs)}"
            )
        return self._lowered(*inputs)


def cast_torch_outputs(
    function: "Function", workspace: dict[int, Any]
) -> list[Any | None]:
    """Restore declared output shapes without detaching native tensors."""
    result: list[Any | None] = []
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
