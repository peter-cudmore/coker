"""PyTorch-native function parameter declarations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace
from coker.backends.backend import import_native_function
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
)
from coker.parameters import DenseTensorVariable, ParameterVariable
from coker.parameters.function_parameters import FunctionParameter


class PytorchModuleParameter(FunctionParameter):
    """Realize a one-input :class:`torch.nn.Linear` as Coker decisions.

    The supplied module is retained as a prototype only. Evaluation supplies
    every parameter through :func:`torch.func.functional_call`, so neither the
    prototype parameters nor its buffers participate in optimization.
    """

    def __init__(
        self, prototype: torch.nn.Module, *, name: str | None = None
    ) -> None:
        if not isinstance(prototype, torch.nn.Module):
            raise TypeError("prototype must be a torch.nn.Module")
        if type(prototype) is not torch.nn.Linear:
            raise ValueError(
                "only torch.nn.Linear modules have a deterministically "
                "derivable Coker function signature; provide a supported "
                "module without external input metadata"
            )
        if prototype.in_features < 1 or prototype.out_features < 1:
            raise ValueError("torch.nn.Linear feature counts must be positive")
        if tuple(prototype.weight.shape) != (
            prototype.out_features,
            prototype.in_features,
        ):
            raise ValueError("torch.nn.Linear weight has an unsupported shape")
        if prototype.bias is not None and tuple(prototype.bias.shape) != (
            prototype.out_features,
        ):
            raise ValueError("torch.nn.Linear bias has an unsupported shape")
        if tuple(prototype.named_buffers()):
            raise ValueError(
                "torch.nn.Linear modules with buffers are unsupported by "
                "import_module_parameter"
            )

        parameter_items = tuple(prototype.named_parameters())
        expected_parameter_names = (
            ("weight", "bias") if prototype.bias is not None else ("weight",)
        )
        if (
            tuple(name for name, _ in parameter_items)
            != expected_parameter_names
        ):
            raise ValueError(
                "torch.nn.Linear parameters must have the standard weight "
                "and optional bias layout"
            )
        if any(
            not parameter.is_floating_point() or parameter.ndim == 0
            for _, parameter in parameter_items
        ):
            raise ValueError(
                "import_module_parameter requires non-scalar floating-point "
                "module parameters"
            )
        if len({id(parameter) for _, parameter in parameter_items}) != len(
            parameter_items
        ):
            raise ValueError(
                "import_module_parameter does not support shared module "
                "parameters"
            )

        self.prototype = prototype
        self.name = name
        self._input_space = _feature_space("input", prototype.in_features)
        self._output_space = _feature_space("output", prototype.out_features)
        self.signature = FunctionSignature(
            inputs=(FunctionInputSpec("input", self._input_space),),
            outputs=(FunctionOutputSpec("output", self._output_space),),
        )
        prefix = name or "pytorch_module"
        self.parameters = tuple(
            DenseTensorVariable(
                f"{prefix}_{parameter_name.replace('.', '_')}",
                parameter.detach()
                .to(dtype=torch.float64, device="cpu")
                .numpy()
                .copy(),
            )
            for parameter_name, parameter in parameter_items
        )

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        """Require the target to match the derived module input and output."""
        if not isinstance(target, FunctionSpace):
            raise TypeError("target must be a FunctionSpace")
        expected = FunctionSpace(
            "pytorch_module",
            arguments=[self._input_space],
            output=[self._output_space],
        )
        if not target.matches_signature(expected):
            raise ValueError(
                "target must match the imported module's single input and "
                "single output signature"
            )
        return target

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        """Return one unbounded dense Coker block per module parameter."""
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        """Evaluate a module state supplied entirely by Coker decisions."""
        if len(parameters) != len(self.parameters):
            raise ValueError(
                "module parameter values do not match imported module "
                "parameters"
            )

        argument_tensor = _as_argument_tensor(argument, self.prototype.weight)
        expected_input_shape = (
            ()
            if isinstance(self._input_space, Scalar)
            else (self._input_space.size,)
        )
        if tuple(argument_tensor.shape) != expected_input_shape:
            raise ValueError(
                "module argument does not match the imported module input "
                "shape"
            )
        native_argument = argument_tensor.reshape(self.prototype.in_features)
        parameter_values = {
            parameter_name: _as_parameter_tensor(
                value,
                prototype_value,
                dtype=native_argument.dtype,
                device=native_argument.device,
            )
            for (parameter_name, prototype_value), value in zip(
                self.prototype.named_parameters(), parameters
            )
        }
        result = torch.func.functional_call(
            self.prototype,
            (parameter_values, {}),
            (native_argument,),
            strict=True,
        )
        return (
            result.reshape(())
            if isinstance(self._output_space, Scalar)
            else result.reshape(self._output_space.size)
        )

    def build_function(self, target: FunctionSpace, backend: str | None):
        """Build a native PyTorch call with explicit Coker decision inputs."""
        target = self.validate_target(target)
        if backend not in (None, "pytorch"):
            raise ValueError(
                "PytorchModuleParameter can only be built for the pytorch "
                "backend"
            )
        parameter_spaces = tuple(
            VectorSpace(parameter.name, parameter.shape)
            for parameter in self.parameters
        )
        signature = FunctionSignature(
            inputs=(
                *(
                    FunctionInputSpec(argument.name, argument)
                    for argument in target.arguments
                ),
                *(
                    FunctionInputSpec(parameter.name, space)
                    for parameter, space in zip(
                        self.parameters, parameter_spaces
                    )
                ),
            ),
            outputs=tuple(
                FunctionOutputSpec(output.name, output)
                for output in target.output
            ),
        )
        return import_native_function(
            lambda argument, *parameters: self.evaluate(parameters, argument),
            signature,
            backend="pytorch",
            name=self.name,
        )


def _feature_space(name: str, size: int) -> Scalar | VectorSpace:
    """Map a linear feature count to Coker's scalar/vector convention."""
    return Scalar(name) if size == 1 else VectorSpace(name, size)


def _as_argument_tensor(
    argument: Any, prototype_weight: torch.Tensor
) -> torch.Tensor:
    """Convert a direct argument without detaching an existing tensor."""
    if isinstance(argument, torch.Tensor):
        return argument
    return torch.as_tensor(
        argument,
        dtype=prototype_weight.dtype,
        device=prototype_weight.device,
    )


def _as_parameter_tensor(
    value: Any,
    prototype_value: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Validate and cast a decision block without breaking autograd."""
    tensor = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(
            value, dtype=prototype_value.dtype, device=prototype_value.device
        )
    )
    if tuple(tensor.shape) != tuple(prototype_value.shape):
        raise ValueError(
            "module parameter value does not match imported module parameter "
            "shape"
        )
    return tensor.to(dtype=dtype, device=device)
