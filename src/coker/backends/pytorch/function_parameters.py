"""PyTorch-native function parameter declarations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace
from coker.algebra.function import Function
from coker.backends.backend import import_native_function
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
)
from coker.parameters import (
    DenseTensorVariable,
    ParameterVariable,
    UnboundedVariable,
)
from coker.parameters.function_parameters import FunctionParameter


class PytorchModuleParameter(FunctionParameter):
    """Realize a one-input PyTorch module as Coker decisions.

    Every trainable module parameter becomes an unbounded Coker decision block.
    Buffers are captured at construction and cloned for each functional
    evaluation. This leaves prototype parameters and buffers unchanged.
    """

    def __init__(
        self,
        prototype: torch.nn.Module,
        input_space: Scalar | VectorSpace,
        output_space: Scalar | VectorSpace,
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(prototype, torch.nn.Module):
            raise TypeError("prototype must be a torch.nn.Module")
        _validate_space(input_space, "input_space")
        _validate_space(output_space, "output_space")

        parameter_items = tuple(prototype.named_parameters())
        if not parameter_items:
            raise ValueError(
                "import_as_parameter requires at least one trainable module "
                "parameter"
            )
        buffer_items = tuple(prototype.named_buffers())
        if any(
            not isinstance(buffer, torch.Tensor) for _, buffer in buffer_items
        ):
            raise ValueError(
                "import_as_parameter requires tensor module buffers"
            )
        if any(
            not parameter.is_floating_point()
            for _, parameter in parameter_items
        ):
            raise ValueError(
                "import_as_parameter requires floating-point module parameters"
            )
        if len({id(parameter) for _, parameter in parameter_items}) != len(
            parameter_items
        ):
            raise ValueError(
                "import_as_parameter does not support shared module parameters"
            )

        self.prototype = prototype
        self.name = name or "pytorch_module"
        self._input_space = input_space
        self._output_space = output_space
        self._parameter_items = parameter_items
        self._reference_parameter = parameter_items[0][1]
        self._buffer_items = tuple(
            (buffer_name, buffer.detach().clone())
            for buffer_name, buffer in buffer_items
        )
        self.signature = FunctionSignature(
            inputs=(FunctionInputSpec(input_space.name, input_space),),
            outputs=(FunctionOutputSpec(output_space.name, output_space),),
        )
        prefix = self.name
        self.parameters = tuple(
            _parameter_declaration(
                f"{prefix}_{parameter_name.replace('.', '_')}", parameter
            )
            for parameter_name, parameter in parameter_items
        )

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        """Require the target to match the supplied module spaces."""
        if not isinstance(target, FunctionSpace):
            raise TypeError("target must be a FunctionSpace")
        expected = FunctionSpace(
            "pytorch_module",
            arguments=[self._input_space],
            output=[self._output_space],
        )
        if not target.matches_signature(expected):
            raise ValueError(
                "target must match the imported module's supplied single "
                "input and single output spaces"
            )
        return target

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        """Return one unbounded Coker block per module parameter."""
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        """Evaluate a module state supplied entirely by Coker decisions."""
        if len(parameters) != len(self.parameters):
            raise ValueError(
                "module parameter values do not match imported module "
                "parameters"
            )

        argument_tensor = _as_tensor(argument, self._reference_parameter)
        expected_input_shape = _space_shape(self._input_space)
        if tuple(argument_tensor.shape) != expected_input_shape:
            raise ValueError(
                "module argument does not match the supplied input space"
            )
        parameter_values = {
            parameter_name: _as_parameter_tensor(
                value,
                prototype_value,
                dtype=argument_tensor.dtype,
                device=argument_tensor.device,
            )
            for (parameter_name, prototype_value), value in zip(
                self._parameter_items, parameters
            )
        }
        buffer_values = {
            buffer_name: buffer.detach()
            .clone()
            .to(device=argument_tensor.device)
            for buffer_name, buffer in self._buffer_items
        }
        result = torch.func.functional_call(
            self.prototype,
            (parameter_values, buffer_values),
            (argument_tensor,),
            strict=True,
        )
        if not isinstance(result, torch.Tensor):
            raise TypeError("imported module must return one torch.Tensor")
        try:
            return result.reshape(_space_shape(self._output_space))
        except RuntimeError as ex:
            raise ValueError(
                "module result does not match the supplied output space"
            ) from ex

    def build_function(
        self, target: FunctionSpace, backend: str | None
    ) -> Function:
        """Build a native PyTorch call with explicit Coker decision inputs."""
        target = self.validate_target(target)
        if backend not in (None, "pytorch"):
            raise ValueError(
                "PytorchModuleParameter can only be built for the pytorch "
                "backend"
            )
        parameter_spaces = tuple(
            _parameter_space(parameter) for parameter in self.parameters
        )
        signature = FunctionSignature(
            inputs=(
                FunctionInputSpec(
                    target.arguments[0].name, target.arguments[0]
                ),
                *(
                    FunctionInputSpec(parameter.name, space)
                    for parameter, space in zip(
                        self.parameters, parameter_spaces
                    )
                ),
            ),
            outputs=(
                FunctionOutputSpec(target.output[0].name, target.output[0]),
            ),
        )
        return import_native_function(
            lambda argument, *parameters: self.evaluate(parameters, argument),
            signature,
            backend="pytorch",
            name=self.name,
        )


def _validate_space(space: object, argument_name: str) -> None:
    if not isinstance(space, (Scalar, VectorSpace)):
        raise TypeError(f"{argument_name} must be a Scalar or VectorSpace")


def _space_shape(space: Scalar | VectorSpace) -> tuple[int, ...]:
    if isinstance(space, Scalar):
        return ()
    return (
        (space.dimension,)
        if isinstance(space.dimension, int)
        else tuple(space.dimension)
    )


def _parameter_declaration(
    name: str, parameter: torch.Tensor
) -> UnboundedVariable | DenseTensorVariable:
    value = parameter.detach().to(dtype=torch.float64, device="cpu").numpy()
    if parameter.ndim == 0:
        return UnboundedVariable(name, guess=float(value))
    return DenseTensorVariable(name, value.copy())


def _parameter_space(
    parameter: UnboundedVariable | DenseTensorVariable,
) -> Scalar | VectorSpace:
    if isinstance(parameter, UnboundedVariable):
        return Scalar(parameter.name)
    return VectorSpace(parameter.name, parameter.shape)


def _as_tensor(argument: Any, reference: torch.Tensor) -> torch.Tensor:
    """Convert a direct argument without detaching an existing tensor."""
    if isinstance(argument, torch.Tensor):
        return argument
    return torch.as_tensor(
        argument,
        dtype=reference.dtype,
        device=reference.device,
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
