"""Dense neural function parameter layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.function import BoundCallable, Function
from coker.parameters import (
    DenseTensorVariable,
    ParameterVariable,
    UnboundedVariable,
)

from .base import FunctionParameter


def _concrete_parameter_name(name: str | None, fallback: str, suffix: str) -> str:
    prefix = name if isinstance(name, str) and name else fallback
    return f"{prefix}_{suffix}"


def _vector_width(space: VectorSpace, description: str) -> int:
    dimension = space.dimension
    if isinstance(dimension, Integral) and not isinstance(dimension, bool):
        width = int(dimension)
    elif (
        isinstance(dimension, tuple)
        and len(dimension) == 1
        and isinstance(dimension[0], Integral)
        and not isinstance(dimension[0], bool)
    ):
        width = int(dimension[0])
    else:
        raise ValueError(f"{description} must be a one-dimensional VectorSpace")
    if width < 1:
        raise ValueError(f"{description} must have positive width")
    return width


def _activation_widths(
    activation: Function | BoundCallable,
) -> tuple[int, int]:
    if not isinstance(activation, (Function, BoundCallable)):
        raise TypeError("activation must be a Coker Function or BoundCallable")
    inputs = (
        activation.input_spaces()
        if isinstance(activation, Function)
        else activation.public_space.arguments
    )
    if len(inputs) != 1 or not isinstance(inputs[0], VectorSpace):
        raise ValueError("activation must have exactly one vector argument")
    input_width = _vector_width(inputs[0], "activation argument")
    outputs = activation.output_shape()
    if (
        len(outputs) != 1
        or not isinstance(outputs[0], Dimension)
        or not outputs[0].is_vector()
    ):
        raise ValueError("activation must have exactly one vector output")
    output_shape = outputs[0].shape
    if len(output_shape) != 1 or output_shape[0] < 1:
        raise ValueError("activation output must have positive width")
    return input_width, output_shape[0]


def _is_scalar_activation(activation: Function | BoundCallable) -> bool:
    if not isinstance(activation, (Function, BoundCallable)):
        raise TypeError("activation must be a Coker Function or BoundCallable")
    inputs = (
        activation.input_spaces()
        if isinstance(activation, Function)
        else activation.public_space.arguments
    )
    outputs = activation.output_shape()
    return (
        len(inputs) == 1
        and isinstance(inputs[0], Scalar)
        and len(outputs) == 1
        and isinstance(outputs[0], Dimension)
        and outputs[0].is_scalar()
    )


def _validate_scalar_target(target: FunctionSpace) -> FunctionSpace:
    if not isinstance(target, FunctionSpace):
        raise TypeError("target must be a FunctionSpace")
    if len(target.arguments) != 1 or not isinstance(target.arguments[0], Scalar):
        raise ValueError("target must have exactly one scalar argument")
    if len(target.output) != 1 or not isinstance(target.output[0], Scalar):
        raise ValueError("target must have exactly one scalar output")
    return target


@dataclass(frozen=True)
class DenseLayer(FunctionParameter):
    """Declare an affine scalar or vector layer with a Coker activation.

    Args:
        input_size: Positive integer input width; must be one for a scalar
            activation. Booleans are not accepted.
        activation: Coker ``Function`` or ``BoundCallable`` with one scalar
            input/output pair, or one positive-width vector input/output pair.
        name: Public declaration name and prefix for concrete block names.

    Raises:
        TypeError: The activation is not a Coker function or bound callable.
        ValueError: Input width or activation signature is incompatible.

    The concrete blocks are weight then bias, both unbounded. Scalar guesses
    are one and zero. Vector guesses are an identity-like matrix of shape
    ``(activation_input_width, input_size)`` and a zero bias vector of activation
    input width. The output width is the activation output width. Supply an
    activation and model whose numeric scales and units are consistent; no unit
    checking or conversion is performed.

    Examples:
        >>> from coker import Scalar, function
        >>> from coker.parameters.function_parameters import DenseLayer
        >>> identity = function([Scalar("x")], lambda x: x, backend="numpy")
        >>> layer = DenseLayer(1, identity, name="gain")
        >>> [block.name for block in layer.list_concrete_parameters()]
        ['gain_weight', 'gain_bias']
        >>> float(layer.evaluate([2.0, 1.0], 3.0))
        7.0
    """

    input_size: int
    activation: Function | BoundCallable
    name: str | None = None
    parameters: tuple[ParameterVariable, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.input_size, bool)
            or not isinstance(self.input_size, Integral)
            or self.input_size < 1
        ):
            raise ValueError("input_size must be a positive integer")
        is_scalar = _is_scalar_activation(self.activation)
        if is_scalar and self.input_size != 1:
            raise ValueError("scalar activation requires input_size 1")
        if not is_scalar:
            _activation_widths(self.activation)
        object.__setattr__(self, "input_size", int(self.input_size))
        object.__setattr__(
            self,
            "parameters",
            (
                (
                    UnboundedVariable(
                        _concrete_parameter_name(self.name, "dense_layer", "weight"),
                        1.0,
                    ),
                    UnboundedVariable(
                        _concrete_parameter_name(self.name, "dense_layer", "bias"),
                        0.0,
                    ),
                )
                if is_scalar
                else (
                    DenseTensorVariable(
                        _concrete_parameter_name(self.name, "dense_layer", "weights"),
                        np.eye(self.hidden_size, self.input_size),
                    ),
                    DenseTensorVariable(
                        _concrete_parameter_name(self.name, "dense_layer", "bias"),
                        np.zeros(self.hidden_size),
                    ),
                )
            ),
        )

    @property
    def hidden_size(self) -> int:
        """int: Activation input width for a vector activation.

        Raises:
            ValueError: The activation is scalar rather than vector-valued.
        """
        return _activation_widths(self.activation)[0]

    @property
    def output_size(self) -> int:
        """int: Activation output width for a vector activation.

        Raises:
            ValueError: The activation is scalar rather than vector-valued.
        """
        return _activation_widths(self.activation)[1]

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        """Validate public input and output widths against the activation.

        Args:
            target: Requested scalar or vector public function space.

        Returns:
            FunctionSpace: The unchanged compatible target.

        Raises:
            TypeError: The target is not a ``FunctionSpace``.
            ValueError: Target arity, scalar/vector kind or widths do not match.
        """
        if _is_scalar_activation(self.activation):
            return _validate_scalar_target(target)

        if not isinstance(target, FunctionSpace):
            raise TypeError("target must be a FunctionSpace")
        if len(target.arguments) != 1 or not isinstance(
            target.arguments[0], VectorSpace
        ):
            raise ValueError("target must have exactly one vector argument")
        if _vector_width(target.arguments[0], "target argument") != self.input_size:
            raise ValueError(
                f"target vector argument must have width {self.input_size}"
            )
        if len(target.output) != 1 or not isinstance(target.output[0], VectorSpace):
            raise ValueError("target must have exactly one vector output")
        if _vector_width(target.output[0], "target output") != self.output_size:
            raise ValueError(f"target vector output must have width {self.output_size}")
        return target

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        """Return the scalar or dense weight and bias declarations as a tuple."""
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        """Apply the activation after an affine scalar or vector transformation.

        Args:
            parameters: Weight and bias values in declaration order and shapes.
            argument: Scalar or vector public input matching ``input_size``.

        Returns:
            Any: Activation result from ``weight * argument + bias`` for
            scalars or ``weight @ argument + bias`` for vectors.
        """
        weight, bias = parameters
        if _is_scalar_activation(self.activation):
            return self.activation(weight * argument + bias)
        return self.activation(weight @ argument + bias)
