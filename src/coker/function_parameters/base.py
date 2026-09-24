"""Function-valued parameter declarations for dynamical systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import Any, Callable, Sequence

import numpy as np

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.function import BoundCallable, Function, function
from coker.dynamics.variables import (
    BoundVector,
    BoundedVariable,
    DenseTensorVariable,
    ParameterVariable,
    UnboundedVariable,
)
from coker.interfaces import FunctionSignatureValue


class FittedFunction(FunctionSignatureValue):
    """A concrete function reconstructed from fitted parameter decisions."""

    def __init__(
        self,
        specification: FunctionParameter,
        space: FunctionSpace,
        function: Callable[..., Any],
        parameters: Any,
    ) -> None:
        self.specification = specification
        self.space = space
        self.function = function
        self.parameters = parameters

    def __call__(self, *arguments: Any, **kwargs: Any) -> Any:
        return self.function(*arguments, **kwargs)

    def input_shape(self):
        return tuple(self.space.input_dimensions())

    def output_shape(self):
        return tuple(self.space.output_dimensions())


class FunctionParameter(ABC):
    """Declare concrete parameter blocks that realize a function parameter."""

    name: str | None

    @abstractmethod
    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        """Validate and return the compatible function parameter space."""

    @abstractmethod
    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        """Return the named parameter blocks used to realize this function."""

    @abstractmethod
    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        """Evaluate the declared function from structured parameter values."""

    def build_function(
        self, target: FunctionSpace, backend: str | None
    ) -> Function:
        """Build a function of the target argument and concrete blocks."""
        target = self.validate_target(target)
        declarations = self.list_concrete_parameters()
        parameter_spaces = [
            _concrete_parameter_space(declaration)
            for declaration in declarations
        ]
        argument_count = len(target.arguments)
        return function(
            [*target.arguments, *parameter_spaces],
            lambda *arguments: self.evaluate(
                tuple(arguments[argument_count:]), arguments[0]
            ),
            backend=backend,
        )


def _concrete_parameter_name(
    name: str | None, fallback: str, suffix: str
) -> str:
    prefix = name if isinstance(name, str) and name else fallback
    return f"{prefix}_{suffix}"


def _concrete_parameter_space(
    declaration: ParameterVariable,
) -> Scalar | VectorSpace:
    if isinstance(declaration, (BoundedVariable, UnboundedVariable)):
        return Scalar(declaration.name)
    if isinstance(declaration, (BoundVector, DenseTensorVariable)):
        return VectorSpace(declaration.name, declaration.shape)
    raise TypeError(
        "function parameter declarations must be scalar or dense variables"
    )


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
        raise ValueError(
            f"{description} must be a one-dimensional VectorSpace"
        )
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
    if len(target.arguments) != 1 or not isinstance(
        target.arguments[0], Scalar
    ):
        raise ValueError("target must have exactly one scalar argument")
    if (
        target.output is None
        or len(target.output) != 1
        or not isinstance(target.output[0], Scalar)
    ):
        raise ValueError("target must have exactly one scalar output")
    return target


def _validate_basis_values(
    guess: Sequence[Real] | None,
    size: int,
    lower_bound: Real | None,
    upper_bound: Real | None,
) -> tuple[tuple[float, ...], float | None, float | None]:
    if (lower_bound is None) != (upper_bound is None):
        raise ValueError("both bounds must be set or omitted")
    if lower_bound is None:
        lower = upper = None
    else:
        if not isinstance(lower_bound, Real) or not isinstance(
            upper_bound, Real
        ):
            raise TypeError("bounds must be real scalars")
        lower, upper = float(lower_bound), float(upper_bound)
        if np.isnan(lower) or np.isnan(upper) or lower >= upper:
            raise ValueError("lower_bound must be less than upper_bound")
    if guess is None:
        values = np.zeros(size)
    else:
        try:
            values = np.asarray(guess, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("guess must be a numeric sequence") from exc
        if values.ndim != 1 or values.size != size:
            raise ValueError(f"guess must have exactly {size} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("guess must contain only finite values")
    if lower is not None and (
        np.any(values < lower) or np.any(values > upper)
    ):
        raise ValueError("guess must lie within the declared bounds")
    return tuple(float(value) for value in values), lower, upper
