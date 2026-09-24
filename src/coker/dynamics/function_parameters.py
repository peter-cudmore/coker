"""Function-valued parameter declarations for dynamical systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
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
from coker.algebra.graph import if_then_else
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


@dataclass(frozen=True)
class MonotonePiecewiseLinear(FunctionParameter):
    """Bounded monotone scalar realization over normalized-domain knots."""

    domain_knots: Sequence[Real]
    lower_bound: Real
    upper_bound: Real
    guess: Sequence[Real] | None = None
    name: str | None = None
    parameters: tuple[ParameterVariable, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        try:
            domain_knots = np.asarray(self.domain_knots, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "domain_knots must be a one-dimensional numeric sequence"
            ) from exc
        if domain_knots.ndim != 1 or domain_knots.size < 2:
            raise ValueError(
                "domain_knots must have at least two one-dimensional entries"
            )
        if not np.all(np.isfinite(domain_knots)) or not np.all(
            np.diff(domain_knots) > 0
        ):
            raise ValueError(
                "domain_knots must be finite and strictly increasing"
            )
        if not isinstance(self.lower_bound, Real) or not isinstance(
            self.upper_bound, Real
        ):
            raise TypeError("bounds must be real scalars")
        lower, upper = float(self.lower_bound), float(self.upper_bound)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                "lower_bound must be finite and less than upper_bound"
            )
        guess, _, _ = _validate_basis_values(
            self.guess, domain_knots.size, None, None
        )
        object.__setattr__(
            self,
            "domain_knots",
            tuple(float(value) for value in domain_knots),
        )
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "guess", guess)
        object.__setattr__(
            self,
            "parameters",
            (
                DenseTensorVariable(
                    _concrete_parameter_name(self.name, "monotone", "theta"),
                    np.asarray(guess, dtype=float),
                ),
            ),
        )

    @property
    def size(self) -> int:
        return len(self.domain_knots)

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        return _validate_scalar_target(target)

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        return self.parameters

    def _values(self, theta: Any) -> list[Any]:
        weights = [np.exp(theta[i]) for i in range(self.size)] + [1.0, 1.0]
        total = sum(weights)
        cumulative = 0.0
        span = self.upper_bound - self.lower_bound
        values = []
        for weight in weights[:-1]:
            cumulative += weight
            values.append(self.lower_bound + span * cumulative / total)
        return values[: self.size]

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        (basis,) = parameters
        values = self._values(basis)
        result = values[-1]
        for index in range(self.size - 2, -1, -1):
            slope = (values[index + 1] - values[index]) / (
                self.domain_knots[index + 1] - self.domain_knots[index]
            )
            segment = values[index] + slope * (
                argument - self.domain_knots[index]
            )
            result = if_then_else(
                argument <= self.domain_knots[index],
                values[index],
                if_then_else(
                    argument <= self.domain_knots[index + 1],
                    segment,
                    result,
                ),
            )
        return if_then_else(
            argument <= self.domain_knots[0], values[0], result
        )


@dataclass(frozen=True)
class DenseLayer(FunctionParameter):
    """Affine scalar or vector layer followed by a Coker activation."""

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
                        _concrete_parameter_name(
                            self.name, "dense_layer", "weight"
                        ),
                        0.0,
                    ),
                    UnboundedVariable(
                        _concrete_parameter_name(
                            self.name, "dense_layer", "bias"
                        ),
                        0.0,
                    ),
                )
                if is_scalar
                else (
                    DenseTensorVariable(
                        _concrete_parameter_name(
                            self.name, "dense_layer", "weights"
                        ),
                        np.zeros((self.hidden_size, self.input_size)),
                    ),
                    DenseTensorVariable(
                        _concrete_parameter_name(
                            self.name, "dense_layer", "bias"
                        ),
                        np.zeros(self.hidden_size),
                    ),
                )
            ),
        )

    @property
    def hidden_size(self) -> int:
        return _activation_widths(self.activation)[0]

    @property
    def output_size(self) -> int:
        return _activation_widths(self.activation)[1]

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        if _is_scalar_activation(self.activation):
            return _validate_scalar_target(target)

        if not isinstance(target, FunctionSpace):
            raise TypeError("target must be a FunctionSpace")
        if len(target.arguments) != 1 or not isinstance(
            target.arguments[0], VectorSpace
        ):
            raise ValueError("target must have exactly one vector argument")
        if (
            _vector_width(target.arguments[0], "target argument")
            != self.input_size
        ):
            raise ValueError(
                f"target vector argument must have width {self.input_size}"
            )
        if (
            target.output is None
            or len(target.output) != 1
            or not isinstance(target.output[0], VectorSpace)
        ):
            raise ValueError("target must have exactly one vector output")
        if (
            _vector_width(target.output[0], "target output")
            != self.output_size
        ):
            raise ValueError(
                f"target vector output must have width {self.output_size}"
            )
        return target

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        weight, bias = parameters
        if _is_scalar_activation(self.activation):
            return self.activation(weight * argument + bias)
        return self.activation(weight @ argument + bias)


@dataclass(frozen=True)
class RadialBasisFunction(FunctionParameter):
    """Fixed-width scalar Gaussian basis expansion with fitted coefficients."""

    centers: Sequence[Real]
    width: Real
    lower_bound: InitVar[Real | None] = None
    upper_bound: InitVar[Real | None] = None
    guess: InitVar[Sequence[Real] | None] = None
    name: str | None = None
    parameters: tuple[ParameterVariable, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(
        self,
        lower_bound: Real | None,
        upper_bound: Real | None,
        guess: Sequence[Real] | None,
    ) -> None:
        try:
            centers = np.asarray(self.centers, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "centers must be a one-dimensional numeric sequence"
            ) from exc
        if (
            centers.ndim != 1
            or centers.size == 0
            or not np.all(np.isfinite(centers))
        ):
            raise ValueError("centers must be a non-empty finite vector")
        if not isinstance(self.width, Real) or not np.isfinite(self.width):
            raise TypeError("width must be a finite real scalar")
        if self.width <= 0:
            raise ValueError("width must be positive")
        guess, lower, upper = _validate_basis_values(
            guess,
            centers.size + 1,
            lower_bound,
            upper_bound,
        )
        object.__setattr__(
            self, "centers", tuple(float(center) for center in centers)
        )
        object.__setattr__(self, "width", float(self.width))
        name = _concrete_parameter_name(
            self.name, "radial_basis", "coefficients"
        )
        parameter: ParameterVariable
        if lower is None:
            parameter = DenseTensorVariable(
                name, np.asarray(guess, dtype=float)
            )
        else:
            parameter = BoundVector(
                name,
                np.full(centers.size + 1, lower),
                np.full(centers.size + 1, upper),
                np.asarray(guess, dtype=float),
            )
        object.__setattr__(self, "parameters", (parameter,))

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        return _validate_scalar_target(target)

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        (basis,) = parameters
        value = basis[-1]
        for index, center in enumerate(self.centers):
            distance = (argument - center) / self.width
            value += basis[index] * np.exp(-0.5 * distance * distance)
        return value


class ClosureParameter(FunctionParameter):
    """Realize a function parameter by binding decisions to a function."""

    def __init__(
        self, function: Function, parameters: Sequence[ParameterVariable]
    ) -> None:
        if not isinstance(function, Function):
            raise TypeError("function must be a Coker Function")
        if not parameters:
            raise ValueError("parameters must not be empty")
        if not all(
            isinstance(
                parameter,
                (
                    BoundedVariable,
                    BoundVector,
                    DenseTensorVariable,
                    UnboundedVariable,
                ),
            )
            for parameter in parameters
        ):
            raise TypeError(
                "parameters must be scalar or dense parameter declarations"
            )
        self.function = function
        self.parameters = tuple(parameters)
        self.name = function.name

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        if not isinstance(target, FunctionSpace):
            raise TypeError("target must be a FunctionSpace")
        expected_inputs = FunctionSpace(
            "closure_parameter",
            arguments=[
                *(
                    _concrete_parameter_space(parameter)
                    for parameter in self.parameters
                ),
                *target.arguments,
            ],
            output=target.output,
        ).input_dimensions()
        if tuple(self.function.input_shape()) != tuple(expected_inputs):
            raise ValueError(
                "function inputs must be the concrete parameters followed by "
                "the target arguments"
            )
        if tuple(self.function.output_shape()) != tuple(
            target.output_dimensions()
        ):
            raise ValueError("function output must match the target output")
        return target

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        return self.function(*parameters, argument)

    @staticmethod
    def bind_callable(
        implementation: Callable[..., Any],
        parameters: Sequence[ParameterVariable],
        inputs: Sequence[Scalar | VectorSpace],
        *,
        name: str | None = None,
        backend: str = "numpy",
    ) -> ClosureParameter:
        """Trace an implementation whose concrete blocks precede its inputs."""
        function_name = name or f"bound_{implementation.__name__}"
        return ClosureParameter(
            function(
                [
                    *(
                        _concrete_parameter_space(parameter)
                        for parameter in parameters
                    ),
                    *inputs,
                ],
                implementation,
                name=function_name,
                backend=backend,
            ),
            parameters,
        )
