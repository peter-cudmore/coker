"""Function-valued parameter declarations for dynamical systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np

from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace
from coker.algebra.function import Function, function
from coker.algebra.graph import if_then_else


class FunctionParameter(ABC):
    """Declare finite numeric decisions that realize a function parameter."""

    name: str | None

    @abstractmethod
    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        """Validate and return the compatible function parameter space."""

    @abstractmethod
    def decision_declarations(
        self,
    ) -> (
        tuple[VectorSpace, np.ndarray]
        | tuple[VectorSpace, np.ndarray, np.ndarray, np.ndarray]
    ):
        """Return a basis space, initial values, and optional bounds."""

    @abstractmethod
    def evaluate(self, basis: Any, argument: Any) -> Any:
        """Evaluate the declared function from basis decisions."""

    def build_function(
        self, target: FunctionSpace, backend: str | None
    ) -> Function:
        """Build a backend-native function of basis values and the argument."""
        target = self.validate_target(target)
        basis, *_ = self.decision_declarations()
        return function(
            [*target.arguments, basis],
            lambda argument, basis_values: self.evaluate(
                basis_values, argument
            ),
            backend=backend,
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
    """Bounded scalar realization over fixed function-domain knots."""

    domain_knots: Sequence[Real]
    lower_bound: Real
    upper_bound: Real
    guess: Sequence[Real] | None = None
    name: str | None = None

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

    @property
    def size(self) -> int:
        return len(self.domain_knots)

    @property
    def basis_space(self) -> VectorSpace:
        return VectorSpace("theta", self.size)

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        return _validate_scalar_target(target)

    def decision_declarations(self) -> tuple[VectorSpace, np.ndarray]:
        return self.basis_space, np.asarray(self.guess, dtype=float)

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

    def evaluate(self, basis: Any, argument: Any) -> Any:
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
class Perceptron(FunctionParameter):
    """Logistic affine scalar output over a fixed-width vector input."""

    input_size: int
    lower_bound: Real | None = None
    upper_bound: Real | None = None
    guess: Sequence[Real] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.input_size, bool)
            or not isinstance(self.input_size, Integral)
            or self.input_size < 1
        ):
            raise ValueError("input_size must be a positive integer")
        size = int(self.input_size) + 1
        guess, lower, upper = _validate_basis_values(
            self.guess, size, self.lower_bound, self.upper_bound
        )
        object.__setattr__(self, "input_size", int(self.input_size))
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "guess", guess)

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        if not isinstance(target, FunctionSpace):
            raise TypeError("target must be a FunctionSpace")
        if len(target.arguments) != 1 or not isinstance(
            target.arguments[0], VectorSpace
        ):
            raise ValueError("target must have exactly one vector argument")
        if target.arguments[0].size != self.input_size:
            raise ValueError(
                f"target vector argument must have size {self.input_size}"
            )
        if (
            target.output is None
            or len(target.output) != 1
            or not isinstance(target.output[0], Scalar)
        ):
            raise ValueError("target must have exactly one scalar output")
        return target

    def decision_declarations(
        self,
    ) -> (
        tuple[VectorSpace, np.ndarray]
        | tuple[VectorSpace, np.ndarray, np.ndarray, np.ndarray]
    ):
        size = self.input_size + 1
        result = (
            VectorSpace("perceptron", size),
            np.asarray(self.guess, dtype=float),
        )
        if self.lower_bound is None:
            return result
        return (
            *result,
            np.full(size, self.lower_bound),
            np.full(size, self.upper_bound),
        )

    def evaluate(self, basis: Any, argument: Any) -> Any:
        linear = basis[self.input_size]
        for index in range(self.input_size):
            linear += basis[index] * argument[index]
        return 1.0 / (1.0 + np.exp(-linear))


@dataclass(frozen=True)
class RadialBasisFunction(FunctionParameter):
    """Fixed-width scalar Gaussian basis expansion with fitted coefficients."""

    centers: Sequence[Real]
    width: Real
    lower_bound: Real | None = None
    upper_bound: Real | None = None
    guess: Sequence[Real] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
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
            self.guess,
            centers.size + 1,
            self.lower_bound,
            self.upper_bound,
        )
        object.__setattr__(
            self, "centers", tuple(float(center) for center in centers)
        )
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "guess", guess)

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        return _validate_scalar_target(target)

    def decision_declarations(
        self,
    ) -> (
        tuple[VectorSpace, np.ndarray]
        | tuple[VectorSpace, np.ndarray, np.ndarray, np.ndarray]
    ):
        size = len(self.centers) + 1
        result = (
            VectorSpace("radial_basis", size),
            np.asarray(self.guess, dtype=float),
        )
        if self.lower_bound is None:
            return result
        return (
            *result,
            np.full(size, self.lower_bound),
            np.full(size, self.upper_bound),
        )

    def evaluate(self, basis: Any, argument: Any) -> Any:
        value = basis[-1]
        for index, center in enumerate(self.centers):
            distance = (argument - center) / self.width
            value += basis[index] * np.exp(-0.5 * distance * distance)
        return value
