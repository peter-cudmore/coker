"""Monotone piecewise-linear function parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Sequence

import numpy as np

from coker.algebra.dimensions import FunctionSpace, Scalar
from coker.algebra.graph import if_then_else
from coker.parameters import DenseTensorVariable, ParameterVariable

from .base import FunctionParameter


def _concrete_parameter_name(
    name: str | None, fallback: str, suffix: str
) -> str:
    prefix = name if isinstance(name, str) and name else fallback
    return f"{prefix}_{suffix}"


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
    guess: Sequence[Real] | None, size: int
) -> tuple[float, ...]:
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
    return tuple(float(value) for value in values)


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
        guess = _validate_basis_values(self.guess, domain_knots.size)
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
