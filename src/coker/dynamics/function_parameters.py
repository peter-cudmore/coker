"""Function-valued parameter declarations for dynamical systems."""

from __future__ import annotations
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence
import numpy as np
from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace

from coker.algebra.graph import if_then_else


@dataclass(frozen=True)
class MonotonePiecewiseLinear:
    """Bounded scalar realization over fixed function-domain knots.

    ``domain_knots`` are positions in the parameter function's scalar input
    domain. They are independent of the variational time discretization.
    """

    domain_knots: Sequence[Real]
    lower_bound: Real
    upper_bound: Real
    guess: Sequence[Real] | None = None

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
        if self.guess is None:
            guess = None
        else:
            try:
                arr = np.asarray(self.guess, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError("guess must be a numeric sequence") from exc
            if arr.ndim != 1 or arr.size != domain_knots.size:
                raise ValueError("guess must have one value per domain knot")
            if not np.all(np.isfinite(arr)):
                raise ValueError("guess must contain only finite values")
            guess = tuple(float(v) for v in arr)
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

    @property
    def initial_values(self) -> np.ndarray:
        if self.guess is not None:
            return np.asarray(self.guess, dtype=float).copy()
        return np.zeros(self.size)

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.full(self.size, -np.inf)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.full(self.size, np.inf)

    def decision_declarations(
        self,
    ) -> tuple[VectorSpace, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.basis_space,
            self.initial_values,
            self.lower_bounds,
            self.upper_bounds,
        )

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

    def _evaluate(self, theta: Any, x: Any) -> Any:
        values = self._values(theta)
        result = values[-1]
        for i in range(self.size - 2, -1, -1):
            slope = (values[i + 1] - values[i]) / (
                self.domain_knots[i + 1] - self.domain_knots[i]
            )
            segment = values[i] + slope * (x - self.domain_knots[i])
            result = if_then_else(
                x <= self.domain_knots[i],
                values[i],
                if_then_else(x <= self.domain_knots[i + 1], segment, result),
            )
        return if_then_else(x <= self.domain_knots[0], values[0], result)
