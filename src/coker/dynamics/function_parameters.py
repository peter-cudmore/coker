"""Function-valued parameter declarations for dynamical systems."""

from __future__ import annotations
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence
import numpy as np
from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace
from coker.algebra.function import Function, function
from coker.algebra.graph import if_then_else


@dataclass(frozen=True)
class MonotonePiecewiseLinear:
    """Bounded scalar piecewise-linear fixed-knot function realization."""

    knots: Sequence[Real]
    lower_bound: Real
    upper_bound: Real
    guess: Sequence[Real] | None = None
    constraint_mode: str = "intrinsic"

    def __post_init__(self) -> None:
        try:
            knots = np.asarray(self.knots, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "knots must be a one-dimensional numeric sequence"
            ) from exc
        if knots.ndim != 1 or knots.size < 2:
            raise ValueError(
                "knots must be one-dimensional with at least two entries"
            )
        if not np.all(np.isfinite(knots)) or not np.all(np.diff(knots) > 0):
            raise ValueError("knots must be finite and strictly increasing")
        if not isinstance(self.lower_bound, Real) or not isinstance(
            self.upper_bound, Real
        ):
            raise TypeError("bounds must be real scalars")
        lower, upper = float(self.lower_bound), float(self.upper_bound)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                "lower_bound must be finite and less than upper_bound"
            )
        if self.constraint_mode not in ("intrinsic", "explicit"):
            raise ValueError(
                "constraint_mode must be 'intrinsic' or 'explicit'"
            )
        if self.guess is None:
            guess = None
        else:
            try:
                arr = np.asarray(self.guess, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError("guess must be a numeric sequence") from exc
            if arr.ndim != 1 or arr.size != knots.size:
                raise ValueError("guess must have one value per knot")
            if not np.all(np.isfinite(arr)):
                raise ValueError("guess must contain only finite values")
            guess = tuple(float(v) for v in arr)
        object.__setattr__(self, "knots", tuple(float(v) for v in knots))
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "guess", guess)

    @property
    def size(self) -> int:
        return len(self.knots)

    @property
    def theta_space(self) -> VectorSpace:
        return VectorSpace("theta", self.size)

    @property
    def parameter_space(self) -> FunctionSpace:
        return FunctionSpace("function", [Scalar("x")], [Scalar("value")])

    @property
    def decision_space(self) -> VectorSpace:
        return self.theta_space

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

    validate_function_space = validate_target

    @property
    def initial_values(self) -> np.ndarray:
        if self.guess is not None:
            return np.asarray(self.guess, dtype=float).copy()
        return (
            np.linspace(self.lower_bound, self.upper_bound, self.size)
            if self.constraint_mode == "explicit"
            else np.zeros(self.size)
        )

    @property
    def lower_bounds(self) -> np.ndarray:
        return (
            np.full(self.size, -np.inf)
            if self.constraint_mode == "intrinsic"
            else np.full(self.size, self.lower_bound)
        )

    @property
    def upper_bounds(self) -> np.ndarray:
        return (
            np.full(self.size, np.inf)
            if self.constraint_mode == "intrinsic"
            else np.full(self.size, self.upper_bound)
        )

    def decision_declarations(
        self,
    ) -> tuple[VectorSpace, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.theta_space,
            self.initial_values,
            self.lower_bounds,
            self.upper_bounds,
        )

    def constraints(self, theta: Any) -> list[Any]:
        return (
            []
            if self.constraint_mode == "intrinsic"
            else [theta[i + 1] - theta[i] for i in range(self.size - 1)]
        )

    def _values(self, theta: Any) -> list[Any]:
        if self.constraint_mode == "explicit":
            return [theta[i] for i in range(self.size)]
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
                self.knots[i + 1] - self.knots[i]
            )
            segment = values[i] + slope * (x - self.knots[i])
            result = if_then_else(
                x <= self.knots[i],
                values[i],
                if_then_else(x <= self.knots[i + 1], segment, result),
            )
        return if_then_else(x <= self.knots[0], values[0], result)

    def symbolic_callable(self, backend: str = "numpy") -> Function:
        return function(
            [self.theta_space, Scalar("x")],
            lambda theta, x: self._evaluate(theta, x),
            backend=backend,
            name="monotone_piecewise_linear",
        )

    build_callable = symbolic_callable

    def bind(self, theta: Any, backend: str = "numpy") -> Function:
        values = np.asarray(theta, dtype=float)
        if values.ndim != 1 or values.size != self.size:
            raise ValueError("theta must contain one value per knot")
        return function(
            [Scalar("x")],
            lambda x: self._evaluate(values, x),
            backend=backend,
            name="bound_monotone_piecewise_linear",
        )
