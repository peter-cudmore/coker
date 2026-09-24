"""Radial-basis function parameters."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from numbers import Real
from typing import Any, Sequence

import numpy as np

from coker.algebra.dimensions import FunctionSpace
from coker.parameters import (
    BoundVector,
    DenseTensorVariable,
    ParameterVariable,
)

from .base import (
    FunctionParameter,
    _concrete_parameter_name,
    _validate_basis_values,
    _validate_scalar_target,
)


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
