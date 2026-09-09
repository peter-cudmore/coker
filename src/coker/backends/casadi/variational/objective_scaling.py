"""Internal normalization of scalar variational objectives.

The optimizer works with a dimensionless objective while callers continue to
observe the objective in physical units.  This module deliberately contains
no CasADi-specific expressions: division and multiplication by ``scale``
work for Python numbers, NumPy scalars/arrays, and CasADi scalar expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class ObjectiveScaling:
    """Affine-free scalar objective scaling.

    ``scale`` is always intended to be finite and strictly positive.  A scale
    of one is the identity transform.  The physical objective is related to
    the solver objective by ``physical = scale * solver``.
    """

    scale: float = 1.0

    def scale_cost(self, physical_cost: Any) -> Any:
        """Convert a physical cost to the normalized solver cost."""

        return physical_cost / self.scale

    def unscale_cost(self, normalized_cost: Any) -> Any:
        """Convert a normalized solver cost to physical units."""

        return normalized_cost * self.scale

    # These names make the direction explicit at call sites that use the
    # general encode/decode terminology used by the other scaling helpers.
    def encode_cost(self, physical_cost: Any) -> Any:
        """Encode a physical cost for the normalized solver."""

        return self.scale_cost(physical_cost)

    def decode_cost(self, normalized_cost: Any) -> Any:
        """Decode a normalized solver cost to physical units."""

        return self.unscale_cost(normalized_cost)


def derive_objective_scaling(
    nominal_cost: Any,
    tolerance: Any,
) -> ObjectiveScaling:
    """Derive a positive objective normalizer from a nominal cost.

    A finite nominal objective and finite tolerance produce
    ``max(1, abs(nominal_cost), tolerance)``. This never amplifies a
    sub-unit objective, which would make its Hessian and gradient less
    well-conditioned. Invalid or non-positive inputs use identity. The
    absolute value only selects magnitude; negative physical objectives
    round-trip unchanged.
    """

    try:
        nominal = float(nominal_cost)
        tol = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        return ObjectiveScaling()

    if not (math.isfinite(nominal) and math.isfinite(tol)):
        return ObjectiveScaling()

    candidate = max(1.0, abs(nominal), tol)
    if not math.isfinite(candidate) or candidate <= 0.0:
        return ObjectiveScaling()
    return ObjectiveScaling(candidate)


__all__ = ["ObjectiveScaling", "derive_objective_scaling"]
