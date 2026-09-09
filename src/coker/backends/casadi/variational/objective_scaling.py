"""Internal normalization of scalar variational objectives."""

from __future__ import annotations

import math


def derive_objective_scale(nominal_cost: object, tolerance: object) -> float:
    """Return a finite scale that never amplifies a sub-unit objective."""
    try:
        nominal = float(nominal_cost)
        tol = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        return 1.0

    if not (math.isfinite(nominal) and math.isfinite(tol)):
        return 1.0
    return max(1.0, abs(nominal), tol)
