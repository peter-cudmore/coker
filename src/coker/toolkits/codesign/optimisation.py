"""Shared optimisation constraints and solver result types."""

from dataclasses import dataclass
from typing import Any, Mapping


from coker.algebra.kernel import Tracer


@dataclass(frozen=True)
class WeightedNorm:
    """Structured squared norm objective retained for QP lowering."""

    weight: Tracer
    residual: Tracer


@dataclass(frozen=True)
class BoundedConstraint:
    """A residual constrained by explicit lower and upper bounds.

    Bounds may be constants or symbolic parameters on the residual's tape.
    """

    residual: Any
    lower_bound: Any
    upper_bound: Any

    @property
    def tape(self):
        """Return the tape owning the residual and symbolic bounds."""
        return self.residual.tape

    def as_halfplane_bound(self):
        """Return the residual and its lower and upper bounds."""
        return self.residual, self.lower_bound, self.upper_bound


def bounded(
    residual: Any, lower_bound: Any, upper_bound: Any
) -> BoundedConstraint:
    """Constrain a residual within explicit lower and upper bounds."""
    return BoundedConstraint(residual, lower_bound, upper_bound)

def weighted_norm(weight: Tracer, residual: Tracer) -> Tracer:
    """Return a structured squared weighted residual.

    The returned scalar is mathematically ``||weight @ residual||²`` and
    retains the factors required by exact QP lowering.
    """
    expression = (weight @ residual).T @ (weight @ residual)
    expression.weighted_norm = WeightedNorm(weight, residual)
    return expression


@dataclass(frozen=True)
class SolveInfo:
    """Normalised outcome reported by an optimisation backend."""

    backend: str
    solver: str
    success: bool
    return_status: str
    unified_return_status: str | None = None
    iteration_count: int | None = None


class SolveFailure(RuntimeError):
    """An optimisation solve error carrying its normalised result metadata."""

    def __init__(self, message: str, solve_info: SolveInfo):
        super().__init__(message)
        self.solve_info = solve_info


def solve_info_from_casadi_stats(
    stats: Mapping[str, Any], *, solver: str = "ipopt"
) -> SolveInfo:
    """Normalise a CasADi solver statistics mapping."""
    iteration_count = stats.get("iter_count")
    if iteration_count is not None:
        iteration_count = int(iteration_count)

    unified_status = stats.get("unified_return_status")
    if unified_status is not None:
        unified_status = str(unified_status)

    return SolveInfo(
        backend="casadi",
        solver=solver,
        success=bool(stats.get("success", False)),
        return_status=str(stats.get("return_status", "unknown")),
        unified_return_status=unified_status,
        iteration_count=iteration_count,
    )


def solve_info_from_scipy_result(
    result, *, solver: str = "trust-constr"
) -> SolveInfo:
    """Normalise a SciPy optimiser result object."""
    iteration_count = getattr(result, "nit", None)
    if iteration_count is not None:
        iteration_count = int(iteration_count)

    status = getattr(result, "status", None)
    unified_status = None if status is None else str(status)
    return SolveInfo(
        backend="numpy",
        solver=solver,
        success=bool(getattr(result, "success", False)),
        return_status=str(getattr(result, "message", "unknown")),
        unified_return_status=unified_status,
        iteration_count=iteration_count,
    )

__all__ = [
    "WeightedNorm",
    "BoundedConstraint",
    "SolveFailure",
    "SolveInfo",
    "bounded",
    "weighted_norm",
    "solve_info_from_casadi_stats",
    "solve_info_from_scipy_result",
]
