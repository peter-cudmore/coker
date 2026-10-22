"""Shared optimisation constraints and solver result types."""

from dataclasses import dataclass
from typing import Any, Mapping


import numpy as np
from scipy.optimize import OptimizeResult

from coker.algebra.graph import Tracer


@dataclass(frozen=True)
class WeightedNorm:
    """Recognition metadata for a squared weighted residual.

    Ordinary models use :func:`weighted_norm`, which attaches this record to
    its scalar result for backends supporting exact quadratic lowering.

    Args:
        weight: Symbolic weighting matrix.
        residual: Symbolic residual vector.
    """

    weight: Tracer
    residual: Tracer


@dataclass(frozen=True)
class BoundedConstraint:
    """A residual constrained by explicit component-wise bounds.

    Args:
        residual: Symbolic expression to constrain.
        lower_bound: Inclusive numeric lower bound or symbolic bound on the same tape.
        upper_bound: Inclusive numeric upper bound or symbolic bound on the same tape.

    Bound ordering and shape are checked by the selected backend, not by this
    record. Prefer :func:`bounded` when constructing a model.
    """

    residual: Any
    lower_bound: Any
    upper_bound: Any

    @property
    def tape(self):
        """Return the residual's tape, which must also own any symbolic bounds."""
        return self.residual.tape

    def as_halfplane_bound(self):
        """Return ``(residual, lower_bound, upper_bound)`` without conversion."""
        return self.residual, self.lower_bound, self.upper_bound


def bounded(residual: Any, lower_bound: Any, upper_bound: Any) -> BoundedConstraint:
    """Constrain a residual within explicit inclusive bounds.

    Args:
        residual: Symbolic scalar or array expression.
        lower_bound: Numeric or symbolic lower bound. Use negative infinity
            for an unbounded lower side.
        upper_bound: Numeric or symbolic upper bound. Use positive infinity
            for an unbounded upper side. Equal bounds express an equality.

    Returns:
        BoundedConstraint retaining the three arguments unchanged. Symbolic
        bounds must belong to the residual's tape. This helper does not check
        bound ordering or shape; validation is backend-specific.

    Examples:
        >>> from coker.toolkits.codesign import ProblemBuilder, bounded
        >>> builder = ProblemBuilder()
        >>> x = builder.new_variable("x")
        >>> constraint = bounded(x, 0.0, 1.0)
        >>> constraint.lower_bound, constraint.upper_bound
        (0.0, 1.0)
    """
    return BoundedConstraint(residual, lower_bound, upper_bound)


def weighted_norm(weight: Tracer, residual: Tracer) -> Tracer:
    """Return the squared weighted residual ``||weight @ residual||²``.

    Args:
        weight: Symbolic weighting matrix.
        residual: Symbolic residual vector compatible with the matrix.

    Returns:
        Scalar symbolic expression with :class:`WeightedNorm` factor metadata
        attached for exact QP lowering. This is a squared norm, not its square
        root, and does not itself select a QP solver.

    Examples:
        >>> from coker.toolkits.codesign import ProblemBuilder, weighted_norm
        >>> builder = ProblemBuilder()
        >>> weight = builder.new_variable("weight", shape=(2, 2))
        >>> residual = builder.new_variable("residual", shape=(2,))
        >>> expression = weighted_norm(weight, residual)
        >>> expression.dim.is_scalar()
        True
    """
    expression = np.dot(weight @ residual, weight @ residual)
    expression.weighted_norm = WeightedNorm(weight, residual)
    return expression


@dataclass(frozen=True)
class SolveInfo:
    """Immutable, backend-independent outcome of an optimisation solve.

    Args:
        backend: Backend name.
        solver: Solver name.
        success: Whether the solver reported a successful outcome.
        return_status: Backend's diagnostic status string.
        unified_return_status: Normalised status, or ``None`` if unavailable.
        iteration_count: Iteration count, or ``None`` if unavailable.

    Examples:
        >>> from coker.toolkits.codesign import SolveInfo
        >>> info = SolveInfo("numpy", "trust-constr", True, "complete")
        >>> info.success, info.iteration_count
        (True, None)
    """

    backend: str
    solver: str
    success: bool
    return_status: str
    unified_return_status: str | None = None
    iteration_count: int | None = None


class SolveFailure(RuntimeError):
    """Report an unsuccessful solve together with its normalised outcome.

    Args:
        message: Diagnostic text describing the failure.
        solve_info: Status of the failed solve.

    Attributes:
        solve_info: The supplied SolveInfo, also published on builder-produced
            programs when their backend raises this exception.

    Examples:
        >>> from coker.toolkits.codesign import SolveFailure, SolveInfo
        >>> info = SolveInfo("numpy", "trust-constr", False, "iteration limit")
        >>> failure = SolveFailure("solve failed", info)
        >>> failure.solve_info is info
        True
    """

    def __init__(self, message: str, solve_info: SolveInfo):
        super().__init__(message)
        self.solve_info = solve_info


def solve_info_from_casadi_stats(
    stats: Mapping[str, Any], *, solver: str = "ipopt"
) -> SolveInfo:
    """Normalise a CasADi solver statistics mapping.

    Args:
        stats: CasADi statistics containing optional success, status and
            iteration-count fields.
        solver: Solver name recorded in the result, defaulting to ``"ipopt"``.

    Returns:
        SolveInfo with backend ``"casadi"``. Missing success defaults to false,
        missing return status to ``"unknown"`` and optional metadata to None.
    """
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
    result: OptimizeResult, *, solver: str = "trust-constr"
) -> SolveInfo:
    """Normalise the documented result fields of a SciPy optimiser.

    Args:
        result: SciPy optimisation result containing optional ``success``,
            ``message``, ``status`` and ``nit`` fields.
        solver: Solver name recorded in the result.

    Returns:
        SolveInfo with backend ``"numpy"``. The message becomes return status;
        numeric status becomes a string. Missing success defaults to false,
        missing message to ``"unknown"`` and optional metadata to None.
    """
    iteration_count = result.get("nit")
    if iteration_count is not None:
        iteration_count = int(iteration_count)

    status = result.get("status")
    unified_status = None if status is None else str(status)
    success = bool(result.get("success", False))
    return_status = str(result.get("message", "unknown"))
    return SolveInfo(
        backend="numpy",
        solver=solver,
        success=success,
        return_status=return_status,
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
