"""Shared optimisation constraints and solver result types."""

from dataclasses import dataclass
from typing import Any, Mapping


import numpy as np
from scipy.optimize import OptimizeResult

from coker.algebra.graph import Tracer


@dataclass(frozen=True)
class WeightedNorm:
    """Advanced lowering metadata for a squared weighted residual.

    Prefer :func:`weighted_norm` to creating this record manually. Backends
    that implement exact quadratic-program lowering can inspect these factors;
    attaching this record does not select a QP solver.

    Args:
        weight: Linear weighting operator applied to the residual.
        residual: Symbolic residual vector.
    """

    weight: Tracer
    residual: Tracer


@dataclass(frozen=True)
class BoundedConstraint:
    """Residual constrained componentwise by inclusive bounds.

    Args:
        residual: Symbolic scalar or array to constrain.
        lower_bound: Constant or symbolic lower bound on the residual's tape.
            Use ``-np.inf`` for an unbounded lower side.
        upper_bound: Constant or symbolic upper bound on the residual's tape.
            Use ``np.inf`` for an unbounded upper side.

    Notes:
        Equal bounds express equality. Symbolic bounds may depend on runtime
        arguments. Shapes must be supported by the selected backend; this
        record does not validate or normalise them. CasADi lowers symbolic
        bounds to residual inequalities evaluated at solve time.
    """

    residual: Any
    lower_bound: Any
    upper_bound: Any

    @property
    def tape(self):
        """Owning residual tape, used by backend extensions."""
        return self.residual.tape

    def as_halfplane_bound(self):
        """Return ``(residual, lower_bound, upper_bound)`` for lowering."""
        return self.residual, self.lower_bound, self.upper_bound


def bounded(residual: Any, lower_bound: Any, upper_bound: Any) -> BoundedConstraint:
    """Create an inclusive componentwise residual constraint.

    Args:
        residual: Symbolic scalar or array on a ProblemBuilder's tape.
        lower_bound: Constant/symbolic lower bound; ``-np.inf`` is unbounded.
        upper_bound: Constant or symbolic upper bound; ``np.inf`` is unbounded.

    Returns:
        BoundedConstraint to append to ``builder.constraints``. Equal lower
        and upper bounds express equality; symbolic bounds must share the
        residual's tape and may depend on runtime arguments.
    """
    return BoundedConstraint(residual, lower_bound, upper_bound)


def weighted_norm(weight: Tracer, residual: Tracer) -> Tracer:
    """Return a structured squared weighted residual.

    The scalar is mathematically ``||weight @ residual||²``. The factors are
    retained as WeightedNorm metadata for backends that support exact QP
    lowering. This helper does not itself select a solver.

    Args:
        weight: Matrix-like linear weighting operator.
        residual: Symbolic residual vector on the builder's tape.

    Returns:
        Scalar objective tracer suitable for ``Minimise``.
    """
    expression = np.dot(weight @ residual, weight @ residual)
    expression.weighted_norm = WeightedNorm(weight, residual)
    return expression


@dataclass(frozen=True)
class SolveInfo:
    """Immutable normalised outcome reported by an optimisation backend.

    Args:
        backend: Backend registry name, such as ``"casadi"`` or ``"numpy"``.
        solver: Solver/algorithm name.
        success: Whether the backend reports successful termination.
        return_status: Backend's human-readable termination status.
        unified_return_status: Normalised status string when supplied by the
            backend, or None. This is not a cross-backend status enum.
        iteration_count: Solver-reported iteration count, or None when absent.

    Notes:
        A successful termination is not a guarantee of global optimality.
        Solver-dependent status strings and iteration counts should not be
        compared across backends.
    """

    backend: str
    solver: str
    success: bool
    return_status: str
    unified_return_status: str | None = None
    iteration_count: int | None = None


class SolveFailure(RuntimeError):
    """Solver-reported failure carrying its normalised outcome.

    Args:
        message: Human-readable diagnostic.
        solve_info: Failed solve metadata, also exposed by the associated
            MathematicalProgram.

    Attributes:
        solve_info: Normalised failed outcome. Inspect this rather than parsing
            solver logs. No result tuple is returned when this is raised.

    Notes:
        Configuration, argument, and backend-native exceptions are not all
        converted to SolveFailure. They may occur before a new solve outcome
        is available.
    """

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
    result: OptimizeResult, *, solver: str = "trust-constr"
) -> SolveInfo:
    """Normalise a SciPy optimiser result mapping."""
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
