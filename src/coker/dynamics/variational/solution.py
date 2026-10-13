from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from coker.algebra.function import InequalityExpression
from coker.dynamics.variables import ControlSolution
from coker.dynamics.transcription.collocation import (
    InterpolatingPolyCollection,
)
from coker.toolkits.codesign.optimisation import SolveInfo


def _to_flat_array(value) -> np.ndarray:
    if value is None:
        return np.zeros((0,))
    if isinstance(value, (tuple, list)):
        assert len(value) == 1, "Expected a single output value"
        (value,) = value
    return np.array(value, dtype=float).reshape((-1,))


def _evaluate_violation(raw_value, lower, upper) -> np.ndarray:
    values = _to_flat_array(raw_value)
    lower_bounds = _to_flat_array(lower)
    upper_bounds = _to_flat_array(upper)
    assert (
        values.shape == lower_bounds.shape == upper_bounds.shape
    ), "Constraint bounds do not match constraint values"
    violations = []
    for value_i, lower_i, upper_i in zip(values, lower_bounds, upper_bounds):
        has_lower = np.isfinite(lower_i)
        has_upper = np.isfinite(upper_i)
        if has_lower and has_upper:
            raise ValueError(
                "Variational iteration callbacks only support half-space "
                "constraints per component"
            )
        if has_lower:
            violations.append(max(lower_i - value_i, 0.0))
        elif has_upper:
            violations.append(max(value_i - upper_i, 0.0))
    return np.array(violations, dtype=float) if violations else np.zeros((0,))


@dataclass(frozen=True)
class SegmentDefectDiagnostic:
    """Physical segment residuals and mesh provenance from a transcription."""

    normalized_interval: Tuple[float, float]
    physical_interval: Tuple[float, float]
    degree: int
    tolerance: float
    state_residual: np.ndarray
    quadrature_residual: np.ndarray


@dataclass
class VariationalSolution:
    """Inspect a numerical variational trajectory and its solve metadata.

    Ordinary callers obtain this object from a solver rather than constructing
    its interpolation/projector plumbing. Evaluate at scalar physical times
    in ``[0, t_final]``; stack individual evaluations for sampled trajectories.

    Attributes:
        cost: Scalar objective value.
        path: Interpolated concatenated differential/algebraic/quadrature path.
        projectors: Advanced ``(x, z, q)`` extraction matrices; absent
            algebraic/quadrature projectors are ``None``.
        control_solutions: Scalar control solutions in declaration order.
        parameters: Name-to-value mapping. Scalars are floats; vector/dense
            blocks are shaped NumPy arrays; function-valued declarations are
            ``FittedFunction`` objects.
        output: Low-level output evaluator callback.
        t_final: Physical duration returned by the solver.
        solve_info: ``SolveInfo`` status, or ``None`` for manually assembled
            solutions. Read ``solve_info.success`` rather than assuming
            that a candidate (especially an iteration callback) succeeded.
        adaptive_refinement_rounds: CasADi mesh updates, when enabled.
        adaptive_maximum_defect: Final scaled state defect, when enabled.
        segment_defects: Physical segment residuals with mesh provenance.
    """

    cost: float
    path: InterpolatingPolyCollection
    projectors: Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]
    control_solutions: List[ControlSolution]
    parameters: Mapping[str, object]
    output: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ]
    t_final: float = 0.0
    solve_info: Optional[SolveInfo] = None
    adaptive_refinement_rounds: Optional[int] = None
    adaptive_maximum_defect: Optional[float] = None
    segment_defects: Tuple[SegmentDefectDiagnostic, ...] = ()
    path_constraint_exprs: List[InequalityExpression] = field(
        default_factory=list
    )
    terminal_constraint_exprs: List[InequalityExpression] = field(
        default_factory=list
    )
    _parameter_vector: np.ndarray = field(
        default_factory=lambda: np.zeros((0,)), init=False, repr=False
    )
    _solver_parameter_vector: np.ndarray | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_solver(
        cls,
        cost: float,
        path: InterpolatingPolyCollection,
        projectors: Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
        ],
        control_solutions: List[ControlSolution],
        parameters: Mapping[str, object],
        output: Callable[
            [
                float,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
            np.ndarray,
        ],
        parameter_vector: np.ndarray,
        solver_parameter_vector: np.ndarray | None = None,
        *,
        t_final: float = 0.0,
        solve_info: Optional[SolveInfo] = None,
        path_constraint_exprs: List[InequalityExpression] | None = None,
        terminal_constraint_exprs: List[InequalityExpression] | None = None,
    ) -> "VariationalSolution":
        """Assemble a solution from backend-owned interpolation data.

        Args:
            cost: Objective value.
            path: Concatenated state interpolation.
            projectors: Differential/algebraic/quadrature extraction matrices.
            control_solutions: Scalar controls in declaration order.
            parameters: Public named fitted values.
            output: Output callback accepting ``(t, x, z, u, p, q)``.
            parameter_vector: Numeric system parameter vector for evaluation.
            solver_parameter_vector: Optional original decision vector.
            t_final: Physical duration.
            solve_info: Backend-neutral solve metadata.
            path_constraint_exprs: Optional compiled half-space constraints.
            terminal_constraint_exprs: Optional compiled terminal constraints.

        Returns:
            A solution retaining both named parameters and numeric evaluation
            parameters. This is a backend extension constructor.
        """
        solution = cls(
            cost,
            path,
            projectors,
            control_solutions,
            parameters,
            output,
            t_final,
            solve_info,
            path_constraint_exprs=(
                [] if path_constraint_exprs is None else path_constraint_exprs
            ),
            terminal_constraint_exprs=(
                []
                if terminal_constraint_exprs is None
                else terminal_constraint_exprs
            ),
        )
        solution._parameter_vector = parameter_vector
        solution._solver_parameter_vector = solver_parameter_vector
        return solution

    def path_constraints(self, t) -> np.ndarray:
        """Return nonnegative compiled half-space violations at scalar time.

        Args:
            t: Physical time within the solution interval.

        Returns:
            A flat array, empty when no compiled path constraints are retained.

        Raises:
            ValueError: A retained component has both finite bounds.
        """
        if not self.path_constraint_exprs:
            return np.zeros((0,))
        args = (
            t,
            self.state(t),
            self.algebraic(t),
            self.control_law(t),
            self._parameter_vector,
            self.quadratures(t),
        )
        violations = [
            _evaluate_violation(expr.value(*args), expr.lower, expr.upper)
            for expr in self.path_constraint_exprs
        ]
        return np.concatenate(violations) if violations else np.zeros((0,))

    def terminal_constraints(self) -> np.ndarray:
        """Return terminal half-space violations, or an empty flat array.

        Only retained compiled half-space expressions are evaluated; this is
        not a certificate for all transcription equalities or builder records.
        """
        if not self.terminal_constraint_exprs:
            return np.zeros((0,))
        t = self.t_final
        args = (
            t,
            self.state(t),
            self.algebraic(t),
            self.control_law(t),
            self._parameter_vector,
            self.quadratures(t),
        )
        violations = [
            _evaluate_violation(expr.value(*args), expr.lower, expr.upper)
            for expr in self.terminal_constraint_exprs
        ]
        return np.concatenate(violations) if violations else np.zeros((0,))

    def as_raw(self) -> np.ndarray:
        """Return knot data with time in row zero and path components below.

        Returns:
            An array of shape ``(1 + n_x + n_z + n_q, n_knots)`` containing
            interpolation knot samples, not observed output samples.
        """
        points = [
            np.vstack([np.array([t]), x])
            for t, x, _ in self.path.knot_points()
        ]
        return np.hstack(points)

    def state(self, t):
        """Return differential state as a flat ``(n_x,)`` array.

        Args:
            t: Scalar physical time within the solution interval.
        """
        v = self.path(t)
        proj = self.projectors[0]
        return proj @ v

    def algebraic(self, t):
        """Return a flat ``(n_z,)`` algebraic array, or ``None`` if absent.

        Args:
            t: Scalar physical time within the solution interval.
        """
        if self.projectors[1] is None:
            return None
        return self.projectors[1] @ self.path(t)

    def quadratures(self, t):
        """Return a flat ``(n_q,)`` integral array, or ``None`` if absent.

        Args:
            t: Scalar physical time within the solution interval.
        """
        if self.projectors[2] is None:
            return None
        return self.projectors[2] @ self.path(t)

    def control_law(self, t):
        """Return ordered scalar controls as ``(n_controls,)``, or ``None``.

        Args:
            t: Scalar physical time valid for each control declaration.

        Piecewise controls use half-open sample bins and do not include the
        terminal boundary when it lies beyond their stored samples.
        """
        if not self.control_solutions:
            return None
        return np.array([c(t) for c in self.control_solutions])

    def __call__(self, t) -> np.ndarray:
        """Evaluate observed output, not the concatenated state path.

        Args:
            t: Scalar physical time within the solution interval.

        Returns:
            Output in the shape produced by the system's output callback.
        """
        x = self.state(t)
        q = self.quadratures(t)
        u = self.control_law(t)
        z = self.algebraic(t)
        return self.output(t, x, z, u, self._parameter_vector, q)

    def to_poly(self) -> InterpolatingPolyCollection:
        """Map the path to an output interpolation (advanced).

        Returns:
            An ``InterpolatingPolyCollection`` obtained by mapping path knots.
            Unlike ``__call__``, this helper passes the named ``parameters``
            mapping to the output callback; the callback must support that
            layout. Use direct evaluation for ordinary solver output callbacks.
        """

        def f(t, v):
            x = self.projectors[0] @ v
            z = (
                self.projectors[1] @ v
                if self.projectors[1] is not None
                else None
            )
            q = (
                self.projectors[2] @ v
                if self.projectors[2] is not None
                else None
            )
            u = self.control_law(t)
            return self.output(t, x, z, u, self.parameters, q)

        return self.path.map(f)
