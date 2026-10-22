from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

from coker.algebra.function import InequalityExpression
from coker.dynamics.variables import ControlSolution
from coker.dynamics.transcription.collocation import (
    InterpolatingPolyCollection,
    InterpolatingPoly,
)
from coker.toolkits.codesign.optimisation import SolveInfo


FloatArray: TypeAlias = npt.NDArray[np.float64]

VariationalOutput: TypeAlias = Callable[
    [
        float,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
        np.ndarray | None,
    ],
    np.ndarray,
]


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
    """A segment's coupled physical residual and transcription provenance.

    CasADi records the weighted, signed residual vector over the segment,
    including differential, algebraic and quadrature equations. This is not
    the maximum absolute defect used to decide adaptive refinement.

    Args:
        normalized_interval: Segment endpoints in normalised time on ``[0, 1]``.
        physical_interval: Corresponding endpoints in physical time.
        degree: Interpolating polynomial degree.
        tolerance: Transcription's segment-defect tolerance.
        full_residual: One-dimensional floating NumPy residual array.

    Raises:
        TypeError: ``full_residual`` is not a NumPy array with floating dtype.
        ValueError: ``full_residual`` is missing or is not one-dimensional.

    Notes:
        Fields cannot be reassigned, but the residual array is not copied or
        made read-only. Treat solver-provided diagnostics as read-only.

    Examples:
        >>> import numpy as np
        >>> from coker.dynamics.variational.solution import SegmentDefectDiagnostic
        >>> diagnostic = SegmentDefectDiagnostic(
        ...     (0.0, 0.5), (0.0, 2.0), 3, 1e-6, np.zeros(2)
        ... )
        >>> diagnostic.physical_interval
        (0.0, 2.0)
        >>> diagnostic.full_residual.shape
        (2,)
    """

    normalized_interval: Tuple[float, float]
    physical_interval: Tuple[float, float]
    degree: int
    tolerance: float
    full_residual: FloatArray

    def __post_init__(self) -> None:
        residual = self.full_residual
        if residual is None:
            raise ValueError("Segment diagnostics require a full residual")
        if not isinstance(residual, np.ndarray):
            raise TypeError("Segment full residual must be a NumPy array")
        if residual.ndim != 1:
            raise ValueError("Segment full residual must be one-dimensional")
        if not np.issubdtype(residual.dtype, np.floating):
            raise TypeError("Segment full residual must have a floating dtype")


@dataclass
class VariationalSolution:
    """A solved trajectory evaluated in physical time.

    Obtain this result by calling a :class:`coker.dynamics.VariationalProblem`
    or its solver. ``solution(t)`` evaluates the declared output, whereas
    :meth:`state`, :meth:`algebraic`, :meth:`quadratures` and :meth:`control_law`
    inspect individual channels. See the dynamics guide's executable fitting
    example for construction and result inspection.

    Args:
        cost: Final scalar objective value.
        path: Combined state/algebraic/quadrature interpolant in normalised
            time on ``[0, 1]``, not physical time.
        projectors: State, algebraic and quadrature projection matrices in
            that order. Absent optional channels have ``None`` projectors.
        control_solutions: Control representations in declaration order,
            evaluated in normalised time; empty or ``None`` means no controls.
        parameters: Named reconstructed parameter values. CasADi numeric
            decisions are floats or shaped NumPy arrays; PyTorch decisions
            can be tensors. Function decisions are ``FittedFunction`` objects.
        output: Output evaluator with arguments ``(t, x, z, u, p, q)``.
            Use ``solution(t)`` rather than calling this field directly.
        t_final: Solved physical horizon. Solver results have a positive
            horizon; the dataclass default alone is not an evaluable solution.
        solve_info: Solver status metadata, or ``None`` when not supplied,
            for example for a manually constructed result.
        adaptive_refinement_rounds: Completed adaptive refinement rounds,
            or ``None`` when adaptive refinement was not used.
        adaptive_maximum_defect: Final maximum absolute sampled coupled
            residual for adaptive refinement, or ``None`` when not available.
        segment_defects: Segment diagnostics, empty when not supplied.
            CasADi supplies these even without adaptive refinement.
        path_constraint_exprs: Stored path inequalities used by
            :meth:`path_constraints`; empty when no inequalities are stored.
        terminal_constraint_exprs: Stored terminal inequalities used by
            :meth:`terminal_constraints`; empty when none are stored.

    Notes:
        Channel interpolation and raw samples use NumPy arrays. The declared
        output retains its evaluator's shape and numeric return convention;
        fitted parameter values retain their backend's representation.
        This mutable dataclass does not defensively copy constructor inputs.
        Treat its stored arrays, mappings, controls and diagnostics as
        read-only: mutating named parameters does not refit the trajectory or
        synchronise the numeric parameter vector used for output evaluation.

        Evaluation is for scalar physical times within the solved horizon.
        Present trajectory channels reject times outside the path intervals
        with ``ValueError``; no extrapolation or vectorised time-grid contract
        is provided. Absent optional channels return ``None`` without
        evaluating the path.
    """

    cost: float
    path: InterpolatingPolyCollection
    projectors: Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
    control_solutions: List[ControlSolution]
    parameters: Mapping[str, object]
    output: VariationalOutput
    t_final: float = 0.0
    solve_info: Optional[SolveInfo] = None
    adaptive_refinement_rounds: Optional[int] = None
    adaptive_maximum_defect: Optional[float] = None
    segment_defects: Tuple[SegmentDefectDiagnostic, ...] = ()
    path_constraint_exprs: List[InequalityExpression] = field(default_factory=list)
    terminal_constraint_exprs: List[InequalityExpression] = field(default_factory=list)
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
        output: VariationalOutput,
        parameter_vector: np.ndarray,
        solver_parameter_vector: np.ndarray | None = None,
        *,
        t_final: float = 0.0,
        solve_info: Optional[SolveInfo] = None,
        path_constraint_exprs: List[InequalityExpression] | None = None,
        terminal_constraint_exprs: List[InequalityExpression] | None = None,
    ) -> "VariationalSolution":
        """Assemble a result with numeric parameter vectors for evaluation.

        This low-level constructor is intended for solver implementations;
        ordinary callers obtain results from a variational solve. Inputs are
        retained, not defensively copied.

        Args:
            cost: Final scalar objective value.
            path: Combined interpolant on normalised time ``[0, 1]``.
            projectors: State, algebraic and quadrature projection matrices.
            control_solutions: Normalised-time controls in declaration order.
            parameters: Named, backend-materialised parameter values.
            output: Evaluator ``output(t, x, z, u, p, q)``.
            parameter_vector: Numeric system parameter vector passed to output
                and stored constraint evaluators, distinct from named values.
            solver_parameter_vector: Optional full solver decision-parameter
                vector retained for solver reuse.
            t_final: Positive physical horizon for trajectory evaluation.
            solve_info: Solver metadata, or ``None`` if unavailable.
            path_constraint_exprs: Stored path inequalities, or ``None`` for
                an empty list.
            terminal_constraint_exprs: Stored terminal inequalities, or
                ``None`` for an empty list.

        Returns:
            VariationalSolution: Result with its evaluator parameter vectors
            attached. This method does not run a solve or validate success.
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
                [] if terminal_constraint_exprs is None else terminal_constraint_exprs
            ),
        )
        solution._parameter_vector = parameter_vector
        solution._solver_parameter_vector = solver_parameter_vector
        return solution

    def path_constraints(self, t) -> np.ndarray:
        """Evaluate nonnegative stored path-inequality violations.

        Args:
            t: Scalar physical time within the solved horizon.

        Returns:
            numpy.ndarray: Flat violations in stored expression/component
            order. Each half-space contributes its distance outside the bound
            or zero when satisfied. No stored expressions returns an empty
            array without evaluating the trajectory.

        Raises:
            ValueError: A component has both finite lower and upper bounds,
                or the time is outside the path intervals.
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
        """Evaluate stored terminal-inequality violations at ``t_final``.

        Returns:
            numpy.ndarray: Flat nonnegative half-space violations in stored
            expression/component order, or an empty array when no terminal
            expressions are stored. Zero denotes a satisfied component.

        Raises:
            ValueError: A component has both finite lower and upper bounds.

        Notes:
            This does not report all solver constraints: initial point
            constraints and transcription equations are not included.
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
        """Return sampled internal channels with physical knot times.

        Returns:
            numpy.ndarray: New array with one column per retained knot.
            Row zero contains physical times; remaining rows contain the
            combined internal path channels, not the declared output. Each
            segment's right endpoint is omitted, so the final horizon is
            not included. Interior boundaries occur as the next segment's
            starting knot.
        """
        points = [
            np.vstack([np.array([t * self.t_final]), x])
            for t, x, _ in self.path.knot_points()
        ]
        return np.hstack(points)

    def state(self, t):
        """Evaluate the differential state at a physical time.

        Args:
            t: Scalar physical time within ``[0, t_final]``.

        Returns:
            numpy.ndarray: Flat differential-state vector.

        Raises:
            ValueError: The time is outside the path intervals.
        """
        v = self.path(t / self.t_final)
        proj = self.projectors[0]
        return proj @ v

    def algebraic(self, t):
        """Evaluate the optional algebraic channel at a physical time.

        Args:
            t: Scalar physical time within ``[0, t_final]``.

        Returns:
            numpy.ndarray | None: Flat algebraic vector, or ``None`` if no
            algebraic projector is present.

        Raises:
            ValueError: The channel is present and time is outside the path.
        """
        if self.projectors[1] is None:
            return None
        return self.projectors[1] @ self.path(t / self.t_final)

    def quadratures(self, t):
        """Evaluate the optional accumulated quadratures at a physical time.

        Args:
            t: Scalar physical time within ``[0, t_final]``.

        Returns:
            numpy.ndarray | None: Flat quadrature vector, or ``None`` if no
            quadrature projector is present.

        Raises:
            ValueError: The channel is present and time is outside the path.
        """
        if self.projectors[2] is None:
            return None
        return self.projectors[2] @ self.path(t / self.t_final)

    def control_law(self, t):
        """Evaluate the optional controls at a physical time.

        Args:
            t: Scalar physical time within ``[0, t_final]``.

        Returns:
            numpy.ndarray | None: Values stacked in control declaration order,
            or ``None`` when no control solutions are stored. Shape and
            boundary behaviour otherwise follow the stored control types.

        Raises:
            AssertionError: A stored piecewise control selects a coefficient
                beyond its available array.
        """
        if not self.control_solutions:
            return None
        return np.array([c(t / self.t_final) for c in self.control_solutions])

    def __call__(self, t) -> np.ndarray:
        """Evaluate the declared output, rather than the internal path.

        Args:
            t: Scalar physical time within ``[0, t_final]``.

        Returns:
            object: Output from ``output(t, x, z, u, p, q)`` with the
            evaluator's shape and numeric return convention. Absent optional
            channels are passed as ``None``; ``p`` is the stored numeric
            system parameter vector, not the named ``parameters`` mapping.

        Raises:
            ValueError: The time is outside the path intervals.
        """
        x = self.state(t)
        q = self.quadratures(t)
        u = self.control_law(t)
        z = self.algebraic(t)
        return self.output(t, x, z, u, self._parameter_vector, q)

    def to_poly(self) -> InterpolatingPolyCollection:
        """Build a piecewise output interpolant in physical time.

        Returns:
            InterpolatingPolyCollection: New piecewise polynomial with the
            path's intervals scaled by ``t_final`` and the same per-segment
            degrees. Output values are sampled at every interpolation node,
            converted to NumPy, and flattened into output vectors.

        Notes:
            Unlike ``solution(t)``, this interpolates sampled outputs instead
            of evaluating the output function at every query. Nonlinear
            outputs and discontinuous controls need not be reproduced exactly
            between nodes. The returned intervals are physical; ``path``
            remains normalised and is not modified.
        """

        def f(t, v):
            t *= self.t_final
            v = v.reshape(-1)
            x = self.projectors[0] @ v
            z = self.projectors[1] @ v if self.projectors[1] is not None else None
            q = self.projectors[2] @ v if self.projectors[2] is not None else None
            u = self.control_law(t)
            return np.asarray(
                self.output(t, x, z, u, self._parameter_vector, q)
            ).reshape(-1)

        return InterpolatingPolyCollection(
            [
                InterpolatingPoly(
                    poly.dimension,
                    tuple(t * self.t_final for t in poly.interval),
                    poly.degree,
                    poly.values,
                    reference_operators=poly._reference_operators,
                )
                for poly in self.path.map(f).polys
            ]
        )
