from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from coker.algebra.kernel import InequalityExpression
from coker.dynamics.controls import ControlSolution
from coker.dynamics.variational.polynomials import InterpolatingPolyCollection
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


@dataclass
class VariationalSolution:
    cost: float
    path: InterpolatingPolyCollection
    projectors: Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]
    control_solutions: List[ControlSolution]
    parameter_solutions: Dict[str, float]
    parameters: np.ndarray
    output: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ]
    t_final: float = 0.0
    solve_info: Optional[SolveInfo] = None
    path_constraint_exprs: List[InequalityExpression] = field(
        default_factory=list
    )
    terminal_constraint_exprs: List[InequalityExpression] = field(
        default_factory=list
    )

    def path_constraints(self, t) -> np.ndarray:
        if not self.path_constraint_exprs:
            return np.zeros((0,))
        args = (
            t,
            self.state(t),
            self.algebraic(t),
            self.control_law(t),
            self.parameters,
            self.quadratures(t),
        )
        violations = [
            _evaluate_violation(expr.value(*args), expr.lower, expr.upper)
            for expr in self.path_constraint_exprs
        ]
        return np.concatenate(violations) if violations else np.zeros((0,))

    def terminal_constraints(self) -> np.ndarray:
        if not self.terminal_constraint_exprs:
            return np.zeros((0,))
        t = self.t_final
        args = (
            t,
            self.state(t),
            self.algebraic(t),
            self.control_law(t),
            self.parameters,
            self.quadratures(t),
        )
        violations = [
            _evaluate_violation(expr.value(*args), expr.lower, expr.upper)
            for expr in self.terminal_constraint_exprs
        ]
        return np.concatenate(violations) if violations else np.zeros((0,))

    def as_raw(self) -> np.ndarray:
        points = [
            np.vstack([np.array([t]), x])
            for t, x, _ in self.path.knot_points()
        ]
        return np.hstack(points)

    def state(self, t):
        v = self.path(t)
        proj = self.projectors[0]
        return proj @ v

    def algebraic(self, t):
        if self.projectors[1] is None:
            return None
        return self.projectors[1] @ self.path(t)

    def quadratures(self, t):
        if self.projectors[2] is None:
            return None
        return self.projectors[2] @ self.path(t)

    def control_law(self, t):
        if not self.control_solutions:
            return None
        return np.array([c(t) for c in self.control_solutions])

    def __call__(self, t) -> np.ndarray:
        x = self.state(t)
        q = self.quadratures(t)
        u = self.control_law(t)
        z = self.algebraic(t)
        return self.output(t, x, z, u, self.parameters, q)

    def to_poly(self) -> InterpolatingPolyCollection:

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
