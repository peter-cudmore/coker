"""ODE and residual-DAE integration for the NumPy backend."""

from enum import Enum

import numpy as np
import scipy as scp

from coker.algebra.ops import Noop
from coker.backends.backend import SolverParameters
from coker.dynamics.residual import (
    ResidualDynamicalSystem,
    _normalise_direct_integration_functions,
)


class Solver(Enum):
    RK45 = "RK45"
    LSODA = "LSODA"
    Radau = "Radau"
    BDF = "BDF"


class NumpySolverParameters(SolverParameters):
    """SciPy ODE and implicit-residual integration settings."""

    DEFAULT_IMPLICIT_MAXIMUM_TIME_STEP = 1e-3
    DEFAULT_IMPLICIT_NONLINEAR_TOLERANCE = np.sqrt(np.finfo(float).eps)

    def __init__(
        self,
        solver: Solver = Solver.RK45,
        *,
        implicit_maximum_time_step: float = DEFAULT_IMPLICIT_MAXIMUM_TIME_STEP,
        implicit_nonlinear_tolerance: float = (DEFAULT_IMPLICIT_NONLINEAR_TOLERANCE),
    ):
        self.solver = solver
        self.implicit_maximum_time_step = self._positive_finite_float(
            implicit_maximum_time_step, "implicit_maximum_time_step"
        )
        self.implicit_nonlinear_tolerance = self._positive_finite_float(
            implicit_nonlinear_tolerance, "implicit_nonlinear_tolerance"
        )

    @staticmethod
    def _positive_finite_float(value, name):
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be finite and positive") from error
        if not np.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result


def evaluate_integrals(
    functions, initial_conditions, end_point, inputs, solver_parameters=None
):
    """Integrate explicit dynamics or a square residual DAE."""
    if isinstance(functions, ResidualDynamicalSystem) and functions.legacy is None:
        return _evaluate_implicit_residual(
            functions,
            initial_conditions,
            end_point,
            inputs,
            solver_parameters,
        )

    dxdt, constraint, dqdt = _normalise_direct_integration_functions(functions)
    x0, z0, q0 = initial_conditions
    u, *parameters = inputs
    if not isinstance(constraint, Noop):
        raise NotImplementedError("Integrators with constraints are not implemented")
    if not isinstance(x0, np.ndarray):
        x0 = np.array([x0])

    if isinstance(end_point, (float, int)):
        if end_point == 0.0:
            return x0, z0, q0
        t_eval, t_span = [end_point], (0, end_point)
    else:
        t_eval, t_span = end_point, (0, end_point[-1])

    quadrature_rate = None if isinstance(dqdt, Noop) else dqdt
    y0 = x0 if quadrature_rate is None else np.concatenate([x0, q0])

    def rhs(time, state):
        state_value = state[: x0.shape[0]]
        state_rate = dxdt(time, state_value, None, u, *parameters)
        if quadrature_rate is None:
            return state_rate
        return np.concatenate(
            [
                state_rate,
                quadrature_rate(time, state_value, None, u, *parameters),
            ]
        )

    method = (
        solver_parameters.solver.value
        if isinstance(solver_parameters, NumpySolverParameters)
        else Solver.RK45.value
    )
    solution = scp.integrate.solve_ivp(rhs, t_span, y0, method=method, t_eval=t_eval)
    return _split_direct_solution(solution.y, x0, dqdt, end_point)


def _split_direct_solution(values, x0, dqdt, end_point):
    is_time_series = isinstance(end_point, np.ndarray)
    x_values = values[: x0.shape[0], :]
    x_out = x_values if is_time_series else x_values[:, -1]
    if isinstance(dqdt, Noop):
        return x_out, None, None
    q_values = values[x0.shape[0] :, :]
    q_out = q_values if is_time_series else q_values[:, -1]
    return x_out, None, q_out


def _residual_initial_value(value, size, name):
    if value is None:
        raise ValueError(
            f"implicit residual integration requires a {name} initial " "condition"
        )
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != size:
        raise ValueError(
            f"{name} initial condition has {result.size} values; " f"expected {size}"
        )
    return result


def _solve_implicit_residual_step(
    root_solver,
    residual,
    root_guess,
    residual_size,
    time,
    nonlinear_tolerance,
):
    root_result = root_solver(residual, root_guess, tol=nonlinear_tolerance)
    root_solution = np.asarray(root_result.x, dtype=float).reshape(-1)
    if root_solution.size != residual_size:
        raise RuntimeError(
            "NumPy implicit residual state-step solve returned "
            f"{root_solution.size} values at time {time}; "
            f"expected {residual_size}"
        )
    if not np.all(np.isfinite(root_solution)):
        raise RuntimeError(
            "NumPy implicit residual state-step solve returned a non-finite "
            f"solution at time {time}"
        )

    candidate_rows = residual(root_solution)
    candidate_residual_norm = np.linalg.norm(candidate_rows, ord=np.inf)
    residual_scale = max(1.0, np.linalg.norm(root_solution, ord=np.inf))
    residual_tolerance = nonlinear_tolerance * residual_scale
    candidate_residual_is_valid = (
        np.all(np.isfinite(candidate_rows))
        and np.isfinite(candidate_residual_norm)
        and np.isfinite(residual_tolerance)
        and candidate_residual_norm <= residual_tolerance
    )
    if not candidate_residual_is_valid:
        raise RuntimeError(
            "NumPy implicit residual state-step solve failed "
            f"at time {time}: {root_result.message}"
        )
    return root_solution


def _evaluate_implicit_residual(
    system, initial_conditions, end_point, inputs, solver_parameters
):
    """Integrate a square index-one residual without a legacy lowering."""
    x0, z0, q0 = initial_conditions
    u, *parameters = inputs
    x_size = system.differential.flat()
    z_size = 0 if system.algebraic is None else system.algebraic.flat()
    q_size = 0 if system.quadrature is None else system.quadrature.flat()
    state_size = x_size + q_size
    if state_size == 0:
        raise NotImplementedError(
            "NumPy implicit residual integration requires a differential or "
            "quadrature state"
        )

    x_initial = _residual_initial_value(x0, x_size, "differential")
    if z_size:
        z_initial = _residual_initial_value(z0, z_size, "algebraic")
    elif z0 is not None:
        raise ValueError("implicit residual integration has no algebraic variables")
    else:
        z_initial = np.empty((0,), dtype=float)
    if q_size:
        q_initial = _residual_initial_value(q0, q_size, "quadrature")
        state_initial = np.concatenate((x_initial, q_initial))
    elif q0 is not None:
        raise ValueError("implicit residual integration has no quadrature variables")
    else:
        q_initial, state_initial = None, x_initial

    integration_times, is_scalar_endpoint, drop_initial = _integration_times(end_point)
    if integration_times is None:
        return _initial_residual_output(
            is_scalar_endpoint, x_initial, z_initial, q_initial, z_size, q_size
        )
    if not np.all(np.isfinite(integration_times)):
        raise ValueError("implicit residual evaluation times must be finite")
    time_deltas = integration_times[1:] - integration_times[:-1]
    if not (np.all(time_deltas > 0) or np.all(time_deltas < 0)):
        raise ValueError(
            "implicit residual evaluation times must be strictly monotonic"
        )

    maximum_time_step, nonlinear_tolerance = _implicit_settings(solver_parameters)
    state_columns, algebraic_columns = [state_initial], (
        [z_initial] if z_size else None
    )
    previous_state, previous_algebraic = state_initial, z_initial
    preceding_state, preceding_time_delta = None, None
    empty_algebraic = np.empty((0,), dtype=float)
    residual_size = state_size + z_size
    root_solver = scp.optimize.root

    for previous_time, output_time in zip(
        integration_times[:-1], integration_times[1:]
    ):
        output_time_delta = output_time - previous_time
        step_count = int(np.ceil(abs(output_time_delta) / maximum_time_step))
        current_time = previous_time
        for step_number in range(step_count):
            next_time = (
                output_time
                if step_number == step_count - 1
                else previous_time + (step_number + 1) * output_time_delta / step_count
            )
            time_delta = next_time - current_time
            weights, preceding_state_for_step = _bdf_weights(
                time_delta,
                preceding_time_delta,
                preceding_state,
                previous_state,
            )

            def residual(unknown):
                next_state = unknown[:state_size]
                next_algebraic = unknown[state_size:] if z_size else empty_algebraic
                state_rate = (
                    weights[0] * next_state
                    + weights[1] * previous_state
                    + weights[2] * preceding_state_for_step
                ) / time_delta
                rows = np.asarray(
                    system.F(
                        next_time,
                        next_state,
                        state_rate,
                        next_algebraic,
                        u,
                        *parameters,
                    ),
                    dtype=float,
                ).reshape(-1)
                if rows.size != residual_size:
                    raise ValueError(
                        "implicit residual must return one row for every "
                        "state rate and algebraic variable"
                    )
                return rows

            root_guess = np.concatenate((previous_state, previous_algebraic))
            root_solution = _solve_implicit_residual_step(
                root_solver,
                residual,
                root_guess,
                residual_size,
                next_time,
                nonlinear_tolerance,
            )
            preceding_state, preceding_time_delta = previous_state, time_delta
            previous_state = root_solution[:state_size]
            previous_algebraic = (
                root_solution[state_size:] if z_size else empty_algebraic
            )
            current_time = next_time
        state_columns.append(previous_state)
        if algebraic_columns is not None:
            algebraic_columns.append(previous_algebraic)

    return _split_residual_solution(
        state_columns,
        algebraic_columns,
        x_size,
        q_size,
        drop_initial,
        is_scalar_endpoint,
    )


def _integration_times(end_point):
    is_scalar_endpoint = isinstance(end_point, (float, int, np.floating, np.integer))
    if is_scalar_endpoint:
        if end_point == 0:
            return None, True, False
        return np.array([0.0, float(end_point)]), True, True
    output_times = np.asarray(end_point, dtype=float)
    if output_times.ndim != 1:
        raise ValueError("implicit residual evaluation times must be one-dimensional")
    if output_times.size == 0:
        raise ValueError("implicit residual evaluation times must not be empty")
    if output_times.size == 1 and output_times[0] == 0:
        return None, False, False
    if output_times[0] == 0:
        return output_times, False, False
    return np.concatenate((np.zeros((1,)), output_times)), False, True


def _initial_residual_output(
    is_scalar, x_initial, z_initial, q_initial, z_size, q_size
):
    if is_scalar:
        return x_initial, z_initial if z_size else None, q_initial
    return (
        x_initial[:, np.newaxis],
        z_initial[:, np.newaxis] if z_size else None,
        q_initial[:, np.newaxis] if q_size else None,
    )


def _implicit_settings(solver_parameters):
    if isinstance(solver_parameters, NumpySolverParameters):
        return (
            solver_parameters.implicit_maximum_time_step,
            solver_parameters.implicit_nonlinear_tolerance,
        )
    return (
        NumpySolverParameters.DEFAULT_IMPLICIT_MAXIMUM_TIME_STEP,
        NumpySolverParameters.DEFAULT_IMPLICIT_NONLINEAR_TOLERANCE,
    )


def _bdf_weights(time_delta, preceding_time_delta, preceding_state, previous_state):
    if preceding_state is None:
        return (1.0, -1.0, 0.0), previous_state
    step_ratio = time_delta / preceding_time_delta
    return (
        (
            (1.0 + 2.0 * step_ratio) / (1.0 + step_ratio),
            -(1.0 + step_ratio),
            step_ratio**2 / (1.0 + step_ratio),
        ),
        preceding_state,
    )


def _split_residual_solution(
    state_columns,
    algebraic_columns,
    x_size,
    q_size,
    drop_initial,
    is_scalar_endpoint,
):
    state_values = np.column_stack(state_columns)
    if drop_initial:
        state_values = state_values[:, 1:]
    if algebraic_columns is not None:
        z_values = np.column_stack(algebraic_columns)
        if drop_initial:
            z_values = z_values[:, 1:]
    else:
        z_values = None
    x_values = state_values[:x_size, :]
    q_values = state_values[x_size:, :] if q_size else None
    if is_scalar_endpoint:
        return (
            x_values[:, -1],
            z_values[:, -1] if z_values is not None else None,
            q_values[:, -1] if q_values is not None else None,
        )
    return x_values, z_values, q_values
