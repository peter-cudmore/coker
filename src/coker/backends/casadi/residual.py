"""Direct-collocation integration for coupled residual dynamical systems."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import casadi as ca
import numpy as np

from coker.algebra.ops import Noop
from coker.backends.backend import get_backend_by_name
from coker.dynamics.residual import ResidualDynamicalSystem
from coker.dynamics.transcription.collocation import _build_reference_operators


__all__ = ["evaluate_residual_integrals"]

# A short fixed mesh makes the feasibility transcription accurate enough for
# trajectory evaluation without exposing another public integration policy.
_COLLOCATION_DEGREE = 3
_MAXIMUM_INTERVAL_DURATION = 0.1


def evaluate_residual_integrals(
    system: ResidualDynamicalSystem,
    initial_conditions: Sequence[Any],
    end_point: float | Sequence[float] | np.ndarray,
    inputs: Sequence[Any],
    solver_parameters: object | None = None,
):
    """Integrate a coupled residual DAE by sequential direct collocation.

    Each mesh interval solves the complete residual ``F(t, w, wdot, z, u, p)``
    as a feasibility NLP.  ``w`` contains the differential state followed by
    any system quadrature, so the returned values retain the backend's public
    ``(x, z, q)`` partition.
    """

    x_size = system.differential.flat()
    z_size = 0 if system.algebraic is None else system.algebraic.flat()
    q_size = 0 if system.quadrature is None else system.quadrature.flat()
    state_size = x_size + q_size
    x0, z0, q0 = _validate_initial_conditions(
        initial_conditions, x_size, z_size, q_size
    )
    output_times, scalar_endpoint = _integration_times(end_point)
    initial_state = ca.vertcat(x0, q0)
    if scalar_endpoint and output_times[0] == 0.0:
        return _format_initial_output(
            scalar_endpoint, initial_state, z0, x_size, q_size, z_size
        )
    residual_arguments = _validate_runtime_inputs(system, inputs)

    residual = _build_symbolic_residual(
        system,
        state_size=state_size,
        algebraic_size=z_size,
        arguments=residual_arguments,
    )
    _validate_residual_shape(residual, state_size, z_size)

    if state_size + z_size == 0:
        return _empty_residual_output(scalar_endpoint, output_times, x_size, q_size)

    if output_times.size == 1 and output_times[0] == 0.0:
        return _format_initial_output(
            scalar_endpoint, initial_state, z0, x_size, q_size, z_size
        )

    solver, unpack = _build_collocation_solver(
        residual,
        state_size=state_size,
        algebraic_size=z_size,
        solver_parameters=solver_parameters,
    )
    state_columns = []
    algebraic_columns = [] if z_size else None
    current_state = initial_state
    current_algebraic = z0
    current_time = 0.0

    for output_time in output_times:
        interval_duration = float(output_time - current_time)
        if interval_duration == 0.0:
            state_columns.append(current_state)
            if algebraic_columns is not None:
                algebraic_columns.append(current_algebraic)
            continue
        step_count = max(
            1, int(np.ceil(abs(interval_duration) / _MAXIMUM_INTERVAL_DURATION))
        )
        step_duration = interval_duration / step_count
        for _ in range(step_count):
            current_state, current_algebraic = _solve_collocation_step(
                solver,
                unpack,
                current_time,
                step_duration,
                current_state,
                current_algebraic,
                state_size=state_size,
                algebraic_size=z_size,
            )
            current_time += step_duration
        state_columns.append(current_state)
        if algebraic_columns is not None:
            algebraic_columns.append(current_algebraic)

    if scalar_endpoint:
        return _split_state(current_state, current_algebraic, x_size, q_size, z_size)

    states = ca.horzcat(*state_columns)
    algebraic = (
        ca.horzcat(*algebraic_columns) if algebraic_columns is not None else None
    )
    x = states[:x_size, :]
    q = states[x_size:, :] if q_size else None
    return x, algebraic, q


def _validate_initial_conditions(initial_conditions, x_size, z_size, q_size):
    if (
        not isinstance(initial_conditions, Sequence)
        or isinstance(initial_conditions, (str, bytes))
        or len(initial_conditions) != 3
    ):
        raise ValueError(
            "implicit residual integration requires initial conditions for x, z, and q"
        )
    x0_raw, z0_raw, q0_raw = initial_conditions
    x0 = _numeric_column(x0_raw, x_size, "differential")
    if z_size:
        z0 = _numeric_column(z0_raw, z_size, "algebraic")
    elif z0_raw is not None:
        raise ValueError("implicit residual integration has no algebraic variables")
    else:
        z0 = ca.DM.zeros(0, 1)
    if q_size:
        q0 = _numeric_column(q0_raw, q_size, "quadrature")
    elif q0_raw is not None:
        raise ValueError("implicit residual integration has no quadrature variables")
    else:
        q0 = ca.DM.zeros(0, 1)
    return x0, z0, q0


def _numeric_column(value, size, name):
    if value is None:
        raise ValueError(
            f"implicit residual integration requires a {name} initial condition"
        )
    try:
        result = ca.DM(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"{name} initial condition must be a numeric CasADi-compatible value"
        ) from error
    if result.numel() != size:
        raise ValueError(
            f"{name} initial condition has {result.numel()} values; expected {size}"
        )
    if not result.is_regular():
        raise ValueError(f"{name} initial condition must be finite")
    return ca.reshape(result, size, 1)


def _integration_times(end_point):
    if isinstance(end_point, (float, int, np.floating, np.integer)):
        try:
            final_time = float(end_point)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "implicit residual endpoint must be finite and non-negative"
            ) from error
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError(
                "implicit residual endpoint must be finite and non-negative"
            )
        return np.asarray([final_time]), True

    output_times = np.asarray(end_point, dtype=float)
    if output_times.ndim != 1:
        raise ValueError("implicit residual evaluation times must be one-dimensional")
    if output_times.size == 0:
        raise ValueError("implicit residual evaluation times must not be empty")
    if not np.isfinite(output_times).all():
        raise ValueError("implicit residual evaluation times must be finite")
    if output_times.size > 1 and not (
        np.all(np.diff(output_times) > 0.0) or np.all(np.diff(output_times) < 0.0)
    ):
        raise ValueError(
            "implicit residual evaluation times must be strictly monotonic"
        )
    if output_times[0] == 0.0:
        return output_times, False

    integration_times = np.concatenate((np.zeros((1,)), output_times))
    time_deltas = np.diff(integration_times)
    if not (np.all(time_deltas > 0.0) or np.all(time_deltas < 0.0)):
        raise ValueError(
            "implicit residual evaluation times must be strictly monotonic from t=0"
        )
    return integration_times[1:], False


def _validate_runtime_inputs(system, inputs):
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise TypeError("implicit residual integration inputs must be a sequence")
    expected = len(system.F.input_spaces()) - 4
    if len(inputs) != expected:
        raise ValueError(
            "implicit residual integration inputs do not match the residual callback: "
            f"expected {expected}, got {len(inputs)}"
        )
    values = list(inputs)
    if isinstance(system.inputs, Noop):
        values[0] = Noop()
    return tuple(values)


def _build_symbolic_residual(system, *, state_size, algebraic_size, arguments):
    time = ca.MX.sym("residual_time")
    state = ca.MX.sym("residual_state", state_size, 1)
    state_rate = ca.MX.sym("residual_state_rate", state_size, 1)
    algebraic = ca.MX.sym("residual_algebraic", algebraic_size, 1)
    backend = get_backend_by_name("casadi")
    values = backend.evaluate(
        system.F, (time, state, state_rate, algebraic, *arguments)
    )
    if len(values) != 1 or values[0] is None:
        raise ValueError("implicit residual must return exactly one present output")
    value = values[0]
    if not isinstance(value, (ca.MX, ca.SX, ca.DM)):
        value = ca.DM(value)
    value = cast(ca.MX, ca.reshape(value, value.numel(), 1))
    return ca.Function(
        "residual_collocation_equations",
        [time, state, state_rate, algebraic],
        [value],
    )


def _validate_residual_shape(residual, state_size, algebraic_size):
    expected_size = state_size + algebraic_size
    if residual.size_out(0) != (expected_size, 1):
        actual_size = int(np.prod(residual.size_out(0)))
        raise ValueError(
            "implicit residual must return one row for every state rate and "
            f"algebraic variable; got {actual_size}, expected {expected_size}"
        )


def _build_collocation_solver(
    residual,
    *,
    state_size,
    algebraic_size,
    solver_parameters,
):
    operators = _build_reference_operators(_COLLOCATION_DEGREE)
    nodes = operators.nodes
    derivative = operators.derivative_matrix
    degree = len(nodes) - 1

    start_time = ca.MX.sym("collocation_start_time")
    duration = ca.MX.sym("collocation_duration")
    start_state = ca.MX.sym("collocation_start_state", state_size, 1)
    state_decisions = ca.MX.sym("collocation_state", state_size, degree)
    algebraic_decisions = ca.MX.sym("collocation_algebraic", algebraic_size, degree)
    state_nodes = (start_state,) + tuple(
        state_decisions[:, index] for index in range(degree)
    )
    constraints = []
    for node_index in range(1, degree + 1):
        state_rate = sum(
            (
                (2.0 * derivative[node_index][basis_index] / duration)
                * state_nodes[basis_index]
            )
            for basis_index in range(degree + 1)
        )
        time = start_time + 0.5 * duration * (nodes[node_index] + 1.0)
        constraints.append(
            residual(
                time,
                state_nodes[node_index],
                state_rate,
                algebraic_decisions[:, node_index - 1],
            )
        )
    decision = ca.vertcat(ca.vec(state_decisions), ca.vec(algebraic_decisions))
    constraint = ca.vertcat(*constraints)
    parameters = ca.vertcat(start_time, duration, start_state)
    nlp = {"x": decision, "p": parameters, "f": ca.MX(0), "g": constraint}
    try:
        solver = ca.nlpsol(
            "residual_collocation_solver",
            "ipopt",
            nlp,
            _solver_options(solver_parameters),
        )
    except RuntimeError as error:
        raise RuntimeError(
            "CasADi residual collocation solver construction failed"
        ) from error

    def unpack(decision_value):
        state_values = ca.reshape(
            decision_value[: state_size * degree], state_size, degree
        )
        algebraic_values = ca.reshape(
            decision_value[state_size * degree :], algebraic_size, degree
        )
        return state_values[:, -1], algebraic_values[:, -1]

    return solver, unpack


def _solver_options(solver_parameters):
    options = {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": False,
    }
    if solver_parameters is None:
        return options
    if not isinstance(solver_parameters, Mapping):
        raise TypeError(
            "CasADi residual collocation solver_parameters must be a mapping of "
            "CasADi IPOPT options"
        )
    options.update(solver_parameters)
    return options


def _solve_collocation_step(
    solver,
    unpack,
    start_time,
    duration,
    start_state,
    start_algebraic,
    *,
    state_size,
    algebraic_size,
):
    degree = _COLLOCATION_DEGREE
    initial_guess = ca.vertcat(
        ca.vec(ca.repmat(start_state, 1, degree)),
        ca.vec(ca.repmat(start_algebraic, 1, degree)),
    )
    parameters = ca.vertcat(ca.DM(start_time), ca.DM(duration), start_state)
    row_count = degree * (state_size + algebraic_size)
    try:
        solution = solver(
            x0=initial_guess,
            p=parameters,
            lbg=ca.DM.zeros(row_count, 1),
            ubg=ca.DM.zeros(row_count, 1),
        )
    except RuntimeError as error:
        raise RuntimeError(
            "CasADi residual collocation failed at "
            f"time {start_time + duration}: {error}"
        ) from error
    status = solver.stats()
    if not status.get("success", False):
        raise RuntimeError(
            "CasADi residual collocation did not converge at "
            f"time {start_time + duration}: "
            f"{status.get('return_status', 'unknown status')}"
        )
    solution_vector = solution["x"]
    if not solution_vector.is_regular():
        raise RuntimeError(
            "CasADi residual collocation returned a non-finite solution at "
            f"time {start_time + duration}"
        )
    return unpack(solution_vector)


def _format_initial_output(scalar, state, algebraic, x_size, q_size, z_size):
    if scalar:
        return _split_state(state, algebraic, x_size, q_size, z_size)
    return (
        state[:x_size, :],
        algebraic if z_size else None,
        state[x_size:, :] if q_size else None,
    )


def _split_state(state, algebraic, x_size, q_size, z_size):
    return (
        state[:x_size],
        algebraic if z_size else None,
        state[x_size:] if q_size else None,
    )


def _empty_residual_output(scalar, output_times, x_size, q_size):
    if scalar:
        return (
            ca.DM.zeros(x_size, 1),
            None,
            (ca.DM.zeros(q_size, 1) if q_size else None),
        )
    return (
        ca.DM.zeros(x_size, output_times.size),
        None,
        (ca.DM.zeros(q_size, output_times.size) if q_size else None),
    )
