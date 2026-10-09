"""Optional SUNDIALS IDA integration through CasADi's IDAS plugin.

CasADi distributes the IDAS plugin with its supported binary packages.  The
plugin accepts semi-explicit DAEs, so this module represents a fully implicit
residual ``F(t, w, wdot, z, u, p)`` by making ``wdot`` algebraic and supplying
``w' = wdot`` as the differential equation.  This retains CasADi's symbolic
Jacobian and its sparsity pattern for IDA.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import casadi as ca
import numpy as np

from coker.backends import get_backend_by_name
from coker.backends.casadi.residual_support import (
    lower_residual,
    prepare_initial_conditions,
    residual_sizes,
)
from coker.dynamics.residual import ResidualDynamicalSystem


_IDAS_PLUGIN = "idas"


def is_available() -> bool:
    """Return whether this CasADi installation provides SUNDIALS IDAS."""
    try:
        return bool(ca.has_integrator(_IDAS_PLUGIN))
    except RuntimeError:
        return False


def evaluate_residual_integrals(
    system: ResidualDynamicalSystem,
    initial_conditions,
    end_point,
    inputs,
    solver_parameters=None,
):
    """Integrate a square fully implicit residual with SUNDIALS IDA.

    ``initial_conditions`` is a ``(x0, z0, q0)`` tuple.  Differential and
    quadrature variables are combined into the residual's ``w`` argument,
    while the supplied algebraic initial values are retained as IDA's initial
    algebraic values.  IDA computes a consistent initial derivative from
    those values before advancing the solution.
    """
    if not is_available():
        raise RuntimeError(
            "Residual integration requires CasADi's optional IDAS integrator "
            "plugin, which is not available in this CasADi installation"
        )

    x_size, z_size, q_size = residual_sizes(system)
    w_size = x_size + q_size
    if w_size == 0:
        raise NotImplementedError(
            "IDA residual integration requires a differential or quadrature state"
        )

    x_initial, z_initial, q_initial = prepare_initial_conditions(
        initial_conditions, x_size, z_size, q_size
    )
    w_initial = ca.vertcat(x_initial, q_initial)

    output_times, scalar_end_point = _output_times(end_point)
    if scalar_end_point and output_times[0] == 0.0:
        return x_initial, z_initial if z_size else None, q_initial if q_size else None

    options, initial_rate = _idas_options(solver_parameters, w_size)
    time = ca.MX.sym("residual_time")
    state = ca.MX.sym("residual_state", w_size)
    state_rate = ca.MX.sym("residual_state_rate", w_size)
    algebraic = ca.MX.sym("residual_algebraic", z_size)
    residual = lower_residual(
        get_backend_by_name("casadi"),
        system,
        time,
        state,
        state_rate,
        algebraic,
        inputs,
    )
    expected_rows = w_size + z_size
    if residual.numel() != expected_rows:
        raise ValueError(
            "residual callback returned "
            f"{residual.numel()} rows; expected {expected_rows}"
        )

    # IDAS receives the native semi-explicit system below.  Keeping this as MX
    # rather than materialising a Jacobian preserves the residual graph's
    # structural sparsity for CasADi's automatic differentiation and IDA.
    dae = {
        "x": state,
        "z": ca.vertcat(algebraic, state_rate),
        "ode": state_rate,
        "alg": residual,
    }
    integrator = ca.integrator(
        "residual_idas", _IDAS_PLUGIN, dae, 0.0, output_times, options
    )
    solution = integrator(x0=w_initial, z0=ca.vertcat(z_initial, initial_rate))

    final_state = solution["xf"]
    final_algebraic = solution["zf"]
    x_final = final_state[:x_size, :]
    z_final = final_algebraic[:z_size, :] if z_size else None
    q_final = final_state[x_size:, :] if q_size else None
    return x_final, z_final, q_final


def _initial_value(value: Any, size: int, name: str):
    if value is None:
        raise ValueError(
            f"IDA residual integration requires a {name} initial condition"
        )
    try:
        result = ca.DM(value)
    except RuntimeError as error:
        raise TypeError(
            f"{name} initial condition must be a CasADi numeric value"
        ) from error
    if result.numel() != size:
        raise ValueError(
            f"{name} initial condition has {result.numel()} values; expected {size}"
        )
    return ca.reshape(result, size, 1)


def _optional_initial_value(value: Any, size: int, name: str):
    if size:
        return _initial_value(value, size, name)
    if value is not None:
        try:
            if ca.DM(value).numel() != 0:
                raise ValueError(f"IDA residual integration has no {name} variables")
        except RuntimeError as error:
            raise TypeError(
                f"{name} initial condition must be a CasADi numeric value"
            ) from error
    return ca.DM.zeros(0, 1)


def _output_times(end_point) -> tuple[list[float], bool]:
    if isinstance(end_point, Real):
        end_time = float(end_point)
        if not np.isfinite(end_time):
            raise ValueError("IDA residual end point must be finite")
        return [end_time], True

    output_times = np.asarray(end_point, dtype=float).reshape(-1)
    if output_times.size == 0:
        raise ValueError("IDA residual evaluation requires at least one output time")
    if not np.all(np.isfinite(output_times)):
        raise ValueError("IDA residual evaluation times must be finite")
    increments = np.diff(output_times)
    if not (np.all(increments > 0.0) or np.all(increments < 0.0)):
        raise ValueError("IDA residual evaluation times must be strictly monotonic")
    return output_times.tolist(), False


def _idas_options(solver_parameters, state_size: int) -> tuple[dict[str, Any], ca.DM]:
    if solver_parameters is None:
        supplied_options: dict[str, Any] = {}
    elif isinstance(solver_parameters, Mapping):
        supplied_options = dict(solver_parameters)
    else:
        raise TypeError(
            "IDA residual solver_parameters must be a mapping of CasADi IDAS options"
        )

    if "calc_ic" in supplied_options and not supplied_options["calc_ic"]:
        raise ValueError(
            "IDA residual integration requires calc_ic=True to determine "
            "the initial state derivative"
        )
    supplied_rate = supplied_options.pop("init_xdot", None)
    if supplied_rate is None:
        initial_rate = ca.DM.zeros(state_size, 1)
    else:
        initial_rate = _initial_value(supplied_rate, state_size, "state-rate")

    # IDAS requires an initial derivative guess while ``calc_ic`` replaces it
    # with a consistent rate.  CasADi's options interface accepts a Python
    # double vector here, not the DM used to initialize the algebraic rate.
    options = {
        **supplied_options,
        "calc_ic": True,
        "init_xdot": initial_rate.full().reshape(-1).tolist(),
    }
    return options, initial_rate


__all__ = ["evaluate_residual_integrals", "is_available"]
