"""Shared CasADi bindings for implicit residual dynamical systems."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import casadi as ca

from coker.algebra.dimensions import FunctionSpace
from coker.dynamics.residual import ResidualDynamicalSystem


def residual_sizes(system: ResidualDynamicalSystem) -> tuple[int, int, int]:
    """Return differential, algebraic, and quadrature widths."""
    return (
        system.differential.flat(),
        0 if system.algebraic is None else system.algebraic.flat(),
        0 if system.quadrature is None else system.quadrature.flat(),
    )


def prepare_initial_conditions(
    initial_conditions: Sequence[Any], x_size: int, z_size: int, q_size: int
) -> tuple[ca.DM, ca.DM, ca.DM]:
    """Validate and reshape residual initial conditions as CasADi columns."""
    if (
        not isinstance(initial_conditions, Sequence)
        or isinstance(initial_conditions, (str, bytes))
        or len(initial_conditions) != 3
    ):
        raise ValueError(
            "implicit residual integration requires initial conditions for x, z, and q"
        )
    x_raw, z_raw, q_raw = initial_conditions
    x = numeric_column(x_raw, x_size, "differential")
    z = optional_initial_column(z_raw, z_size, "algebraic")
    q = optional_initial_column(q_raw, q_size, "quadrature")
    return x, z, q


def numeric_column(value: Any, size: int, name: str) -> ca.DM:
    """Return one finite, fixed-width numeric CasADi column."""
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


def optional_initial_column(value: Any, size: int, name: str) -> ca.DM:
    """Return an optional numeric column, rejecting values for absent variables."""
    if size:
        return numeric_column(value, size, name)
    if value is not None:
        numeric_column(value, 0, name)
    return ca.DM.zeros(0, 1)


def lower_residual(
    backend,
    system: ResidualDynamicalSystem,
    time,
    state,
    state_rate,
    algebraic,
    inputs: Sequence[Any],
) -> ca.MX:
    """Lower the coupled residual at already-partitioned symbolic values."""
    control, *parameters = inputs
    residual_control = (
        (lambda _time: control) if isinstance(system.inputs, FunctionSpace) else control
    )
    values = backend.evaluate(
        system.F,
        (time, state, state_rate, algebraic, residual_control, *parameters),
    )
    if len(values) != 1 or values[0] is None:
        raise ValueError("implicit residual must return exactly one present output")
    value = values[0]
    if not isinstance(value, (ca.MX, ca.SX, ca.DM)):
        value = ca.DM(value)
    return cast(ca.MX, ca.reshape(value, value.numel(), 1))


def quiet_ipopt_options(options: dict | None = None) -> dict:
    """Apply the standard silent IPOPT configuration."""
    result = {} if options is None else dict(options)
    result.update({"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False})
    return result
