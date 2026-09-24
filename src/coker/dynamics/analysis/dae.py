"""Symbolic lowering and tangent geometry for semi-explicit index-one DAEs."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from coker.algebra.function import Function
from coker.algebra.ops import Noop
from coker.backends.sympy import SympyBackend
from coker.backends.sympy.analysis import (
    ConstraintGeometry,
    constraint_geometry,
)
from coker.dynamics.model import DynamicalSystem

from .symbolic import (
    SymbolicSystem,
    UnsupportedSystemError,
    _argument_symbols,
    _argument_symbols_for_declarations,
    _control_dimension,
    _control_symbols,
    _control_values,
    _ensure_distinct_symbols,
    _function_expressions,
    _parameter_declarations,
    _reject_non_autonomous_or_implicit,
    _reject_nonsmooth,
    _replace_expressions,
    _require_finite_dimension,
    _scalar_symbol,
    lower_system,
)


@dataclass(frozen=True)
class SymbolicDAESystem:
    """A semi-explicit DAE ``xdot=f(x, z, u, p), 0=g(x, z, p)``."""

    state: tuple[sp.Symbol, ...]
    algebraic: tuple[sp.Symbol, ...]
    parameters: tuple[sp.Symbol, ...]
    controls: tuple[sp.Symbol, ...]
    dynamics: tuple[sp.Expr, ...]
    constraints: tuple[sp.Expr, ...]
    outputs: tuple[sp.Expr, ...]


def normalize_symbolic_system(
    system: object,
) -> SymbolicSystem | SymbolicDAESystem:
    """Return symbolic ODE or DAE data for a supported analysis input."""
    if isinstance(system, (SymbolicSystem, SymbolicDAESystem)):
        return system
    if isinstance(system, DynamicalSystem):
        return (
            lower_dae_system(system)
            if not isinstance(system.g, Noop)
            else lower_system(system)
        )
    raise UnsupportedSystemError(
        "system analysis requires a DynamicalSystem, SymbolicSystem, "
        "or SymbolicDAESystem"
    )


def geometry(system: SymbolicDAESystem) -> ConstraintGeometry:
    """Build index-one tangent operations for a DAE constraint manifold."""
    try:
        return constraint_geometry(
            system.state,
            system.algebraic,
            system.parameters,
            system.constraints,
        )
    except ValueError as error:
        raise UnsupportedSystemError(str(error)) from error


def lower_dae_system(system: DynamicalSystem) -> SymbolicDAESystem:
    """Lower a supported semi-explicit DAE without solving its constraints."""
    if not isinstance(system.g, Function):
        raise UnsupportedSystemError(
            "DAE constraints must be a Coker Function"
        )
    if system.dqdt is not None and not isinstance(system.dqdt, Noop):
        raise UnsupportedSystemError("quadrature states are not supported")

    declarations = _parameter_declarations(system.parameters)
    control_dimension = _control_dimension(system.inputs)
    parameter_slots = len(declarations) if declarations else 1
    dynamics_shapes = system.dxdt.input_shape()
    constraint_shapes = system.g.input_shape()
    output_shapes = system.y.input_shape()
    if (
        len(dynamics_shapes) != 4 + parameter_slots
        or len(constraint_shapes) != 4 + parameter_slots
        or len(output_shapes) != 5 + parameter_slots
    ):
        raise UnsupportedSystemError(
            "DAE has an unsupported argument signature"
        )

    state_dimension = _require_finite_dimension(dynamics_shapes[1], "state")
    algebraic_dimension = _require_finite_dimension(
        dynamics_shapes[2], "algebraic state"
    )
    dynamics_dimension = _require_finite_dimension(
        system.dxdt.output_shape()[0], "dynamics output"
    )
    if dynamics_dimension != state_dimension:
        raise UnsupportedSystemError(
            "dynamics output shape does not match state"
        )

    backend = SympyBackend()
    try:
        dynamics_args, raw_dynamics = backend.lower_to_symbolic(system.dxdt)
        constraint_args, raw_constraints = backend.lower_to_symbolic(system.g)
        output_args, raw_outputs = backend.lower_to_symbolic(system.y)
    except (NotImplementedError, TypeError, ValueError) as error:
        raise UnsupportedSystemError(
            f"SymPy cannot lower this DAE: {error}"
        ) from error

    time = _scalar_symbol(dynamics_args[0], "dynamics time")
    state = _argument_symbols(dynamics_args[1], state_dimension, "state")
    algebraic = _argument_symbols(
        dynamics_args[2], algebraic_dimension, "algebraic state"
    )
    parameters = _argument_symbols_for_declarations(
        dynamics_args[4 : 4 + parameter_slots], declarations, "parameter"
    )
    _ensure_distinct_symbols((time,), state, algebraic, parameters)

    dynamics = _function_expressions(system.dxdt, raw_dynamics, "dynamics")
    constraints = _function_expressions(
        system.g, raw_constraints, "constraints"
    )
    outputs = _function_expressions(system.y, raw_outputs, "outputs")
    if len(constraints) != len(algebraic):
        raise UnsupportedSystemError(
            "index-one DAE constraints must match algebraic state dimension"
        )

    if control_dimension is None:
        controls: tuple[sp.Symbol, ...] = ()
        dynamics_controls: tuple[sp.Expr, ...] = ()
        constraint_controls: tuple[sp.Expr, ...] = ()
        output_controls: tuple[sp.Expr, ...] = ()
    else:
        dynamics_controls = _control_values(
            dynamics_args[3], time, control_dimension, "control"
        )
        constraint_controls = _control_values(
            constraint_args[3],
            _scalar_symbol(constraint_args[0], "constraint time"),
            control_dimension,
            "constraint control",
        )
        output_controls = _control_values(
            output_args[3],
            _scalar_symbol(output_args[0], "output time"),
            control_dimension,
            "output control",
        )
        controls = _control_symbols(dynamics_controls, state, parameters)

    replacements = dict(zip(dynamics_controls, controls))
    dynamics = _replace_expressions(dynamics, replacements)
    constraint_replacements = {
        _scalar_symbol(constraint_args[0], "constraint time"): time,
        **dict(
            zip(
                _argument_symbols(
                    constraint_args[1], state_dimension, "constraint state"
                ),
                state,
            )
        ),
        **dict(
            zip(
                _argument_symbols(
                    constraint_args[2],
                    algebraic_dimension,
                    "constraint algebraic state",
                ),
                algebraic,
            )
        ),
        **dict(
            zip(
                _argument_symbols_for_declarations(
                    constraint_args[4 : 4 + parameter_slots],
                    declarations,
                    "constraint parameter",
                ),
                parameters,
            )
        ),
        **dict(zip(constraint_controls, controls)),
    }
    output_replacements = {
        _scalar_symbol(output_args[0], "output time"): time,
        **dict(
            zip(
                _argument_symbols(
                    output_args[1], state_dimension, "output state"
                ),
                state,
            )
        ),
        **dict(
            zip(
                _argument_symbols(
                    output_args[2],
                    algebraic_dimension,
                    "output algebraic state",
                ),
                algebraic,
            )
        ),
        **dict(
            zip(
                _argument_symbols_for_declarations(
                    output_args[4 : 4 + parameter_slots],
                    declarations,
                    "output parameter",
                ),
                parameters,
            )
        ),
        **dict(zip(output_controls, controls)),
    }
    constraints = _replace_expressions(constraints, constraint_replacements)
    outputs = _replace_expressions(outputs, output_replacements)
    for expressions, subject in (
        (dynamics, "dynamics"),
        (constraints, "constraints"),
        (outputs, "outputs"),
    ):
        _reject_non_autonomous_or_implicit(expressions, time, subject)
        _reject_nonsmooth(expressions, subject)
    if any(expression.has(*controls) for expression in constraints):
        raise UnsupportedSystemError(
            "DAE constraints must not depend on controls"
        )

    return SymbolicDAESystem(
        state, algebraic, parameters, controls, dynamics, constraints, outputs
    )
