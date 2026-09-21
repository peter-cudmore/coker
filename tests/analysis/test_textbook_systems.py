"""Moderate-size textbook benchmarks for symbolic system analysis."""

import numpy as np

from coker import FunctionSpace, Scalar, VectorSpace
from coker.algebra.ops import Noop
from coker.dynamics import (
    AnalysisStatus,
    DynamicsSpec,
    analyse_controllability,
    analyse_identifiability,
    create_autonomous_ode,
)
from coker.dynamics.system import (
    create_control_system,
    create_dynamics_from_spec,
)


def _control_space(dimension: int) -> FunctionSpace:
    return FunctionSpace(
        "u",
        arguments=[Scalar("time")],
        output=[VectorSpace("u", dimension)],
    )


def _assert_full_rank(result, dimension: int) -> None:
    assert result.status is AnalysisStatus.TRUE
    assert result.rank == dimension
    assert result.required_rank == dimension
    assert result.rank_conditions


def test_controllability_handles_brockett_integrator():
    """The three-state Brockett integrator is bracket-generating."""
    system = create_control_system(
        x0=np.zeros(3),
        control=_control_space(2),
        xdot=lambda _t, x, u, _p: np.asarray([u[0], u[1], x[1] * u[0]]),
        backend="numpy",
    )

    result = analyse_controllability(system)

    _assert_full_rank(result, 3)


def test_controllability_handles_rc_ladder_dae():
    """A three-capacitor RC ladder is controllable from its boundary input."""
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=_control_space(1),
            parameters=None,
            algebraic=VectorSpace("current", 3),
            initial_conditions=lambda _z, _u, _p: (
                np.zeros(3),
                np.zeros(3),
            ),
            dynamics=lambda time, _x, current, control, _p: (
                np.asarray(
                    [
                        current[0] + control(time)[0],
                        current[1],
                        current[2],
                    ]
                )
            ),
            constraints=lambda _t, voltage, current, _u, _p: (
                np.asarray(
                    [
                        current[0] + 2 * voltage[0] - voltage[1],
                        current[1] - voltage[0] + 2 * voltage[1] - voltage[2],
                        current[2] - voltage[1] + 2 * voltage[2],
                    ]
                )
            ),
            outputs=lambda _t, voltage, _z, _u, _p, _q: voltage,
            quadratures=Noop(),
        ),
        backend="sympy",
    )

    result = analyse_controllability(system)

    _assert_full_rank(result, 3)


def test_identifiability_handles_measured_sir_epidemic_model():
    """Infections and removals identify SIR transmission and recovery rates."""
    system = create_autonomous_ode(
        x0=np.ones(3),
        xdot=lambda state, rates: np.asarray(
            [
                -rates[0] * state[0] * state[1],
                rates[0] * state[0] * state[1] - rates[1] * state[1],
                rates[1] * state[1],
            ]
        ),
        parameters=VectorSpace("rate", 2),
        output=lambda state, _rates: np.asarray([state[1], state[2]]),
        backend="sympy",
    )

    result = analyse_identifiability(system)

    _assert_full_rank(result, 5)


def test_identifiability_handles_three_compartment_dae():
    """A three-compartment pharmacokinetic DAE identifies four rates."""
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=VectorSpace("rate", 4),
            algebraic=VectorSpace("flux", 3),
            initial_conditions=lambda _z, _u, _p: (
                np.ones(3),
                np.zeros(3),
            ),
            dynamics=lambda _t, _x, flux, _u, _p: flux,
            constraints=lambda _t, amount, flux, _u, rates: (
                np.asarray(
                    [
                        flux[0] + rates[0] * amount[0] - rates[1] * amount[1],
                        flux[1]
                        - rates[0] * amount[0]
                        + (rates[1] + rates[2]) * amount[1]
                        - rates[3] * amount[2],
                        flux[2] - rates[2] * amount[1] + rates[3] * amount[2],
                    ]
                )
            ),
            outputs=lambda _t, amount, _z, _u, _p, _q: amount,
            quadratures=Noop(),
        ),
        backend="sympy",
    )

    result = analyse_identifiability(system)

    _assert_full_rank(result, 7)
