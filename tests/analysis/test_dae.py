import numpy as np

from coker import FunctionSpace, Scalar, VectorSpace
from coker.algebra.ops import Noop
from coker.dynamics import (
    AnalysisStatus,
    DynamicsSpec,
    analyse_controllability,
    analyse_identifiability,
)
from coker.dynamics.system import create_dynamics_from_spec


def _scalar_control_space():
    return FunctionSpace(
        "u", arguments=[Scalar("time")], output=[VectorSpace("u", 1)]
    )


def test_controllability_handles_index_one_dae_constraint_manifold():
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=_scalar_control_space(),
            parameters=None,
            algebraic=VectorSpace("z", 1),
            initial_conditions=lambda _z, _u, _p: (
                np.zeros(1),
                np.zeros(1),
            ),
            dynamics=lambda t, _x, _z, u, _p: u(t),
            constraints=lambda _t, x, z, _u, _p: z - x,
            outputs=lambda _t, _x, z, _u, _p, _q: z,
            quadratures=Noop(),
        ),
        backend="sympy",
    )

    result = analyse_controllability(system)

    assert result.status is AnalysisStatus.TRUE
    assert result.rank == result.required_rank == 1
    assert result.rank_conditions


def test_identifiability_handles_index_one_dae_constraint_manifold():
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=VectorSpace("rate", 1),
            algebraic=VectorSpace("z", 1),
            initial_conditions=lambda _z, _u, _p: (
                np.ones(1),
                np.ones(1),
            ),
            dynamics=lambda _t, x, _z, _u, p: -p[0] * x,
            constraints=lambda _t, x, z, _u, _p: z - x,
            outputs=lambda _t, _x, z, _u, _p, _q: z,
            quadratures=Noop(),
        ),
        backend="sympy",
    )

    result = analyse_identifiability(system)

    assert result.status is AnalysisStatus.TRUE
    assert result.rank == result.required_rank == 2
    assert result.rank_conditions


def test_identifiability_treats_dae_constant_parameter_as_known():
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=VectorSpace("rate", 1),
            algebraic=VectorSpace("z", 1),
            initial_conditions=lambda _z, _u, _p: (
                np.ones(1),
                np.ones(1),
            ),
            dynamics=lambda _t, x, _z, _u, p: -p[0] * x,
            constraints=lambda _t, x, z, _u, _p: z - x,
            outputs=lambda _t, _x, z, _u, _p, _q: z,
            quadratures=Noop(),
        ),
        backend="sympy",
    )

    result = analyse_identifiability(system, parameters=(2.0,))

    assert result.status is AnalysisStatus.TRUE
    assert result.rank == result.required_rank == 1
