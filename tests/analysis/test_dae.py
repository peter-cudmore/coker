import numpy as np
import pytest
import sympy as sp

from coker import Dimension, FunctionSpace, Scalar, VectorSpace, function
from coker.algebra.ops import Noop
from coker.dynamics import (
    AnalysisStatus,
    ControllabilityResult,
    DynamicsSpec,
    IdentifiabilityResult,
    ObservabilityResult,
    ResidualDynamicalSystem,
    analyse_controllability,
    analyse_identifiability,
    analyse_observability,
    create_dynamics_from_spec,
    lower_dae_system,
)

_ANALYSES = (
    (analyse_observability, ObservabilityResult),
    (analyse_controllability, ControllabilityResult),
    (analyse_identifiability, IdentifiabilityResult),
)


def _scalar_control_space():
    return FunctionSpace("u", arguments=[Scalar("time")], output=[VectorSpace("u", 1)])


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


def test_observability_uses_intrinsic_index_one_dae_rank_and_regularity():
    # z**2 = x[0] and xdot[0] = 2*z*x[1] imply zdot = x[1]
    # on z != 0. Thus h = z and L_f h = x[1] observe both intrinsic states.
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=None,
            algebraic=VectorSpace("z", 1),
            initial_conditions=lambda _z, _u, _p: (
                np.array([4.0, 3.0]),
                np.array([2.0]),
            ),
            dynamics=lambda _t, x, z, _u, _p: np.array([2 * z[0] * x[1], 0]),
            constraints=lambda _t, x, z, _u, _p: np.array([z[0] * z[0] - x[0]]),
            outputs=lambda _t, _x, z, _u, _p, _q: z,
            quadratures=Noop(),
        ),
        backend="sympy",
    )
    symbolic = lower_dae_system(system)
    position, velocity = symbolic.state
    (algebraic,) = symbolic.algebraic
    expected_matrix = sp.Matrix([[1 / (2 * algebraic), 0], [0, 1]])
    regularity = 2 * algebraic
    rank_witness = expected_matrix.det()

    result = analyse_observability(system)

    assert isinstance(result, ObservabilityResult)
    assert result.status is AnalysisStatus.TRUE
    assert result.reason is None
    assert result.rank == result.required_rank == 2
    assert result.matrix.shape == (2, 2)  # Not the ambient (x[0], x[1], z).
    assert (result.matrix - expected_matrix).applyfunc(sp.simplify) == sp.zeros(2)
    assert tuple(
        sp.simplify(actual - expected)
        for actual, expected in zip(
            result.generators, (algebraic, velocity), strict=True
        )
    ) == (0, 0)
    assert any(
        sp.simplify(condition - regularity) == 0 for condition in result.rank_conditions
    )
    assert any(
        sp.simplify(condition - rank_witness) == 0
        for condition in result.rank_conditions
    )
    regular_point = {position: 4, velocity: 3, algebraic: 2}
    assert symbolic.constraints[0].subs(regular_point) == 0
    assert sp.simplify(result.matrix.subs(regular_point).det() - sp.Rational(1, 4)) == 0
    assert all(
        condition.subs(regular_point) != 0 for condition in result.rank_conditions
    )


@pytest.mark.parametrize(("analyse", "result_type"), _ANALYSES)
def test_analysis_reports_higher_index_dae_as_inconclusive(analyse, result_type):
    # xdot = z, x = 0 is index two: differentiating the constraint once
    # gives z = 0. Its algebraic constraint Jacobian is identically zero.
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=None,
            algebraic=VectorSpace("z", 1),
            initial_conditions=lambda _z, _u, _p: (
                np.zeros(1),
                np.zeros(1),
            ),
            dynamics=lambda _t, _x, z, _u, _p: z,
            constraints=lambda _t, x, _z, _u, _p: x,
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        ),
        backend="sympy",
    )

    result = analyse(system)

    _assert_unsupported_analysis(result, result_type)
    reason = result.reason.lower()
    assert ("jacobian" in reason and "singular" in reason) or (
        "index" in reason and any(word in reason for word in ("one", "1", "higher"))
    )


@pytest.mark.parametrize(("analyse", "result_type"), _ANALYSES)
def test_analysis_reports_controlled_constraints_as_inconclusive(analyse, result_type):
    # dg/dz = 1 is regular, but z = x + u(t) requires input derivatives
    # outside the supported control-independent constraint geometry.
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
            constraints=lambda t, x, z, u, _p: z - x - u(t),
            outputs=lambda _t, _x, z, _u, _p, _q: z,
            quadratures=Noop(),
        ),
        backend="sympy",
    )

    result = analyse(system)

    _assert_unsupported_analysis(result, result_type)
    reason = result.reason.lower()
    assert "constraint" in reason
    assert "control" in reason or "input" in reason


@pytest.mark.parametrize(("analyse", "result_type"), _ANALYSES)
def test_analysis_reports_independent_residual_as_inconclusive(analyse, result_type):
    # Construct a genuinely independent residual, not a converted explicit
    # system whose legacy certificate could accidentally mask this boundary.
    system = ResidualDynamicalSystem(
        inputs=Noop(),
        parameters=None,
        x0=function(
            [None, Noop(), None],
            lambda _z, _u, _p: (np.ones(1), None),
            backend="numpy",
        ),
        F=function(
            [
                Scalar("t"),
                VectorSpace("w", 1),
                VectorSpace("wdot", 1),
                VectorSpace("z", 0),
                Noop(),
                None,
            ],
            lambda _t, w, wdot, _z, _u, _p: wdot * wdot * wdot + w,
            backend="numpy",
        ),
        y=function(
            [Scalar("t"), VectorSpace("x", 1), None, Noop(), None, None],
            lambda _t, x, _z, _u, _p, _q: x,
            backend="numpy",
        ),
        differential=Dimension(1),
        algebraic=None,
        quadrature=None,
    )

    result = analyse(system)

    _assert_unsupported_analysis(result, result_type)
    reason = result.reason.lower()
    # A reason can identify residuals directly or explain the accepted
    # symbolic/semi-explicit representations; it must identify this boundary.
    assert (
        "residual" in reason and ("support" in reason or "semi-explicit" in reason)
    ) or ("require" in reason and "dynamicalsystem" in reason and "symbolic" in reason)


def _assert_unsupported_analysis(result, result_type):
    assert isinstance(result, result_type)
    assert result.status is AnalysisStatus.INCONCLUSIVE
    assert result.rank == result.matrix.rank() == 0
    assert result.generators == ()
    assert result.rank_conditions == ()
    assert result.reason is not None
