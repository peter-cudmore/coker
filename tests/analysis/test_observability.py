import numpy as np
import sympy as sp

from coker import FunctionSpace, Scalar, SymbolicVector, VectorSpace
from coker.dynamics import (
    AnalysisStatus,
    ObservabilityResult,
    analyse_observability,
    create_autonomous_ode,
)
from coker.dynamics.system import create_control_system


def _assert_rank_result(result, *, status, rank, required_rank):
    assert isinstance(result, ObservabilityResult)
    assert result.status is status
    assert result.rank == rank
    assert result.required_rank == required_rank
    assert isinstance(result.matrix, sp.MatrixBase)
    assert all(condition != 0 for condition in result.rank_conditions)
    assert result.reason is None


def test_observability_recovers_state_from_decay_output():
    system = create_autonomous_ode(
        x0=np.array([1.0]),
        xdot=lambda state, _parameters: -state,
        output=lambda state, _parameters: state,
        backend="sympy",
    )

    result = analyse_observability(system)

    _assert_rank_result(
        result,
        status=AnalysisStatus.TRUE,
        rank=1,
        required_rank=1,
    )


def test_observability_reports_unmeasured_decoupled_state():
    system = create_autonomous_ode(
        x0=np.ones(2),
        xdot=lambda state, _parameters: -state,
        output=lambda state, _parameters: state[0],
        backend="sympy",
    )

    result = analyse_observability(system)

    _assert_rank_result(
        result,
        status=AnalysisStatus.FALSE,
        rank=1,
        required_rank=2,
    )


def test_observability_limits_lie_derivative_order():
    system = create_autonomous_ode(
        x0=np.ones(2),
        xdot=lambda state, _parameters: SymbolicVector.from_list(
            [state[1], 0.0]
        ),
        output=lambda state, _parameters: state[0],
        backend="sympy",
    )

    result = analyse_observability(system, max_order=0)

    assert result.status is AnalysisStatus.INCONCLUSIVE
    assert result.rank == 1
    assert result.required_rank == 2
    assert result.reason


def test_observability_accepts_control_affine_systems():
    control = FunctionSpace(
        "u",
        arguments=[Scalar("time")],
        output=[VectorSpace("u", 1)],
    )
    system = create_control_system(
        x0=np.ones(2),
        control=control,
        xdot=lambda _time, state, value, _parameters: SymbolicVector.from_list(
            [value[0], state[0]]
        ),
        output=lambda _time, state, _value, _parameters: state[1],
        backend="sympy",
    )

    result = analyse_observability(system)

    _assert_rank_result(
        result,
        status=AnalysisStatus.TRUE,
        rank=2,
        required_rank=2,
    )


def test_observability_declines_non_affine_controls():
    control = FunctionSpace(
        "u",
        arguments=[Scalar("time")],
        output=[VectorSpace("u", 1)],
    )
    system = create_control_system(
        x0=np.ones(1),
        control=control,
        xdot=lambda _time, _state, value, _parameters: value * value,
        backend="sympy",
    )

    result = analyse_observability(system)

    assert result.status is AnalysisStatus.INCONCLUSIVE
    assert result.reason
