import numpy as np
import sympy as sp

from coker import FunctionSpace, Scalar, VectorSpace
from coker.analysis import AnalysisStatus, analyse_identifiability
from coker.dynamics import create_autonomous_ode
from coker.dynamics.system import create_control_system


def _assert_rank_result(result, *, status, rank, required_rank):
    assert result.status is status
    assert result.rank == rank
    assert result.required_rank == required_rank
    assert isinstance(result.matrix, sp.MatrixBase)
    assert all(condition != 0 for condition in result.generic_conditions)
    assert result.reason is None


def test_identifiability_recovers_decay_rate_with_unknown_initial_state():
    system = create_autonomous_ode(
        x0=np.array([1.0]),
        xdot=lambda state, rate: -rate[0] * state,
        parameters=VectorSpace("rate", 1),
        backend="sympy",
    )

    result = analyse_identifiability(system)

    _assert_rank_result(
        result,
        status=AnalysisStatus.IDENTIFIABLE,
        rank=2,
        required_rank=2,
    )


def test_identifiability_rejects_product_parameterisation():
    system = create_autonomous_ode(
        x0=np.array([1.0]),
        xdot=lambda state, parameters: -(parameters[0] * parameters[1])
        * state,
        parameters=VectorSpace("rate", 2),
        backend="sympy",
    )

    result = analyse_identifiability(system)

    _assert_rank_result(
        result,
        status=AnalysisStatus.NOT_IDENTIFIABLE,
        rank=2,
        required_rank=3,
    )


def test_identifiability_handles_multiple_outputs_and_parameters():
    system = create_autonomous_ode(
        x0=np.array([1.0, 1.0]),
        xdot=lambda state, rates: -rates * state,
        parameters=VectorSpace("rate", 2),
        backend="sympy",
    )

    result = analyse_identifiability(system)

    _assert_rank_result(
        result,
        status=AnalysisStatus.IDENTIFIABLE,
        rank=4,
        required_rank=4,
    )


def test_identifiability_reports_non_affine_control_as_inconclusive():
    control = FunctionSpace(
        "u",
        arguments=[Scalar("time")],
        output=[VectorSpace("u", 1)],
    )
    system = create_control_system(
        x0=np.array([1.0]),
        xdot=lambda _time, _state, value, _parameters: value * value,
        control=control,
        backend="sympy",
    )

    result = analyse_identifiability(system)

    assert result.status is AnalysisStatus.INCONCLUSIVE
    assert result.reason
