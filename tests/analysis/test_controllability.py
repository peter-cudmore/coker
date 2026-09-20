import numpy as np

from coker import FunctionSpace, Scalar, SymbolicVector, VectorSpace
from coker.dynamics import AnalysisStatus, analyse_controllability
from coker.dynamics.system import create_control_system


def _scalar_control_space():
    return FunctionSpace(
        "u",
        arguments=[Scalar("t")],
        output=[VectorSpace("u", 1)],
    )


def _bracket_generated_system():
    return create_control_system(
        x0=np.zeros(2),
        control=_scalar_control_space(),
        xdot=lambda _t, x, u, _p: SymbolicVector.from_list([u[0], x[0] ** 2]),
        backend="numpy",
    )


def test_controllability_uses_lie_brackets_beyond_control_field_rank():
    result = analyse_controllability(_bracket_generated_system())

    assert result.status is AnalysisStatus.ACCESSIBLE
    assert result.rank == 2
    assert result.required_rank == 2
    assert result.generic_conditions
    assert all(condition != 0 for condition in result.generic_conditions)


def test_controllability_reports_rank_deficient_control_system():
    system = create_control_system(
        x0=np.zeros(2),
        control=_scalar_control_space(),
        xdot=lambda _t, _x, u, _p: SymbolicVector.from_list([u[0], 0.0]),
        backend="numpy",
    )

    result = analyse_controllability(system)

    assert result.status is AnalysisStatus.NOT_ACCESSIBLE
    assert result.rank == 1
    assert result.required_rank == 2


def test_controllability_declines_non_affine_controls():
    system = create_control_system(
        x0=np.zeros(2),
        control=_scalar_control_space(),
        xdot=lambda _t, _x, u, _p: SymbolicVector.from_list([u[0] ** 2, 0.0]),
        backend="numpy",
    )

    result = analyse_controllability(system)

    assert result.status is AnalysisStatus.INCONCLUSIVE
    assert result.reason


def test_controllability_reports_when_max_order_prevents_conclusion():
    result = analyse_controllability(_bracket_generated_system(), max_order=0)

    assert result.status is AnalysisStatus.INCONCLUSIVE
    assert result.rank == 1
    assert result.required_rank == 2
    assert result.reason
