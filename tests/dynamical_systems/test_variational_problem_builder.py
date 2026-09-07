import pytest

from coker import FunctionSpace, Scalar, VectorSpace, function
from coker.dynamics import (
    BoundedVariable,
    PiecewiseConstantVariable,
    VariationalProblemBuilder,
)
from coker.dynamics.dynamical_system import create_control_system


def make_parameterised_integrator():
    control = FunctionSpace(
        "u",
        arguments=[Scalar("t")],
        output=[Scalar("u(t)")],
    )
    return create_control_system(
        parameters=VectorSpace("p", 1),
        control=control,
        x0=lambda p: p,
        xdot=lambda _t, x, _u, _p: 0 * x,
        backend="numpy",
    )


def test_variational_problem_builder_collects_problem_terms():
    system = make_parameterised_integrator()
    constraint = (
        function(
            system.y.input_spaces(),
            lambda _t, _x, _z, _u, p, _q: p[0],
            backend="numpy",
        )
        >= 0
    )
    control = PiecewiseConstantVariable("u", sample_rate=4)
    parameter = BoundedVariable("initial_state", -1, 1, guess=0.25)

    with VariationalProblemBuilder(system, t_final=2.0) as builder:
        builder.minimise(
            lambda solution, input_law, p: solution(2.0, input_law, p) ** 2
        )
        builder.add_input(control)
        builder.add_parameter(parameter)
        builder.add_path_constraint(constraint)
        builder.add_terminal_constraint(constraint)
        problem = builder.build()

    assert problem.system is system
    assert problem.t_final == 2.0
    assert problem.control == [control]
    assert problem.parameters == [parameter]
    assert problem.path_constraints == [constraint]
    assert problem.terminal_constraints == [constraint]


def test_variational_problem_builder_requires_a_loss():
    builder = VariationalProblemBuilder(
        make_parameterised_integrator(), t_final=1.0
    )

    with pytest.raises(ValueError, match="requires a loss functional"):
        builder.build()
