import pytest

from coker import FunctionSpace, Scalar, VectorSpace, function
from coker.dynamics import (
    BoundedVariable,
    PiecewiseConstantVariable,
    VariationalProblemBuilder,
)
from coker.dynamics.dynamical_system import create_control_system
from coker.toolkits.codesign import Minimise


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


def _build_with_constraints(constraints):
    system = make_parameterised_integrator()
    with VariationalProblemBuilder(system, t_final=1.0) as builder:
        problem = builder.build(
            Minimise(builder.output(builder.t)[0] ** 2),
            subject_to=constraints(builder),
        )
    return problem


def test_temporal_bindings_lower_to_path_initial_and_terminal():
    problem = _build_with_constraints(
        lambda b: [
            b.state(b.t)[0] <= 1,
            b.state(0)[0] == 0,
            b.state(b.t_final)[0] == 1,
            b.parameters()[0] >= 0,
        ]
    )

    assert len(problem.path_constraints) == 1
    assert len(problem.initial_constraints) == 1
    assert len(problem.terminal_constraints) == 2
    assert problem.path_constraints[0].temporal_binding == "path"
    assert problem.initial_constraints[0].temporal_binding == "initial"
    assert {c.temporal_binding for c in problem.terminal_constraints} == {
        "terminal"
    }


def test_endpoint_expression_in_path_constraint_is_broadcast():
    problem = _build_with_constraints(
        lambda b: [b.state(b.t)[0] <= b.state(b.t_final)[0]]
    )
    assert len(problem.path_constraints) == 1
    assert problem.path_constraints[0].temporal_binding == "path"


def test_endpoint_inputs_are_valid_symbolic_accessors():
    system = make_parameterised_integrator()
    with VariationalProblemBuilder(system, t_final=1.0) as builder:
        initial = builder.input(0)
        terminal = builder.input(builder.t_final)
        assert initial.tape is terminal.tape


def test_output_supports_path_constraints_and_terminal_objectives():
    system = make_parameterised_integrator()
    with VariationalProblemBuilder(system, t_final=1.0) as builder:
        path = builder.output(builder.t)[0] <= 1
        terminal_cost = builder.output(builder.t_final)[0] ** 2
        problem = builder.build(Minimise(terminal_cost), subject_to=[path])
    assert len(problem.path_constraints) == 1


def test_unsupported_concrete_time_is_rejected():
    system = make_parameterised_integrator()
    with VariationalProblemBuilder(system, t_final=1.0) as builder:
        with pytest.raises(ValueError, match="allowed bindings"):
            builder.state(0.5)
