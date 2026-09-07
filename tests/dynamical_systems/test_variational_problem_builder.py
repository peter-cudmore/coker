import numpy as np
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


def test_integrate_registers_scalar_channels_without_mutating_source():
    system = make_parameterised_integrator()
    original_dqdt = system.dqdt
    with VariationalProblemBuilder(
        system, t_final=2.0, backend="numpy"
    ) as builder:
        q_running = builder.integrate(builder.output(builder.t)[0] ** 2)
        q_constant = builder.integrate(builder.parameters()[0] ** 2)
        problem = builder.build(Minimise(q_running + 2 * q_constant))

    assert q_running.dim.is_scalar()
    assert q_constant.dim.is_scalar()
    assert system.dqdt is original_dqdt
    assert problem.system is not system
    assert len(problem.system.dqdt.output) == 1
    assert problem.system.dqdt.output_shape()[0].flat() == 2

    derivative = problem.system.dqdt(
        0.0, np.array([3.0]), None, lambda _t: np.array([0.0]), np.array([2.0])
    )
    np.testing.assert_allclose(derivative, [9.0, 4.0])


def test_integrate_rejects_vector_integrands_before_lowering():
    system = make_parameterised_integrator()
    with VariationalProblemBuilder(system, t_final=1.0) as builder:
        with pytest.raises(ValueError, match="integrand must be scalar"):
            builder.integrate(builder.state(builder.t))


def test_fixed_horizon_is_not_a_decision():
    system = make_parameterised_integrator()
    with VariationalProblemBuilder(system, t_final=2.0) as builder:
        problem = builder.build(Minimise(builder.state(builder.t)[0] ** 2))

    assert problem.final_time_map.is_fixed
    assert problem.final_time_map.value == 2.0
    assert problem.horizon_decision is None


def test_bounded_horizon_is_first_class_and_not_a_parameter():
    system = make_parameterised_integrator()
    horizon = BoundedVariable("T", 0.1, 10.0, guess=1.0)
    parameter = BoundedVariable("mass", 1.0, 10.0, guess=2.0)
    parameter_map = np.array([[1.0]])
    with VariationalProblemBuilder(
        system,
        t_final=horizon,
        parameters=[parameter],
        system_parameter_map=parameter_map,
    ) as builder:
        problem = builder.build(Minimise(builder.state(builder.t)[0] ** 2))

    assert problem.final_time_map.is_free
    assert problem.final_time_map.declaration is horizon
    assert problem.horizon_decision is horizon
    assert problem.parameters == [parameter]
    np.testing.assert_array_equal(problem.system_parameter_map, parameter_map)
    assert problem.decision_declarations == [horizon]


def test_free_horizon_retains_control_and_temporal_constraint_scopes():
    system = make_parameterised_integrator()
    horizon = BoundedVariable("T", 0.1, 10.0, guess=1.0)
    control = PiecewiseConstantVariable(
        "u", sample_rate=4, lower_bound=-1, upper_bound=1
    )
    with VariationalProblemBuilder(
        system, t_final=horizon, control=[control]
    ) as builder:
        terminal = builder.state(builder.t_final)[0] == 1
        path = builder.input(builder.t) <= 1
        problem = builder.build(
            Minimise(builder.state(builder.t_final)[0] ** 2),
            subject_to=[terminal, path],
        )

    assert problem.decision_declarations == [horizon, control]
    assert len(problem.terminal_constraints) == 1
    assert len(problem.path_constraints) == 1
    assert problem.terminal_constraints[0].temporal_binding == "terminal"
    assert problem.path_constraints[0].temporal_binding == "path"


@pytest.mark.parametrize("invalid", [0, -1, "T", None, True])
def test_horizon_declaration_must_be_positive_number_or_bounded_variable(
    invalid,
):
    with pytest.raises((TypeError, ValueError), match="t_final"):
        VariationalProblemBuilder(
            make_parameterised_integrator(), t_final=invalid
        )


def test_bounded_horizon_requires_positive_lower_bound():
    with pytest.raises(ValueError, match="t_final"):
        VariationalProblemBuilder(
            make_parameterised_integrator(),
            t_final=BoundedVariable("T", -1, 10, guess=1),
        )


def test_sysopt_decay_integral_matches_closed_form_quadrature():
    """The builder integral follows the decay quadrature contract."""
    decay = create_control_system(
        parameters=VectorSpace("p", 1),
        control=FunctionSpace(
            "u",
            arguments=[Scalar("t")],
            output=[Scalar("u(t)")],
        ),
        x0=lambda _p: np.array([1.0]),
        xdot=lambda _t, x, _u, p: -p[0] * x,
        backend="numpy",
    )
    horizon = 2.0
    rate = 0.7
    with VariationalProblemBuilder(
        decay,
        t_final=horizon,
        parameters=[BoundedVariable("a", 0.1, 2.0, guess=rate)],
        backend="numpy",
    ) as builder:
        running = builder.integrate(builder.output(builder.t)[0] ** 2)
        problem = builder.build(Minimise(running))

    times = np.linspace(0.0, horizon, 1001)
    values = np.exp(-rate * times)
    quadrature = np.array(
        [
            problem.system.dqdt(
                time,
                np.array([value]),
                None,
                lambda _time: np.array([0.0]),
                np.array([rate]),
            )[0]
            for time, value in zip(times, values)
        ]
    )
    expected = (1.0 - np.exp(-2.0 * rate * horizon)) / (2.0 * rate)
    np.testing.assert_allclose(
        np.trapezoid(quadrature, times), expected, rtol=2e-5
    )


def test_sysopt_codesign_free_horizon_keeps_decisions_and_scopes():
    """A free horizon retains bounded control and terminal/path contracts."""
    system = make_parameterised_integrator()
    horizon = BoundedVariable("T", 0.1, 4.0, guess=1.0)
    control = PiecewiseConstantVariable(
        "u", sample_rate=8, lower_bound=-1.0, upper_bound=1.0
    )
    with VariationalProblemBuilder(
        system, t_final=horizon, control=[control]
    ) as builder:
        energy = builder.integrate(builder.input(builder.t) ** 2)
        problem = builder.build(
            Minimise(energy + (builder.state(builder.t_final)[0] - 1) ** 2),
            subject_to=[
                builder.state(builder.t_final)[0] == 1,
                builder.input(builder.t) <= 1,
                builder.input(builder.t) >= -1,
            ],
        )

    assert problem.horizon_decision is horizon
    assert problem.horizon_decision.lower_bound == 0.1
    assert problem.horizon_decision.upper_bound == 4.0
    assert problem.horizon_decision.guess == 1.0
    assert problem.decision_declarations == [horizon, control]
    assert len(problem.terminal_constraints) == 1
    assert len(problem.path_constraints) == 2
    assert all(
        constraint.temporal_binding == "terminal"
        for constraint in problem.terminal_constraints
    )
    assert all(
        constraint.temporal_binding == "path"
        for constraint in problem.path_constraints
    )
