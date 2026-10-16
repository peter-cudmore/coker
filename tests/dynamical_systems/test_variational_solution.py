import numpy as np
import pytest

from coker import FunctionSpace, Scalar, VectorSpace, function
from coker.backends.casadi.variational.factory import _TranscriptionFactory
from coker.dynamics import (
    ConstantControlVariable,
    DynamicsSpec,
    VariationalProblem,
    VariationalProblemBuilder,
    VariationalSolution,
    create_autonomous_ode,
    create_control_system,
    create_dynamics_from_spec,
)
from coker.dynamics.transcription.collocation import (
    InterpolatingPoly,
    InterpolatingPolyCollection,
)
from coker.dynamics.variables import PiecewiseControlSolution
from coker.parameters import BoundedVariable
from coker.toolkits.codesign import Minimise


@pytest.mark.parametrize("horizon", [0.5, 1.0, 4.0])
def test_evaluate_decay_at_physical_times(horizon):
    system = create_autonomous_ode(
        x0=np.array([2.0]),
        xdot=lambda x, p: -p[0] * x,
        parameters=VectorSpace("p", 1),
        backend="casadi",
    )
    times = np.linspace(horizon / 8, horizon, 8)
    observations = 2.0 * np.exp(-0.7 * times)

    def loss(trajectory, parameters):
        return sum(
            (trajectory(float(time), parameters) - float(observed)) ** 2
            for time, observed in zip(times, observations)
        )

    fit = VariationalProblem(
        system=system,
        loss=loss,
        parameters=[BoundedVariable("rate", 0.05, 2.0, guess=0.3)],
        t_final=horizon,
        backend="casadi",
    )()

    assert fit.solve_info.success
    assert fit.parameters["rate"] == pytest.approx(0.7, abs=1e-4)
    # Endpoints, mesh boundaries, and non-knot points on both sides of t=1.
    for fraction in [0.0, 0.125, 0.25, 0.375, 0.5, 0.73, 0.75, 1.0]:
        time = fraction * horizon
        expected = 2.0 * np.exp(-0.7 * time)
        np.testing.assert_allclose(fit(time), expected, atol=1e-5)
        np.testing.assert_allclose(fit.state(time), expected, atol=1e-5)
    for time in [-0.01, horizon + 0.01]:
        with pytest.raises(ValueError, match="not in any interval"):
            fit(time)


@pytest.mark.parametrize("horizon", [0.5, 4.0])
@pytest.mark.parametrize("free_horizon", [False, True])
def test_evaluate_outputs_quadratures_and_constraints(horizon, free_horizon):
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=None,
            parameters=VectorSpace("p", 1),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda _t, x, _z, _u, _p: 0 * x + 1,
            constraints=None,
            outputs=lambda t, x, _z, _u, p, q: t + x + q + p[0],
            quadratures=lambda _t, x, _z, _u, _p: x,
        ),
        backend="casadi",
    )
    duration = (
        BoundedVariable(
            "duration", horizon / 2, horizon * 2, guess=0.75 * horizon
        )
        if free_horizon
        else horizon
    )
    with VariationalProblemBuilder(
        system, t_final=duration, parameters=[2.0], backend="casadi"
    ) as builder:
        problem = builder.build(Minimise((builder.t_final - horizon) ** 2))
    bound = function(
        system.y.input_spaces(),
        lambda t, x, _z, _u, _p, _q: t - x[0],
        backend="casadi",
    )
    fit = problem()
    # Exercise violation evaluation independently of constraint lowering.
    fit.path_constraint_exprs = [bound >= 0]
    fit.terminal_constraint_exprs = [bound >= 0]

    assert fit.solve_info.success
    assert fit.t_final == pytest.approx(horizon, abs=1e-5)
    output_poly = fit.to_poly()
    for fraction in [0.0, 0.25, 0.63, 0.75, 1.0]:
        time = fraction * fit.t_final
        np.testing.assert_allclose(fit.state(time), time, atol=1e-6)
        np.testing.assert_allclose(
            fit.quadratures(time), time**2 / 2, atol=1e-6
        )
        expected = 2 * time + time**2 / 2 + 2
        np.testing.assert_allclose(fit(time), expected, atol=1e-6)
        np.testing.assert_allclose(output_poly(time), expected, atol=1e-6)
        np.testing.assert_allclose(fit.path_constraints(time), 0, atol=1e-6)
    np.testing.assert_allclose(fit.terminal_constraints(), 0, atol=1e-6)
    raw = fit.as_raw()
    np.testing.assert_allclose(raw[1], raw[0], atol=1e-6)
    # Internal meshes and diagnostics remain normalized/physical respectively.
    assert fit.path.intervals[-1][1] == 1.0
    assert fit.segment_defects[-1].physical_interval[1] == fit.t_final
    for segment in fit.segment_defects:
        np.testing.assert_allclose(segment.state_residual, 0, atol=1e-6)
        np.testing.assert_allclose(segment.quadrature_residual, 0, atol=1e-6)


@pytest.mark.parametrize("horizon", [0.5, 4.0])
def test_evaluate_algebraics_and_controls_in_physical_time(horizon):
    path = InterpolatingPolyCollection(
        [
            InterpolatingPoly(
                2,
                (0.0, 1.0),
                1,
                np.array([[0.0], [0.0], [horizon], [2 * horizon]]),
            )
        ]
    )
    fit = VariationalSolution.from_solver(
        cost=0,
        path=path,
        projectors=(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), None),
        control_solutions=[
            PiecewiseControlSolution("u", 1.5, np.array([1.0, 2.0]))
        ],
        parameters={"p": 3.0},
        parameter_vector=np.array([3.0]),
        output=lambda t, x, z, u, p, _q: t + x + z + u + p[0],
        t_final=horizon,
    )
    for fraction, control in [(0.0, 1), (0.625, 1), (0.75, 2), (1.0, 2)]:
        time = fraction * horizon
        np.testing.assert_allclose(fit.control_law(time), control)
        np.testing.assert_allclose(fit.algebraic(time), 2 * time)
        np.testing.assert_allclose(fit(time), 4 * time + control + 3)
    system = create_control_system(
        x0=np.array([0.0]),
        xdot=lambda t, _x, u, p: t + u + p[0],
        control=FunctionSpace("u", [Scalar("t")], [VectorSpace("u", 1)]),
        parameters=VectorSpace("p", 1),
        backend="casadi",
    )
    factory = _TranscriptionFactory(
        VariationalProblem(
            system=system,
            loss=lambda trajectory, _u, p: trajectory(horizon, _u, p)[0],
            parameters=[3.0],
            control=[ConstantControlVariable("u")],
            t_final=horizon,
            backend="casadi",
        )
    )
    rates, _ = factory.evaluate_defect_rates(
        np.array([0.0, 0.625, 0.75, 1.0]), np.zeros((1, 4)), fit
    )
    # Defect nodes are normalized, but controls and dynamics receive seconds.
    np.testing.assert_allclose(
        rates, [[4.0, 4 + 0.625 * horizon, 5 + 0.75 * horizon, 5 + horizon]]
    )
