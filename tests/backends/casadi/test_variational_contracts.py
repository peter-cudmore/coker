"""Solve-level contracts for residual and registered-integral transcriptions."""

import numpy as np
import pytest

from coker import Dimension, FunctionSpace, Scalar, VectorSpace, function
from coker.backends.casadi import CasadiVariationalOptions
from coker.dynamics import (
    ConstantControlVariable,
    DynamicsSpec,
    ResidualDynamicalSystem,
    TranscriptionOptions,
    VariationalProblem,
    VariationalProblemBuilder,
    create_autonomous_ode,
    create_control_system,
    create_dynamics_from_spec,
)
from coker.dynamics.transcription.collocation import lgr_points
from coker.parameters import BoundedVariable
from coker.toolkits.codesign import Minimise


def test_derivative_tolerance_does_not_relax_certified_algebraics():
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=None,
            parameters=None,
            algebraic=VectorSpace("z", 1),
            initial_conditions=lambda z, _u, _p: (np.array([0.0]), z),
            dynamics=lambda _t, x, _z, _u, _p: 0 * x,
            constraints=lambda _t, _x, z, _u, _p: z,
            outputs=lambda _t, _x, z, _u, _p, _q: z,
            quadratures=None,
        ),
        backend="casadi",
    )
    node_time = float((lgr_points(3)[1] + 1) / 2)
    problem = VariationalProblem(
        system=system,
        loss=lambda trajectory, _p: -trajectory(node_time)[0],
        t_final=1.0,
        backend="casadi",
        transcription_options=TranscriptionOptions(
            minimum_n_intervals=1,
            minimum_degree=3,
            absolute_tolerance=1e-9,
            derivative_defect_tolerance=1e-3,
            backend_options=CasadiVariationalOptions(
                optimiser_options={
                    "ipopt.bound_relax_factor": 0.0,
                    "ipopt.tol": 1e-10,
                    "ipopt.constr_viol_tol": 1e-10,
                }
            ),
        ),
    )
    solution = problem()

    assert solution.solve_info.success
    assert abs(float(solution.algebraic(node_time)[0])) <= 2e-9


def test_free_horizon_registered_integral_uses_solved_duration():
    system = create_autonomous_ode(
        x0=np.array([0.0]), xdot=lambda x, _p: 0 * x, backend="casadi"
    )
    with VariationalProblemBuilder(
        system,
        t_final=BoundedVariable("duration", 0.5, 3.0, guess=0.75),
        backend="casadi",
    ) as builder:
        builder.integrate(builder.t_final)
        problem = builder.build(Minimise((builder.t_final - 2.0) ** 2))
    solution = problem()

    assert solution.solve_info.success
    assert solution.t_final == pytest.approx(2.0, abs=1e-5)
    np.testing.assert_allclose(
        solution.quadratures(solution.t_final), [solution.t_final**2], atol=1e-5
    )
    np.testing.assert_allclose(
        np.stack([segment.full_residual for segment in solution.segment_defects]),
        np.zeros((len(solution.path.polys), 2)),
        atol=1e-5,
    )


def test_registered_integral_evaluates_control_dependent_output():
    system = create_control_system(
        x0=np.array([0.0]),
        xdot=lambda _t, x, _u, _p: 0 * x,
        control=FunctionSpace("u", [Scalar("t")], [Scalar("value")]),
        output=lambda _t, _x, u, _p: np.array([u]),
        backend="casadi",
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.5,
        control=[ConstantControlVariable("u", lower_bound=2.0, upper_bound=2.0)],
        backend="casadi",
    ) as builder:
        builder.integrate(builder.output(builder.t)[0] ** 2)
        problem = builder.build(Minimise(builder.state(builder.t_final)[0] ** 2))
    solution = problem()

    assert solution.solve_info.success
    np.testing.assert_allclose(solution.quadratures(1.5), [6.0], atol=1e-5)
    np.testing.assert_allclose(
        np.stack([segment.full_residual for segment in solution.segment_defects]),
        np.zeros((len(solution.path.polys), 2)),
        atol=1e-5,
    )


@pytest.mark.parametrize("parameter_container", [list, tuple])
def test_residual_builder_accepts_equivalent_parameter_sequences(parameter_container):
    gain = Scalar("gain")
    system = ResidualDynamicalSystem(
        inputs=None,
        parameters=parameter_container([gain]),
        x0=function(
            [None, None, gain],
            lambda _z, _u, _gain: (np.array([0.0]), None),
            backend="casadi",
        ),
        F=function(
            [
                Scalar("t"),
                VectorSpace("w", 1),
                VectorSpace("wdot", 1),
                None,
                None,
                gain,
            ],
            lambda _t, _w, wdot, _z, _u, p: np.array([wdot[0] - p]),
            backend="casadi",
        ),
        y=function(
            [Scalar("t"), VectorSpace("x", 1), None, None, gain, None],
            lambda _t, x, _z, _u, _p, _q: x,
            backend="casadi",
        ),
        differential=Dimension(1),
        algebraic=None,
        quadrature=None,
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[BoundedVariable("gain", 0.1, 2.0, guess=0.3)],
        backend="casadi",
    ) as builder:
        problem = builder.build(Minimise((builder.output(builder.t_final)[0] - 1) ** 2))
    solution = problem()

    assert solution.solve_info.success
    assert solution.parameters["gain"] == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(solution.state(1.0), [1.0], atol=1e-4)
