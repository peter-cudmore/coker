"""Regression coverage for adaptive CasADi transcription reuse."""

import importlib.util

import numpy as np
import pytest
from coker import FunctionSpace, Scalar, VectorSpace
from coker.backends.casadi import CasadiVariationalOptions
from coker.dynamics import (
    BoundedVariable,
    ConstantControlVariable,
    TranscriptionOptions,
    VariationalProblem,
    VariationalProblemBuilder,
    create_autonomous_ode,
)
from coker.dynamics.system import create_control_system
from coker.toolkits.codesign import Minimise


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None,
    reason="CasADi not available",
)


def _slow_fast_problem(
    *,
    initialise_near_guess=True,
    enable_scaling=True,
    target_rate_offset=None,
):
    """Build a parameterized trajectory with local fast transient error."""

    def xdot(x, parameters):
        return (8.0 + 1e-3 * parameters[0]) * (np.ones((1,)) - x)

    system = create_autonomous_ode(
        parameters=VectorSpace("p", 1),
        x0=np.zeros((1,)),
        xdot=xdot,
        backend="casadi",
    )
    options = CasadiVariationalOptions(
        refinement_enabled=True,
        mesh_tolerance=5e-2,
        maximum_degree=12,
        maximum_iterations=8,
        initialise_near_guess=initialise_near_guess,
        enable_scaling=enable_scaling,
    )
    return VariationalProblem(
        loss=(
            (
                lambda solution, parameters: (
                    parameters[0] - target_rate_offset
                )
                ** 2
            )
            if target_rate_offset is not None
            else lambda solution, parameters: solution(1.0, parameters) ** 2
        ),
        system=system,
        parameters=[
            BoundedVariable(
                "rate_offset",
                lower_bound=0.0,
                upper_bound=1.0,
                guess=0.0,
            )
        ],
        t_final=1.0,
        transcription_options=TranscriptionOptions(
            minimum_n_intervals=3,
            minimum_degree=2,
            absolute_tolerance=1e-9,
            backend_options=options,
        ),
        backend="casadi",
    )


def _free_horizon_control_problem():
    control_space = FunctionSpace(
        "u",
        arguments=[Scalar("t")],
        output=[Scalar("u(t)")],
    )
    system = create_control_system(
        x0=np.zeros((1,)),
        xdot=lambda _t, x, _u, _parameters: 8.0 * (np.ones((1,)) - x),
        control=control_space,
        backend="numpy",
    )
    options = CasadiVariationalOptions(
        refinement_enabled=True,
        mesh_tolerance=5e-2,
        maximum_degree=12,
        maximum_iterations=8,
        initialise_near_guess=True,
        enable_scaling=False,
    )
    with VariationalProblemBuilder(
        system,
        t_final=BoundedVariable("duration", 0.5, 1.5, guess=0.5),
        control=[
            ConstantControlVariable("u", lower_bound=-1.0, upper_bound=1.0)
        ],
        transcription_options=TranscriptionOptions(
            minimum_n_intervals=3,
            minimum_degree=2,
            absolute_tolerance=1e-9,
            backend_options=options,
        ),
        backend="casadi",
    ) as builder:
        problem = builder.build(
            Minimise(
                (builder.t_final - 1.25) ** 2 + (builder.input(0) - 0.5) ** 2
            )
        )
    return problem, builder


def _mesh_signature(solution):
    return tuple(
        (tuple(poly.interval), poly.degree) for poly in solution.path.polys
    )


def _assert_refined(solution):
    degrees = [poly.degree for poly in solution.path.polys]
    assert solution.solve_info is not None
    assert solution.solve_info.success
    assert len(solution.path.polys) > 2 or max(degrees) > 2


def test_parameter_sweeps_keep_refined_mesh():
    problem = _slow_fast_problem()

    first = problem.get_solver("casadi").solve(rate_offset=0.0)
    _assert_refined(first)

    second = problem.get_solver("casadi").solve(rate_offset=0.5)
    _assert_refined(second)

    assert _mesh_signature(second) == _mesh_signature(first)


def test_adaptive_refinement_solves_from_default_guess():
    problem = _slow_fast_problem(
        initialise_near_guess=False,
        enable_scaling=False,
    )
    solution = problem.get_solver("casadi").solve(rate_offset=0.0)
    _assert_refined(solution)


def test_adaptive_refinement_solves_free_parameters():
    problem = _slow_fast_problem(
        initialise_near_guess=True,
        enable_scaling=False,
        target_rate_offset=0.75,
    )

    solution = problem.get_solver("casadi").solve()
    _assert_refined(solution)

    assert solution.parameters["rate_offset"] == pytest.approx(0.75)


def test_adaptive_refinement_solves_controls_and_free_horizon():
    problem, _builder = _free_horizon_control_problem()

    solution = problem.get_solver("casadi").solve()
    _assert_refined(solution)

    assert solution.t_final == pytest.approx(1.25)
    assert solution.control_law(0.0)[0] == pytest.approx(0.5)
