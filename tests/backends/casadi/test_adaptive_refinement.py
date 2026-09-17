"""Behavioral coverage for the opt-in CasADi adaptive mesh refinement."""

import importlib.util

import numpy as np
import pytest
from coker.backends.casadi import CasadiVariationalOptions
from coker.dynamics import (
    TranscriptionOptions,
    VariationalProblem,
    create_autonomous_ode,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None,
    reason="CasADi not available",
)


def test_legacy_casadi_options_are_forwarded():
    callback = object()
    transcription = TranscriptionOptions(
        verbose=True,
        optimiser_options={"ipopt.max_iter": 3},
        initialise_near_guess=False,
        enable_scaling=False,
        interation_callback=callback,
    )

    from coker.backends.casadi.variational.solver import _casadi_options

    options = _casadi_options(
        type("Problem", (), {"transcription_options": transcription})()
    )

    assert options.verbose
    assert options.optimiser_options == {"ipopt.max_iter": 3}
    assert not options.initialise_near_guess
    assert not options.enable_scaling
    assert options.interation_callback is callback


def _boundary_layer_problem(*, options: CasadiVariationalOptions):
    """Build a deliberately under-resolved, stable fast-mode trajectory."""
    system = create_autonomous_ode(
        x0=np.array([0.0]),
        xdot=lambda x, _parameters: 8.0 * (np.ones((1,)) - x),
        backend="casadi",
    )
    transcription = TranscriptionOptions(
        minimum_n_intervals=2,
        minimum_degree=2,
        absolute_tolerance=1e-9,
        backend_options=options,
    )
    problem = VariationalProblem(
        loss=lambda solution, _parameters: solution(1.0) ** 2,
        system=system,
        t_final=1.0,
        transcription_options=transcription,
        backend="casadi",
    )
    return problem


def _interval_starts(solution):
    return list(solution.path.interval_starts())


def test_casadi_adaptive_refinement_resolves_fast_mode():
    problem = _boundary_layer_problem(
        options=CasadiVariationalOptions(
            refinement_enabled=True,
            mesh_tolerance=5e-2,
            maximum_degree=2,
            maximum_iterations=5,
        )
    )
    solution = problem.get_solver("casadi").solve()

    assert solution.solve_info is not None
    assert solution.solve_info.success
    # The exact solution is 1 - exp(-8 t); refinement must resolve its
    samples = np.linspace(0.0, 1.0, 101)
    expected = 1.0 - np.exp(-8.0 * samples)
    actual = np.array([float(solution.state(float(t))[0]) for t in samples])
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-3)

    assert len(_interval_starts(solution)) > 2


def test_casadi_refinement_disabled_preserves_initial_mesh():
    problem = _boundary_layer_problem(options=CasadiVariationalOptions())

    solution = problem.get_solver("casadi").solve()

    assert solution.solve_info is not None
    assert solution.solve_info.success
    assert len(_interval_starts(solution)) == 2
    assert np.isfinite(solution.cost)
    np.testing.assert_allclose(solution.state(0.0), [0.0], atol=1e-7)
