"""Regression coverage for adaptive CasADi transcription reuse."""

import importlib.util

import numpy as np
import pytest
from coker import VectorSpace
from coker.backends import get_backend_by_name
from coker.backends.casadi import CasadiVariationalOptions
from coker.backends.casadi.variational import solver as solver_module
from coker.dynamics import (
    BoundedVariable,
    TranscriptionOptions,
    VariationalProblem,
    create_autonomous_ode,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None,
    reason="CasADi not available",
)


def _slow_fast_problem(*, initialise_near_guess=True, enable_scaling=True):
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
        loss=lambda solution, parameters: solution(1.0, parameters) ** 2,
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


def _mesh_signature(solution):
    return tuple(
        (tuple(poly.interval), poly.degree) for poly in solution.path.polys
    )


def _assert_refined(solution):
    degrees = [poly.degree for poly in solution.path.polys]
    assert solution.solve_info is not None
    assert solution.solve_info.success
    assert len(solution.path.polys) > 2 or max(degrees) > 2


def test_parameter_sweeps_reuse_compiled_exact_mesh(monkeypatch):
    problem = _slow_fast_problem()
    builds = []
    create_once = solver_module._create_solver

    def track_construction(*args, **kwargs):
        builds.append(None)
        return create_once(*args, **kwargs)

    monkeypatch.setattr(solver_module, "_create_solver", track_construction)
    solver = problem.get_solver("casadi")

    first = solver.solve(rate_offset=0.0)
    _assert_refined(first)
    builds_after_first_solve = len(builds)

    second = solver.solve(rate_offset=0.5)
    _assert_refined(second)

    assert _mesh_signature(second) == _mesh_signature(first)
    assert len(builds) == builds_after_first_solve


def test_adaptive_refinement_uses_previous_path_as_refined_guess(monkeypatch):
    problem = _slow_fast_problem(
        initialise_near_guess=False,
        enable_scaling=False,
    )
    create_once = solver_module._create_solver
    nlp_calls = []

    class NlpCallRecorder:
        def __init__(self, delegate):
            self.delegate = delegate
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append(kwargs)
            return self.delegate(*args, **kwargs)

        def stats(self):
            return self.delegate.stats()

    def track_construction(*args, **kwargs):
        constructed = create_once(*args, **kwargs)
        recorder = NlpCallRecorder(constructed._solver)
        constructed._solver = recorder
        nlp_calls.append(recorder)
        return constructed

    monkeypatch.setattr(solver_module, "_create_solver", track_construction)
    solution = problem.get_solver("casadi").solve(rate_offset=0.0)
    _assert_refined(solution)

    assert len(nlp_calls) > 1
    refined_initial_guess = np.asarray(nlp_calls[-1].calls[0]["x0"]).reshape(
        -1
    )
    # The single parameter is the final decision component. The remaining
    # values are the path block, which is all zero for the default guess.
    assert np.ptp(refined_initial_guess[:-1]) > 0.1
