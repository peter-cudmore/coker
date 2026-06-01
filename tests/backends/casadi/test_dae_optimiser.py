from dataclasses import replace

import numpy as np
import pytest

from coker import Dimension, VectorSpace
from coker.algebra.kernel import Noop
from coker.backends.casadi.dae_optimiser import (
    COLLOCATION_SOLVER,
    DAE_OPTIMISER_SOLVER,
    VARIATIONAL_SOLVER_OPTION,
)
from coker.dynamics import create_autonomous_ode
from coker.dynamics.dynamical_system import create_dynamics_from_spec
from coker.dynamics.types import (
    BoundedVariable,
    DynamicsSpec,
    VariationalProblem,
)


try:
    import casadi  # noqa: F401

    casadi_available = True
except ImportError:
    casadi_available = False


def _solve(problem: VariationalProblem, mode=None):
    options = dict(problem.transcription_options.optimiser_options)
    if mode is None:
        options.pop(VARIATIONAL_SOLVER_OPTION, None)
    else:
        options[VARIATIONAL_SOLVER_OPTION] = mode
    configured_problem = replace(
        problem,
        transcription_options=replace(
            problem.transcription_options,
            optimiser_options=options,
        ),
    )
    return configured_problem()


def _ode_parameter_fit_problem():
    target = np.array([2.0, 1.0])

    def x0(p):
        return p[0]

    def xdot(x, p):
        return p[1]

    def truth(t, p):
        return p[0] + p[1] * t

    system = create_autonomous_ode(
        parameters=VectorSpace("p", 2),
        x0=x0,
        xdot=xdot,
        backend="numpy",
    )

    def loss(f, p_inner):
        return sum(
            (f(t_i, p_inner) - truth(t_i, target)) ** 2
            for t_i in [0.1, 0.4, 1.0]
        )

    problem = VariationalProblem(
        loss=loss,
        system=system,
        parameters=[
            BoundedVariable(
                "value", lower_bound=0.5, upper_bound=3.0, guess=1.5
            ),
            float(target[1]),
        ],
        t_final=1.0,
        backend="casadi",
    )
    return target, problem


@pytest.mark.skipif(not casadi_available, reason="CasAdi not available")
def test_dae_optimiser_matches_collocation_on_ode_parameter_fit():
    _target, problem = _ode_parameter_fit_problem()

    optimised = _solve(problem, DAE_OPTIMISER_SOLVER)
    collocation = _solve(problem, COLLOCATION_SOLVER)

    assert (
        abs(
            optimised.parameter_solutions["value"]
            - collocation.parameter_solutions["value"]
        )
        < 1e-5
    )
    assert abs(optimised.cost - collocation.cost) < 1e-8
    for t_i in np.linspace(0.0, 1.0, 5):
        assert np.allclose(optimised(t_i), collocation(t_i), atol=1e-5)


@pytest.mark.skipif(not casadi_available, reason="CasAdi not available")
def test_dae_optimiser_matches_collocation_on_dae_parameter_fit():
    target = np.array([2.0])

    def initial_conditions(_z, _u, p):
        _ = p
        return np.zeros((1,)), np.zeros((1,))

    def dynamics(t, x, z, u, p):
        _ = t, x, u
        return p[0:1] + 0.0 * z

    def constraints(t, x, z, u, p):
        _ = t, x, u, p
        return z

    def outputs(t, x, z, u, p, q):
        _ = t, z, u, p, q
        return x[0]

    z_space = VectorSpace("z", 1)
    z_space.dim = Dimension((1,))
    spec = DynamicsSpec(
        inputs=Noop(),
        parameters=VectorSpace("p", 1),
        algebraic=z_space,
        initial_conditions=initial_conditions,
        dynamics=dynamics,
        constraints=constraints,
        outputs=outputs,
        quadratures=Noop(),
    )
    system = create_dynamics_from_spec(spec, backend="numpy")

    def loss(f, p_inner):
        return sum(
            (f(t_i, p_inner) - target[0] * t_i) ** 2
            for t_i in [0.25, 0.5, 1.0]
        )

    problem = VariationalProblem(
        loss=loss,
        system=system,
        parameters=[
            BoundedVariable(
                "slope", lower_bound=0.5, upper_bound=3.0, guess=1.5
            ),
        ],
        t_final=1.0,
        backend="casadi",
    )

    optimised = _solve(problem, DAE_OPTIMISER_SOLVER)
    collocation = _solve(problem, COLLOCATION_SOLVER)

    assert (
        abs(
            optimised.parameter_solutions["slope"]
            - collocation.parameter_solutions["slope"]
        )
        < 1e-5
    )
    assert abs(optimised.cost - collocation.cost) < 1e-8
    for t_i in np.linspace(0.0, 1.0, 5):
        assert np.allclose(optimised(t_i), collocation(t_i), atol=1e-5)
        assert np.allclose(
            optimised.algebraic(t_i), collocation.algebraic(t_i), atol=1e-5
        )
