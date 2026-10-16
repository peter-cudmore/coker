import numpy as np
import pytest

from coker import VectorSpace
from coker.dynamics import (
    VariationalProblem,
    VariationalProblemBuilder,
)
from coker.dynamics import create_autonomous_ode
from coker.toolkits.codesign import Minimise
from coker.backends.pytorch.variational import PytorchVariationalSolverOptions
from coker.algebra.ops import Noop
from coker.dynamics import DynamicsSpec
from coker.dynamics import create_dynamics_from_spec
from coker.parameters import BoundedVariable


def make_problem(*, guess=0.0, t_final=1.0):
    parameters = VectorSpace("p", 1)
    system = create_autonomous_ode(
        x0=1.0,
        xdot=lambda x, p: p[0] * x,
        parameters=parameters,
        backend="pytorch",
    )
    parameter = BoundedVariable("rate", -2.0, 2.0, guess=guess)
    target = float(np.exp(0.7 * t_final))
    return VariationalProblem(
        t_final=t_final,
        system=system,
        parameters=[parameter],
        backend="pytorch",
        loss=lambda solution, p: (solution(t_final, p) - target) ** 2,
    )


@pytest.mark.skip(
    reason="PyTorch trajectory interpolation is unstable; see "
    "https://github.com/peter-cudmore/coker/issues/103"
)
@pytest.mark.parametrize("horizon", [0.5, 1.0, 4.0])
def test_cuda_neural_ode_fit_and_solution(horizon):
    solution = make_problem(t_final=horizon).get_solver("pytorch")()
    assert solution.solve_info.success
    assert abs(solution.parameters["rate"] - 0.7) < 2e-3
    assert solution.cost < 1e-6
    for time in np.linspace(0, horizon, 9):
        expected = np.exp(0.7 * time)
        np.testing.assert_allclose(solution.state(time), expected, atol=2e-3)
        np.testing.assert_allclose(solution(time), expected, atol=2e-3)


def test_cuda_fixed_and_unknown_parameter_validation():
    solver = make_problem().get_solver("pytorch")
    fixed = solver.solve(rate=0.7)
    assert fixed.parameters["rate"] == pytest.approx(0.7)
    assert fixed.cost < 1e-6
    with pytest.raises(ValueError, match="Unknown variational parameter"):
        solver.solve(other=0.0)


def test_cuda_parameter_guess_rejects_outside_bounds():
    with pytest.raises(ValueError, match="outside their bounds"):
        make_problem(guess=3.0).get_solver("pytorch").solve()


def test_variational_options_follow_solver_option_contract():
    options = PytorchVariationalSolverOptions(warm_start=True)

    assert options.warm_start
    assert options.optimiser_method == "LBFGS"
    with pytest.raises(ValueError, match="Unsupported PyTorch optimiser"):
        PytorchVariationalSolverOptions(optimiser_method="SGD")


def test_cuda_integrates_registered_quadratures():
    parameters = VectorSpace("p", 1)
    system = create_autonomous_ode(
        x0=1.0,
        xdot=lambda x, p: p[0] * x,
        parameters=parameters,
        backend="pytorch",
    )
    rate = 0.7
    target = float(np.expm1(2 * rate) / (2 * rate))
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[BoundedVariable("rate", -2.0, 2.0)],
        backend="pytorch",
    ) as builder:
        integral = builder.integrate(builder.output(builder.t)[0] ** 2)
        problem = builder.build(Minimise((integral - target) ** 2))

    solution = problem.get_solver("pytorch").solve()

    assert solution.solve_info.success
    assert solution.parameters["rate"] == pytest.approx(rate, abs=2e-3)
    assert solution.quadratures(1.0)[0] == pytest.approx(target, abs=2e-3)


@pytest.mark.parametrize("horizon", [0.5, 1.0, 4.0])
def test_cuda_integrates_system_quadratures(horizon):
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=VectorSpace("p", 1),
            algebraic=None,
            initial_conditions=lambda _z, _u, p: (p * 0, None),
            dynamics=lambda _t, x, _z, _u, _p: x * 0,
            constraints=Noop(),
            outputs=lambda _t, _x, _z, _u, _p, q: q,
            quadratures=lambda _t, _x, _z, _u, p: p[0],
        ),
        backend="pytorch",
    )
    problem = VariationalProblem(
        t_final=horizon,
        system=system,
        parameters=[BoundedVariable("rate", -2.0, 2.0, guess=0.7)],
        backend="pytorch",
        loss=lambda solution, p: (solution(horizon, p) - 0.7 * horizon) ** 2,
    )

    solution = problem.get_solver("pytorch").solve()

    assert solution.solve_info.success
    for time in np.linspace(0, horizon, 9):
        expected = 0.7 * time
        np.testing.assert_allclose(
            solution.quadratures(time), expected, atol=2e-3
        )
        np.testing.assert_allclose(solution(time), expected, atol=2e-3)
    raw = solution.as_raw()
    np.testing.assert_allclose(raw[2], 0.7 * raw[0], atol=2e-3)
