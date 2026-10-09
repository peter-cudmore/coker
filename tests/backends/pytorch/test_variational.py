import numpy as np
import pytest

from coker import VectorSpace
from coker.dynamics import (
    VariationalProblem,
    VariationalProblemBuilder,
)
from coker.dynamics import create_autonomous_ode
from coker.toolkits.codesign import Minimise
from coker.backends.pytorch.variational import (
    PytorchVariationalSolverOptions,
    create_variational_solver,
)
from coker.algebra.ops import Noop
from coker.dynamics.model import DynamicsSpec
from coker.dynamics.system import create_dynamics_from_spec
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


@pytest.fixture(
    params=[
        None,
        PytorchVariationalSolverOptions(
            trajectory_degree=6, trajectory_n_intervals=8
        ),
        PytorchVariationalSolverOptions(
            trajectory_degree=10, trajectory_n_intervals=3
        ),
    ]
)
def trajectory_options(request):
    return request.param


@pytest.mark.parametrize("horizon", [0.5, 1.0, 4.0])
def test_interpolate_exact_exponential_samples(horizon, trajectory_options):
    solver = create_variational_solver(
        make_problem(t_final=horizon), trajectory_options
    )
    times = solver._trajectory_times()

    def exact(time):
        return np.array(
            [
                np.exp(0.7 * time),
                np.expm1(0.7 * time) / 0.7,
                np.expm1(1.4 * time) / 1.4,
            ]
        ).T

    path = solver._interpolate_trajectory(exact(times))
    boundaries = [path.intervals[0][0]] + [end for _, end in path.intervals]
    near_boundaries = [
        boundary + offset
        for boundary in boundaries
        for offset in (-1e-7, 0.0, 1e-7)
        if 0 <= boundary + offset <= 1
    ]
    queries = np.unique(np.r_[np.linspace(0, 1, 1001), near_boundaries])
    actual = np.array([path(time) for time in queries])
    np.testing.assert_allclose(
        actual, exact(queries * horizon), rtol=1e-7, atol=1e-8
    )
    for left, right in zip(path.polys[:-1], path.polys[1:]):
        np.testing.assert_array_equal(
            left.end_point()[1], right.start_point()[1]
        )


@pytest.mark.parametrize("horizon", [0.5, 1.0, 4.0])
def test_cuda_neural_ode_fit_and_solution(horizon):
    solution = make_problem(t_final=horizon).get_solver("pytorch")()
    assert solution.solve_info.success
    assert abs(solution.parameters["rate"] - 0.7) < 2e-3
    assert solution.cost < 1e-6
    output_poly = solution.to_poly()
    for time in np.linspace(0, horizon, 9):
        expected = np.exp(0.7 * time)
        np.testing.assert_allclose(solution.state(time), expected, atol=2e-3)
        np.testing.assert_allclose(solution(time), expected, atol=2e-3)
        np.testing.assert_allclose(output_poly(time), expected, atol=2e-3)


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


@pytest.mark.parametrize(
    "degree, intervals",
    [(0, 4), (-1, 4), (1.5, 4), (8, 0), (8, -1), (8, 1.5)],
)
def test_reject_invalid_trajectory_grid(degree, intervals):
    with pytest.raises(ValueError, match="must be a positive integer"):
        PytorchVariationalSolverOptions(
            trajectory_degree=degree, trajectory_n_intervals=intervals
        )


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
@pytest.mark.parametrize(
    "options",
    [
        None,
        PytorchVariationalSolverOptions(
            trajectory_degree=1, trajectory_n_intervals=1
        ),
    ],
)
def test_cuda_integrates_system_quadratures(horizon, options):
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

    solution = create_variational_solver(problem, options).solve()

    assert solution.solve_info.success
    for time in np.linspace(0, horizon, 9):
        expected = 0.7 * time
        np.testing.assert_allclose(
            solution.quadratures(time), expected, atol=2e-3
        )
        np.testing.assert_allclose(solution(time), expected, atol=2e-3)
    raw = solution.as_raw()
    np.testing.assert_allclose(raw[2], 0.7 * raw[0], atol=2e-3)


@pytest.mark.parametrize("horizon", [0.5, 1.0, 4.0])
def test_evaluate_piecewise_state_and_quadratures(horizon, trajectory_options):
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=VectorSpace("p", 1),
            algebraic=None,
            initial_conditions=lambda _z, _u, p: (1 + 0 * p, None),
            dynamics=lambda _t, x, _z, _u, p: p[0] * x,
            constraints=Noop(),
            outputs=lambda t, x, _z, _u, _p, _q: x + t,
            quadratures=lambda _t, x, _z, _u, _p: x,
        ),
        backend="pytorch",
    )
    with VariationalProblemBuilder(
        system,
        t_final=horizon,
        parameters=[BoundedVariable("rate", -2.0, 2.0)],
        backend="pytorch",
    ) as builder:
        integral = builder.integrate(builder.state(builder.t)[0] ** 2)
        problem = builder.build(Minimise(integral))

    solution = create_variational_solver(problem, trajectory_options).solve(
        rate=0.7
    )
    if trajectory_options is not None:
        assert (
            len(solution.path.polys)
            == trajectory_options.trajectory_n_intervals
        )
        assert all(
            poly.degree == trajectory_options.trajectory_degree
            for poly in solution.path.polys
        )
    boundaries = [solution.path.intervals[0][0]] + [
        end for _, end in solution.path.intervals
    ]
    output_poly = solution.to_poly()
    near_boundaries = [
        boundary + offset
        for boundary in boundaries
        for offset in (-1e-7, 0.0, 1e-7)
        if 0 <= boundary + offset <= 1
    ]
    fractions = np.unique(np.r_[np.linspace(0, 1, 1001), near_boundaries])
    for time in fractions * horizon:
        state = np.exp(0.7 * time)
        quadratures = [
            np.expm1(0.7 * time) / 0.7,
            np.expm1(1.4 * time) / 1.4,
        ]
        np.testing.assert_allclose(solution.state(time), state, atol=2e-3)
        np.testing.assert_allclose(
            solution.quadratures(time), quadratures, rtol=1e-5, atol=2e-3
        )
        np.testing.assert_allclose(solution(time), state + time, atol=2e-3)
        np.testing.assert_allclose(output_poly(time), state + time, atol=2e-3)

    assert solution.path.intervals[0][0] == 0
    assert solution.path.intervals[-1][1] == 1
    raw = solution.as_raw()
    assert raw[0, 0] == 0
    assert 0 < raw[0, -1] < horizon
    np.testing.assert_allclose(raw[1], np.exp(0.7 * raw[0]), atol=2e-3)
    np.testing.assert_allclose(
        raw[2], np.expm1(0.7 * raw[0]) / 0.7, rtol=1e-5, atol=2e-3
    )
    np.testing.assert_allclose(
        raw[3], np.expm1(1.4 * raw[0]) / 1.4, rtol=1e-5, atol=2e-3
    )
