import numpy as np

from coker import VectorSpace

from coker.dynamics import (
    BoundedVariable,
    VariationalProblem,
    VariationalSolution,
    create_autonomous_ode,
)


class LossCheckingCallback:
    def __init__(self, sample_times, solution, param):
        self.sample_times = sample_times
        self.solution = solution
        self.param = param
        self.loss_differences = []
        self.losses = []

    def __call__(self, iterate: int, solution: VariationalSolution, **_kwargs):

        assert solution.algebraic(0) is None
        assert solution.quadratures(0) is None
        assert solution.control_law(0) is None
        assert solution.parameters.shape == self.param.shape
        assert solution.terminal_constraints().shape == (0,)
        assert solution.path_constraints(0.5).shape == (0,)

        recomputed_loss = 0.0
        for t_i in self.sample_times:
            y_i = np.atleast_1d(solution(t_i))
            x_i = np.atleast_1d(solution.state(t_i))
            truth = np.atleast_1d(self.solution(t_i, self.param))
            error = truth - y_i
            recomputed_loss += float(error.T @ error)
            assert np.allclose(x_i, y_i)

        self.losses.append(solution.cost)
        self.loss_differences.append(abs(solution.cost - recomputed_loss))
        return True


def test_variational_iteration_callback_loss_matches_payload(
    variational_backend,
):
    def x0(p):
        return p[0]

    def xdot(x, p):
        return p[1]

    param = np.array([2.0, 1.0])
    sample_times = np.arange(0.0, 1.0, 0.4)

    def solution(t, p):
        return p[0] + p[1] * t

    system = create_autonomous_ode(
        parameters=VectorSpace("p", 2),
        x0=x0,
        xdot=xdot,
        backend=variational_backend,
    )

    def loss(f, p_inner):
        total_error = 0.0
        for t_i in sample_times:
            total_error += (solution(t_i, param) - f(t_i, p_inner)) ** 2
        return total_error

    callback = LossCheckingCallback(sample_times, solution, param)

    problem = VariationalProblem(
        loss=loss,
        system=system,
        parameters=[
            BoundedVariable("offset", upper_bound=3, lower_bound=0.5, guess=2),
            float(param[1]),
        ],
        t_final=1,
        backend=variational_backend,
    )
    problem.transcription_options.interation_callback = callback

    solution_out = problem()

    assert solution_out.cost < 1e-6
    assert callback.loss_differences
    assert max(callback.loss_differences) < 1e-9


def test_variational_iteration_callback_stepwise_payload_loss(
    variational_backend,
):
    def x0(p):
        return p[0]

    def xdot(x, p):
        return p[1] * x

    param = np.array([1.5, -0.75])
    sample_times = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

    def solution(t, p):
        return p[0] * np.exp(p[1] * t)

    system = create_autonomous_ode(
        parameters=VectorSpace("p", 2),
        x0=x0,
        xdot=xdot,
        backend=variational_backend,
    )

    def loss(f, p_inner):
        total_error = 0.0
        for t_i in sample_times:
            total_error += (solution(t_i, param) - f(t_i, p_inner)) ** 2
        return total_error

    callback = LossCheckingCallback(sample_times, solution, param)

    problem = VariationalProblem(
        loss=loss,
        system=system,
        parameters=[
            BoundedVariable(
                "initial", upper_bound=3.0, lower_bound=0.1, guess=2.8
            ),
            BoundedVariable(
                "rate", upper_bound=1.0, lower_bound=-2.0, guess=0.8
            ),
        ],
        t_final=1,
        backend=variational_backend,
    )
    problem.transcription_options.initialise_near_guess = False
    problem.transcription_options.optimiser_options = {"ipopt.max_iter": 50}
    problem.transcription_options.interation_callback = callback

    solution_out = problem()

    assert solution_out.cost < 1e-6
    assert abs(solution_out.parameter_solutions["initial"] - param[0]) < 1e-4
    assert abs(solution_out.parameter_solutions["rate"] - param[1]) < 1e-4
    assert len(callback.losses) > 1
    assert max(callback.loss_differences) < 1e-9
