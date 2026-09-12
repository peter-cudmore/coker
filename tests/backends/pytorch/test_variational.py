import numpy as np
import pytest
import torch

from coker import VectorSpace
from coker.dynamics import BoundedVariable, VariationalProblem
from coker.dynamics.system import create_autonomous_ode
from coker.dynamics.variational.problem import QuadratureSpec


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for PyTorch variational solving",
)


def make_problem(*, guess=0.0):
    parameters = VectorSpace("p", 1)
    system = create_autonomous_ode(
        x0=1.0,
        xdot=lambda x, p: p[0] * x,
        parameters=parameters,
        backend="pytorch",
    )
    parameter = BoundedVariable("rate", -2.0, 2.0, guess=guess)
    target = float(np.exp(0.7))
    return VariationalProblem(
        t_final=1.0,
        system=system,
        parameters=[parameter],
        backend="pytorch",
        loss=lambda solution, p: (solution(1.0, p) - target) ** 2,
    )


def test_cuda_neural_ode_fit_and_solution():
    solution = make_problem().get_solver("pytorch")()
    assert solution.solve_info.success
    assert abs(solution.parameter_solutions["rate"] - 0.7) < 2e-3
    assert solution.cost < 1e-6
    assert abs(float(solution.state(1.0)[0]) - float(np.exp(0.7))) < 2e-3


def test_cuda_fixed_and_unknown_parameter_validation():
    solver = make_problem().get_solver("pytorch")
    fixed = solver.solve(rate=0.7)
    assert fixed.parameter_solutions["rate"] == pytest.approx(0.7)
    assert fixed.cost < 1e-6
    with pytest.raises(ValueError, match="Unknown variational parameter"):
        solver.solve(other=0.0)


def test_cuda_parameter_guess_initializes_bound_transform():
    solver = make_problem(guess=0.7).get_solver("pytorch")
    parameter = solver._parameters[0]
    value = solver._parameter_value(solver._raw_guess(parameter), parameter)
    assert float(value.cpu()) == pytest.approx(0.7, abs=1e-6)

    with pytest.raises(ValueError, match="outside its bounds"):
        make_problem(guess=3.0).get_solver("pytorch").solve()


def test_cuda_rejects_quadratures_explicitly():
    problem = make_problem()
    problem.quadratures = [QuadratureSpec("q")]
    with pytest.raises(NotImplementedError, match="Quadratures"):
        problem.get_solver("pytorch")
