import numpy as np
import pytest
from coker import FunctionSpace, Scalar, VectorSpace
from coker.dynamics import (
    DynamicsSpec,
    VariationalProblem,
    create_autonomous_ode,
    direct_sum,
)
from coker.parameters.function_parameters import FunctionParameter, RadialBasisFunction
from coker.algebra.ops import Noop
from coker.dynamics.system import (
    CompositionOperator,
    create_control_system,
    create_dynamics_from_spec,
)
from ..util import is_close
from coker.parameters import BoundedVariable


class _ConstantFunctionParameter(FunctionParameter):
    def __init__(self, name, guess):
        self.name = name
        self.guess = guess

    def validate_target(self, target):
        return target

    def list_concrete_parameters(self):
        return (BoundedVariable("constant", -1.0, 1.0, self.guess),)

    def evaluate(self, parameters, _argument):
        (value,) = parameters
        return value


def test_direct_sum_scalar(variational_backend):

    x1_0 = np.array([1])
    x2_0 = np.array([-1])

    def xdot(x, _):
        return -x

    def solution(t):
        return np.array([1, -1]) * np.exp(-t)

    system_1 = create_autonomous_ode(
        x0=x1_0, xdot=xdot, backend=variational_backend
    )

    system_2 = create_autonomous_ode(
        x0=x2_0, xdot=xdot, backend=variational_backend
    )

    system_3, projections = direct_sum(system_1, system_2)

    for t in np.linspace(0, 1, 10):
        assert is_close(system_3(t), solution(t), tolerance=1e-3)


def test_direct_sum_vector(variational_backend):
    x1_0 = np.array([1, 2])
    x2_0 = np.array([-1, 0])

    def u1(t):
        return np.array([1, 0]) * np.cos(t)

    def u2(t):
        return np.array([0, 1]) * np.sin(t)

    p1 = np.array([1, 2])
    p2 = np.array([3, 4])
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    def xdot(matrix, _, x, u, p):
        return matrix @ x + u + p

    system_1 = create_control_system(
        x0=lambda args: x1_0,
        xdot=lambda t, x, u, p: xdot(A, t, x, u, p),
        control=FunctionSpace(
            "u", arguments=[Scalar("t")], output=[VectorSpace("u", 2)]
        ),
        parameters=VectorSpace("p", 2),
        backend=variational_backend,
    )

    system_2 = create_control_system(
        x0=lambda args: x2_0,
        xdot=lambda t, x, u, p: xdot(B, t, x, u, p),
        control=FunctionSpace(
            "u", arguments=[Scalar("t")], output=[VectorSpace("u", 2)]
        ),
        parameters=VectorSpace("p", 2),
        backend=variational_backend,
    )

    system_3, projections = direct_sum(system_1, system_2)

    def u_3(t):
        u_3_inner = np.concatenate([u1(t), u2(t)])
        return u_3_inner

    def xdot_3(t, x, u, p):
        matrix = np.block([[A, np.zeros((2, 2))], [np.zeros((2, 2)), B]])
        return matrix @ x + u + p

    p3 = np.concatenate([p1, p2]).reshape((4,))
    system_3_actual = create_control_system(
        x0=lambda args: np.concatenate([x1_0, x2_0]),
        xdot=xdot_3,
        control=FunctionSpace(
            "u", arguments=[Scalar("t")], output=[VectorSpace("u", 4)]
        ),
        parameters=VectorSpace("p", 4),
        backend=variational_backend,
    )

    for t_i in np.linspace(0, 1, 10):
        test = system_3(t_i, u_3, p3)
        truth = system_3_actual(t_i, u_3, p3)
        assert is_close(test, truth, tolerance=1e-3)


def test_composition_operators():
    op = CompositionOperator.from_dimensions("x", (2,), (3,))
    mats = op.as_matrices()

    assert mats[0].shape == (2, 5)
    assert mats[1].shape == (3, 5)

    assert is_close(mats[0], np.hstack([np.eye(2), np.zeros((2, 3))]))
    assert is_close(mats[1], np.hstack([np.zeros((3, 2)), np.eye(3)]))

    test_array = np.array([1, 2, 3, 4, 5])
    result = op.inverse(test_array)
    assert is_close(result[0], np.array([1, 2]))
    assert is_close(result[1], np.array([3, 4, 5]))

    inverse = op(*result)
    assert is_close(inverse, test_array, tolerance=1e-9)


def test_direct_sum_reconstructs_function_parameter(variational_backend):
    rate = FunctionSpace(
        "rate", arguments=[Scalar("t")], output=[Scalar("rate")]
    )
    gain_system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(Scalar("gain"),),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda _t, _x, _z, _u, p: np.array([p[0]]),
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        ),
        backend=variational_backend,
    )
    rate_system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(rate,),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda t, _x, _z, _u, p: np.array([p[0](t)]),
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=lambda t, _x, _z, _u, p: p[0](t),
        ),
        backend=variational_backend,
    )
    system, _ = direct_sum(gain_system, rate_system)
    problem = VariationalProblem(
        loss=lambda solution, parameters: (
            (solution(1.0, parameters)[0] - 0.5) ** 2
            + (solution(1.0, parameters)[1] - 0.5) ** 2
        ),
        t_final=1.0,
        system=system,
        parameters=[
            BoundedVariable("gain", -1.0, 1.0),
            RadialBasisFunction([0.0], 1.0, name="rate"),
        ],
        backend=variational_backend,
    )

    solution = problem()

    fitted_rate = solution.parameters["rate"]
    assert callable(fitted_rate)
    assert np.isfinite(fitted_rate(0.25))
    assert np.isfinite(solution.quadratures(1.0)[0])
    np.testing.assert_allclose(solution.state(1.0), [0.5, 0.5], atol=1e-2)


def test_casadi_direct_sum_function_parameter_dae():
    pytest.importorskip("casadi")
    rate = FunctionSpace(
        "rate", arguments=[Scalar("t")], output=[Scalar("rate")]
    )
    gain_system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(Scalar("gain"),),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda _t, _x, _z, _u, p: np.array([p[0]]),
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        ),
        backend="casadi",
    )
    rate_system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(rate,),
            algebraic=VectorSpace("z", 1),
            initial_conditions=lambda _z, _u, p: (
                np.array([0.0]),
                np.array([p[0](0.0)]),
            ),
            dynamics=lambda _t, _x, z, _u, _p: z,
            constraints=lambda t, _x, z, _u, p: np.array([z[0] - p[0](t)]),
            outputs=lambda _t, x, _z, _u, _p, q: x + q,
            quadratures=lambda t, _x, _z, _u, p: np.array([p[0](t)]),
        ),
        backend="casadi",
    )
    system, _ = direct_sum(gain_system, rate_system)
    problem = VariationalProblem(
        loss=lambda solution, parameters: (
            (solution(1.0, parameters)[0] - 0.5) ** 2
            + (solution(1.0, parameters)[1] - 0.5) ** 2
        ),
        t_final=1.0,
        system=system,
        parameters=[
            BoundedVariable("gain", -1.0, 1.0),
            _ConstantFunctionParameter("rate", guess=0.25),
        ],
        backend="casadi",
    )

    solution = problem()

    assert solution.solve_info.success
    assert solution.cost < 1e-4
    assert solution.parameters["gain"] == pytest.approx(0.5, abs=1e-2)
    fitted_rate = solution.parameters["rate"]
    assert callable(fitted_rate)
    assert fitted_rate(0.25) == pytest.approx(0.25, abs=1e-2)
    np.testing.assert_allclose(solution.state(1.0), [0.5, 0.25], atol=1e-2)
    np.testing.assert_allclose(solution.algebraic(0.25), [0.25], atol=1e-2)
    np.testing.assert_allclose(solution.quadratures(1.0), [0.25], atol=1e-2)
