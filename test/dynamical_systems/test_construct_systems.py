import pytest
import numpy as np
from coker import FunctionSpace, Scalar
from coker.dynamics import *
from coker.dynamics.dynamical_system import create_control_system

# Dynamics
# xdot = a x + u
# a < 0
# t in [0, T]

# solution is
# x(t) = x_0 exp(at) + \int_0^t \exp(a (t- \tau)) u(\tau) d\tau

Signal = lambda name: FunctionSpace(
    name,
    arguments=[Scalar('t')],
    output=[Scalar('u(t)')],
)

def test_homogenous_integrator(variational_backend):

    x0 = np.array([1])

    def xdot(x, *args):
        return -x

    def solution(t):
        return np.exp(-t)

    system = create_homogenous_ode(
        x0=x0,
        xdot=xdot,
        backend=variational_backend
    )

    for t in np.linspace(0, 1, 10):
        assert np.isclose(solution(t), system(t), atol=1e-3)


def test_scalar_linear_system(variational_backend):
    def x0(p):
        x0_val = p[1:2]

        return x0_val

    def xdot(t, x, u, p):
        return p[0] * x + u


    def u_func(t):
        return 2

    param = np.array([1, 2])

    def solution(t, p):
        x_0 = x0(p)
        a = p[0]
        x_t = x_0 * np.exp(a * t) + (np.exp(a * t) - 1) * 2 / a

        y = x_t
        return y

    system = create_control_system(
        x0=x0,
        xdot=xdot,
        parameters=VectorSpace('p', 2),
        control=Signal('u')
    )

    assert system.x0
    assert system.x0(None, u_func, param)[0] == 2

    arg_0 = (0,  # t,
             1,  # x
             None,  # z,
             u_func, param)
    dxdt = system.dxdt(*arg_0)
    assert dxdt == 3

    t_final = 4
    soln = system(t_final, u_func, param)
    expected = solution(t_final, param)

    assert np.isclose(soln, expected)


def test_vector_linear_system(variational_backend):
    def x0(p):
        return p[2:]

    def xdot(t, x, u, p):
        A = np.array([[p[0], -1], [1, p[1]]])
        B = np.array([0, 1])
        return A @ x + B * u

    def y(t, x, u, p):
        return x

    def u(t):
        return 1

    param = np.array([1, 2, 0, 1])

    system = create_control_system(control=Signal('u'), parameters=VectorSpace('p', 4), x0=x0, xdot=xdot, output=y,
        backend=variational_backend)

    assert np.all(system.x0(0, u, param)[0] == np.array([0, 1]))

    arg_0 = (0,  # t,
             np.array([0, 1]),  # x
             None,  # z,
             u, param)
    dxdt = system.dxdt(*arg_0)
    assert np.all(dxdt == np.array([-1, 3]))

    t_final = 4
    soln = system(t_final, u, param)

    assert np.isfinite(soln).all()  # Todo: solve this analytically and test the result


def test_fitting():
    def x0(u0, p):
        return p[1]

    def xdot(x, u, p):
        return p[0] * x + u

    def y(x, u, p):
        return x

    def u(t):
        return 0

    param = np.array([-1, 2])

    def solution(t, p):
        x_0 = x0(u(0), p)
        a = p[0]
        x_t = x_0 * np.exp(a * t)

        y = x_t
        return y

    system = create_homogenous_ode(
        inputs=Signal('u'),
        parameters=VectorSpace('p', 2),
        x0=x0,
        xdot=xdot,
        output=y,
        backend='numpy')

    n = 10
    samples = np.array([(t, solution(t, param)) for t in np.linspace(0, 1, n)])

    def loss(f, *args):
        error = np.array([f(t, *args) - y for t, y in samples]).reshape((n,))
        return np.dot(error, error) / n


    loss_value = loss(system, u, param)

    assert loss_value < 1e-6

    decision_variables = (
        [ConstantControlVariable('control', upper_bound=4, lower_bound=0)],
        [BoundedVariable('rate', upper_bound=4, lower_bound=0.1 ), 2]
    )

    problem = VariationalProblem(
        loss=loss,
        system=system,
        arguments=decision_variables,
        t_final=1,
        constraints=[],
        backend='casadi'
    )

    min_cost, argmins = problem()
    assert abs(argmins['control'] - 2) < 1e-4
    assert abs(argmins['rate']  - 1) < 1e-4
