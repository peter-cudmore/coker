import numpy as np

from coker.toolkits.dynamical_systems import *

# Dynamics
# xdot = a x + u
# a < 0
# t in [0, T]

# solution is
# x(t) = x_0 exp(at) + \int_0^t \exp(a (t- \tau)) u(\tau) d\tau


def test_scalar_linear_system():

    def x0(u0, p):
        return p[1]

    def xdot(x, u, p):
        return p[0] * x + u

    def y(x, u, p):
        return x

    def u(t):
        return 2

    param = np.array([1, 2])

    def solution(t, p):
        x_0 = x0(2, p)
        a = p[0]
        x_t = x_0*np.exp(a * t)  + (np.exp(a * t) - 1) * 2 / a

        y = x_t
        return y


    system = create_homogenous_ode(
        inputs=Signal('u'),
        parameters=VectorSpace('p', 2),
        x0=x0,
        xdot=xdot,
        output=y,
        backend='numpy'
    )


    assert system.x0

    assert system.x0(0,  None, u, param) == 2
    arg_0 = (
        0, # t,
        1, # x
        None, #z,
        u(0),
        param
    )
    dxdt = system.dxdt(*arg_0)
    assert dxdt == 3

    t_final = 4
    soln = system(t_final, u, param)
    expected = solution(t_final, param)

    assert np.isclose(soln, expected)


def test_vector_linear_system():

    def x0(u0, p):

        return p[2:]

    def xdot(x, u, p):
        A =  np.array([[p[0], -1], [1, p[1]]])
        B = np.array([0, 1])
        return A @ x + B * u

    def y(x, u, p):
        return x

    def u(t):
        return 1

    param = np.array([1, 2, 0, 1])

    system = create_homogenous_ode(
        inputs=Signal('u'),
        parameters=VectorSpace('p', 4),
        x0=x0,
        xdot=xdot,
        output=y,
        backend='numpy'
    )

    assert np.all(system.x0(0,  None, u, param) == np.array([0, 1]))

    arg_0 = (
        0, # t,
        np.array([0, 1]), # x
        None, #z,
        u(0),
        param
    )
    dxdt = system.dxdt(*arg_0)
    assert np.all(dxdt == np.array([-1, 3]))

    t_final = 4
    soln = system(t_final, u, param)

    assert np.isfinite(soln).all()
    # Todo: solve this analytically and test the result


def test_fit():
    def x0(u0, p):
        return p[1]

    def xdot(x, u, p):
        return p[0] * x + u

    def y(x, u, p):
        return x

    def u(t):
        return 2

    param = np.array([1, 2])

    def solution(t, p):
        x_0 = x0(2, p)
        a = p[0]
        x_t = x_0 * np.exp(a * t) + (np.exp(a * t) - 1) * 2 / a

        y = x_t
        return y

    system = create_homogenous_ode(
        inputs=Signal('u'),
        parameters=VectorSpace('p', 2),
        x0=x0,
        xdot=xdot,
        output=y,
        backend='numpy'
    )
    n = 10
    samples = np.array(
        [(t, solution(t, param)) for t in np.linspace(0, 1, n)]
    )

    def loss(f):
        
        error = np.array(
            [
                f(t, u, param) - y
                for t, y in samples
            ]
        )
        loss = np.dot(error, error) / n

        return loss

    loss_value = loss(system)

    assert loss_value < 1e-6


