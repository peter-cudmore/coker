import numpy as np

from coker import Scalar, Signal
from coker.toolkits.dynamical_systems import *

# Dynamics
# xdot = a x + u
# a < 0
# t in [0, T]

# solution is
# x(t) = x_0 exp(at) + \int_0^t \exp(a (t- \tau)) u(\tau) d\tau


def test_linear_system():

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