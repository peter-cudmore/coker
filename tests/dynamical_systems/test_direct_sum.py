import numpy as np
from coker import FunctionSpace, Scalar, VectorSpace
from coker.dynamics import create_autonomous_ode, direct_sum
from coker.dynamics.dynamical_system import create_control_system
from ..util import is_close

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

    u1 = lambda t: np.array([1,0]) * np.cos(t)
    u2 = lambda t: np.array([0,1]) * np.sin(t)
    p1 = np.array([1, 2])
    p2 = np.array([3, 4])
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    def xdot(matrix, _, x, u, p):
        return matrix @ x + u + p

    system_1 = create_control_system(
        x0=lambda args: x1_0,
        xdot=lambda t,x,u,p: xdot(A, t, x, u, p),
        control=FunctionSpace('u', arguments=[Scalar('t')], output=[VectorSpace('u', 2)]),
        parameters=VectorSpace('p', 2),
        backend=variational_backend,
    )

    system_2 = create_control_system(
        x0=lambda args: x2_0,
        xdot=lambda t, x, u, p: xdot(B, t, x, u, p),
        control=FunctionSpace('u', arguments=[Scalar('t')], output=[VectorSpace('u', 2)]),
        parameters=VectorSpace('p', 2),
        backend=variational_backend,
    )

    system_3, projections = direct_sum(system_1, system_2)

    def u_3(t):
        u_3_inner = np.concatenate([u1(t), u2(t)])
        return u_3_inner

    def xdot_3(t, x, u, p):
        matrix = np.block([[A, np.zeros((2,2))],
                            [np.zeros((2,2)), B]])
        return matrix @ x + u + p
    p3 = np.concatenate([p1, p2]).reshape((4,))
    system_3_actual = create_control_system(
        x0=lambda args: np.concatenate([x1_0, x2_0]),
        xdot=xdot_3,
        control=FunctionSpace('u', arguments=[Scalar('t')], output=[VectorSpace('u', 4)]),
        parameters=VectorSpace('p', 4),
        backend=variational_backend,
    )

    for t_i in np.linspace(0, 1, 10):
        test = system_3(t_i, u_3, p3)
        truth = system_3_actual(t_i, u_3, p3)
        assert is_close(test, truth, tolerance=1e-3)

