import numpy as np

from coker import *
from .util import is_close


def test_symbolic_scalar():

    def f_impl(x):
        return 2 * (x + 1)

    f = kernel(
        arguments=[Scalar('x')],
        implementation=f_impl
      )

    assert f(1) == 4


def test_symbolic_vector():

    A = np.array([[0, 1], [-1, 0]], dtype=float)
    b = np.array([-1, 1], dtype=float)
    x_test = np.array([1, 1], dtype=float)
    y_result = np.array([0, 0], dtype=float)

    def f_impl(x):
        ax = A @ x
        return ax + b

    y_test = f_impl(x_test)

    assert is_close(y_result, y_test)

    f = kernel(
        arguments=[VectorSpace(name='x', dimension=2)],
        implementation=f_impl
    )

    y_eval = f(x_test)

    assert is_close(y_test, y_eval)


def test_slicing_symbolic_vector():

    matrix_dimensions = Dimension((3, 3))
    slc = slice(0, 2)

    projector = get_projection(matrix_dimensions, slc)

    assert projector.shape == (2, 3)

    A = np.array([[0, 1], [-1, 0]], dtype=float)
    b = np.array([-1, 1], dtype=float)
    x_test = np.array([1, 1, 1], dtype=float)
    y_result = np.array([0, 0], dtype=float)

    def f_impl(x):
        ax = A @ x[0:2]
        return ax + b

    y_test = f_impl(x_test)

    assert is_close(y_result, y_test)

    f = kernel(
        arguments=[VectorSpace(name='x', dimension=3)],
        implementation=f_impl
    )

    assert f.output.shape == (2,)
    y_eval = f(x_test)

    assert is_close(y_test, y_eval)


def test_dot_and_cross():

    a = np.array([1, 0, 1], dtype=float)
    b = np.array([2, 1, 1], dtype=float)
    x_test = np.array([0, 0, 1], dtype=float)

    def f_impl(x):
       ax = np.cross(a, x)
       bax = np.dot(b, ax)
       return bax

    y_test = f_impl(x_test)
    f = kernel(
        arguments=[VectorSpace(name='x', dimension=3)],
        implementation=f_impl
    )

    y = f(x_test)
    assert abs(y - y_test) < 1e-9


def test_build_array():

    def f_impl(x):
        result = Tensor((2,))
        result[0] = 1
        result[1] = x[0] + x[1]
        return result

    arg = np.array([1, 2], dtype=float)
    expected = f_impl(arg)
    assert is_close(expected, np.array([1, 3], dtype=float))

    f = kernel(
        arguments=[VectorSpace(name='x', dimension=2)],
        implementation=f_impl
    )

    result = f(arg)

    assert is_close(expected, result)


def test_cos_and_sin():

    def f_impl(x):
        y_0 = np.cos(x[0])
        y_1 = np.sin(x[1])
        return Tensor.from_list([y_0, y_1])

    expected = f_impl(np.array([0, 0]))
    assert expected.tolist() == [1, 0]

    f = kernel(
        arguments=[VectorSpace(name='x', dimension=2)],
        implementation=f_impl
    )

    result = f(np.array([0, 0]))

    assert is_close(result, expected)

