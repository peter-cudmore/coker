import numpy as np

from coker import (
    function,
    Scalar,
    VectorSpace,
    FunctionSpace,
    Dimension,
    get_projection,
    SymbolicVector,
)
from ..util import is_close


def test_symbolic_scalar(backend):

    def f_impl(x):
        return 2 * (x + 1)

    f = function(
        arguments=[Scalar("x")],
        implementation=f_impl,
        backend=backend,
    )

    assert f(1) == 4


def test_symbolic_vector_projection(backend):
    A = np.array([[0, 1, 0]], dtype=float)

    def f_impl(x):
        ax = A @ x
        return ax

    f = function([VectorSpace("x", 3)], f_impl, backend=backend)
    assert f.output[0].shape == (1,)

    x_test = np.array([2, 3, 5], dtype=float)
    y_test = f_impl(x_test)
    y_result = f(x_test)
    assert y_result.shape == y_test.shape
    assert is_close(y_result, y_test)


def test_symbolic_vector_matrix_product(backend):

    A = np.array([[0, 1], [-1, 0]], dtype=float)
    b = np.array([-1, 1], dtype=float)
    x_test = np.array([1, 1], dtype=float)
    y_result = np.array([0, 0], dtype=float)

    def f_impl(x):
        ax = A @ x
        return ax + b

    y_test = f_impl(x_test)

    assert is_close(y_result, y_test)

    f = function([VectorSpace(name="x", dimension=2)], f_impl, backend)

    y_eval = f(x_test)

    assert is_close(y_test, y_eval)


def test_componentwise_operations(backend):

    v = np.array([1, 2, 3])

    def f_divide_impl(x):
        return x / v

    def f_mul_impl(x):
        return x * v

    f_divide = function(
        [VectorSpace(name="x", dimension=3)], f_divide_impl, backend
    )
    f_mul = function([VectorSpace(name="x", dimension=3)], f_mul_impl, backend)

    x_test = np.array([2, 3, 5], dtype=float)

    y_divide = f_divide(x_test)
    y_divide_expected = f_divide_impl(x_test)
    y_mul = f_mul(x_test)
    y_mul_expected = f_mul_impl(x_test)

    assert is_close(y_divide, y_divide_expected, 1e-6)
    assert is_close(y_mul, y_mul_expected, 1e-6)


def test_slicing_symbolic_vector(backend):

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

    f = function(
        arguments=[VectorSpace(name="x", dimension=3)],
        implementation=f_impl,
        backend=backend,
    )

    assert f.output[0].shape == (2,)
    y_eval = f(x_test)

    assert is_close(y_test, y_eval)


def test_cross_product(backend):
    u_test = np.array([1, 0, 1], dtype=float)
    v_test = np.array([2, 1, 1], dtype=float)

    def f_impl(u, v):
        return np.cross(u, v)

    f_test = f_impl(u_test, v_test)

    f = function(
        arguments=[
            VectorSpace(name="x", dimension=3),
            VectorSpace(name="y", dimension=3),
        ],
        implementation=f_impl,
        backend=backend,
    )

    result = f(u_test, v_test)

    assert is_close(result, f_test)


def test_dot(backend):

    a = np.array([1, 0, 1], dtype=float)
    x_test = np.array([0, 0, 1], dtype=float)

    def f_dot(x):
        return np.dot(a, x)

    y_dot_test = f_dot(x_test)
    y_dot_result = function(
        [VectorSpace(name="x", dimension=3)], f_dot, backend
    )(x_test)
    assert np.allclose(y_dot_result, y_dot_test)


def test_dot_and_cross(backend):
    a = np.array([1, 0, 1], dtype=float)
    b = np.array([2, 1, 1], dtype=float)
    x_test = np.array([0, 0, 1], dtype=float)

    def f_impl(x):
        ax = np.cross(a, x)
        bax = np.dot(b, ax)
        return bax

    y_test = f_impl(x_test)
    f = function(
        arguments=[VectorSpace(name="x", dimension=3)],
        implementation=f_impl,
        backend=backend,
    )

    y = f(x_test)
    assert abs(y - y_test) < 1e-9


def test_build_array(backend):

    def f_impl(x):
        result = SymbolicVector((2,))
        result[0] = 1
        result[1] = x[0] + x[1]
        return result

    arg = np.array([1, 2], dtype=float)
    result = np.array([1, 3], dtype=float)
    # Should turn into
    # A = [0, 1][1, 0]^T  + [0, 1][0, 1]^T
    # ie (e_1 outer e_0) + (e_1 outer e_1)
    # A = [[0,0],[1, 1]], b = [1, 0]

    expected = f_impl(arg)
    assert is_close(expected, result)

    f = function(
        arguments=[VectorSpace(name="x", dimension=2)],
        implementation=f_impl,
        backend=backend,
    )

    output = f(arg)
    assert isinstance(output, np.ndarray)
    assert is_close(output, result)


def test_cos_and_sin(backend):

    def f_impl(x):
        return np.cos(x[0]) + np.sin(x[1])

    arg = np.array([0, 0], dtype=float)
    expected = f_impl(arg)
    f = function(
        arguments=[VectorSpace(name="x", dimension=2)],
        implementation=f_impl,
        backend=backend,
    )

    result = f(np.array([0, 0]))
    assert abs(result - expected) < 1e-4, f"Got {result}, expected {expected}"


def test_tensor_product(backend):

    a = np.array([[[1, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 1, 1]]], dtype=float)
    assert a.shape == (2, 2, 3)

    def f_impl(x):
        return a @ x

    x_test = np.array([2, 3, 4], dtype=float)

    b_test = f_impl(x_test)
    f = function(
        arguments=[VectorSpace(name="x", dimension=3)],
        implementation=f_impl,
        backend=backend,
    )

    b = f(x_test)
    assert is_close(b, b_test)


def test_functional(backend):

    def f_inner(A, b, x):
        return A @ x + b

    def f_outer(f, x):
        A = np.array([[0, 1], [1, 0]], dtype=float)
        b = np.array([0, 0], dtype=float)
        return f(A, b, x)

    f_result = f_outer(f_inner, np.array([2, 3], dtype=float))
    assert is_close(f_result, np.array([3, 2], dtype=float))

    f_coker = function(
        arguments=[
            FunctionSpace(
                name="f_inner",
                arguments=[
                    VectorSpace(name="A", dimension=(2, 2)),
                    VectorSpace(name="b", dimension=2),
                    VectorSpace(name="x", dimension=2),
                ],
                output=[VectorSpace(name="y", dimension=2)],
                signature=None,
            ),
            VectorSpace(name="x", dimension=2),
        ],
        implementation=f_outer,
        backend=backend,
    )

    f_coker_result = f_coker(f_inner, np.array([2, 3], dtype=float))
    assert is_close(f_coker_result, np.array([3, 2], dtype=float))


def test_zeros():
    from coker.algebra import zeros
    from coker.algebra.kernel import TraceContext

    with TraceContext():
        x = zeros((3, 3))

        assert x.shape == (3, 3)
        x[:, 0] = np.array([1, 2, 3])
        assert x[0, 0] == 1
