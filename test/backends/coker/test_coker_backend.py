import numpy as np
import pytest
from coker.backends.coker import *
from coker.backends.coker.core import create_opgraph
from coker import kernel, VectorSpace, Scalar, Dimension

A = np.array([
    [1, 2],
    [3, 4]
])
b = np.array([4, 5])

c = 2




def f_quadratic_layer(x):
    y = A @ x + b
    return y.T @ y + c


def test_input_layer():
    input_dimensions = [
        Dimension(None),  # Scalar
        Dimension((3,)),  # 3 vector
    ]
    input_layer = InputLayer()
    [input_layer.add_input(d) for d in input_dimensions]

    assert input_layer.dimension == 4

    scalar_proj = input_layer.get_projection(0)
    vector_proj = input_layer.get_projection(1)

    scalar_vector = np.array([1, 0, 0, 0])
    scalar_result = scalar_proj @ scalar_vector
    assert scalar_result[0] == 1

    vector_proj_matrix = vector_proj.toarray()
    assert vector_proj_matrix.shape == (3, 4)
    expected_matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0],  [0, 0, 0, 1]])

    assert np.allclose(vector_proj_matrix, expected_matrix)

    mapped_input = input_layer(
        0, np.array([1, 2, 3])
    )
    assert np.allclose(mapped_input, np.array([0,1,2,3]))


def test_scalar_weights():
    # Scalar input
    spec = MemorySpec(0, 1)

    w = BilinearWeights(spec, constant=dok_ndarray((1, ), {(0,): 1}))

    assert w.constant.keys == {(0,): 1}
    assert w.constant.toarray() == np.array([1])
    assert w(0) == 1
    assert w(1) == 1


def test_coker_graph():
    alpha = 3
    beta = 4
    f = kernel(
        arguments=[Scalar('x')], implementation=lambda x: alpha * x + beta * x * x
    )

    g = create_opgraph(f)

    assert len(g.layers) == 3, f"Expected 4 layers (in, compute, out), got {len(g.layers)}"

    result = f(1)

    assert result == g(1)
    dx = 2
    y, dy = g.push_forward(
        1,  # x = 1
        2   # dx = 2
    )
    assert y == result
    assert dy == (alpha + 2 * beta) * dx


@pytest.mark.skip
def test_coker_vector():
    A = np.array([
        [2, 0, 0,],
        [0, 0, -2],
        [0, 2, 0]
    ])
    b = np.array([1, 1, 1])

    test_functions = [
        (lambda x: A @ x + b, lambda x, dx: A @ dx) ,
        (lambda x: np.dot(A @ x, b), lambda x, dx: np.dot(A @ dx, b)),
        (lambda x: b.T @ np.cos(x), lambda x, dx: - b.T @ (np.sin(x) * dx)),
        (lambda x: x.T @ x, lambda x, dx: 2 * x.T @ dx)
    ]

    for i, (test_function, derivative) in enumerate(test_functions):

        f = kernel(
            arguments=[VectorSpace('x', 3)], implementation=test_function
        )

        g = create_opgraph(f)
        x_vec = np.array([1, 2, 3])
        result = f(x_vec)

        assert np.allclose(result, g(x_vec))
        dx = np.array([-1, -1, -1])
        y, dy = g.push_forward(x_vec, dx)
        assert np.allclose(y, result)


@pytest.mark.skip
def test_sparse_matrix_times_weights():

    # create a weights; eg
    # T(x) = Q(x,x) + Ax + b
    #
    # create a sparse matrix
    # eg. c_hat = hat(c)
    #
    # make sure  c_hat @ T(x) makes sense
    #

    assert False
