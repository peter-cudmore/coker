import numpy as np
import pytest

from coker.backends.coker.ast_preprocessing import *
from coker.backends.coker import *
from coker import kernel, VectorSpace, Scalar, Dimension

A = np.array([
    [1, 2],
    [3, 4]
])
b = np.array([4, 5])

c = 2


class TestSparseTensor:

    def test_convert(self):
        eye = np.eye(3, dtype=float)
        sparse_eye = dok_ndarray.fromarray(eye)
        assert sparse_eye.shape == (3,3)
        assert sparse_eye.keys == {(0,0):1.0, (1,1):1.0, (2,2):1.0}
        assert np.allclose(sparse_eye.toarray(), eye)

    def test_matmul(self):
        t = dok_ndarray(shape=(3, 3), data={(2,2): 1})
        t_test= np.array([[0,0,0], [0,0,0],[0,0, 1]])
        assert np.allclose(t.toarray(), t_test)

        b = np.array([1, 2, 3])

        y_expected = t_test @ b

        y = t @ b

        assert np.allclose(y.toarray(), y_expected)

        c = np.eye(3)

        id_1 = t @ c
        id_2 = c @ t

        id_1_array = id_1.toarray()
        assert np.allclose(id_1_array, id_2)


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

    assert len(g.layers) == 4, f"Expected 4 layers (in, compute, merge, out), got {len(g.layers)}"

    result = f(1)

    assert result == g(1)
    dx = 2
    y, dy = g.push_forward(
        1,  # x = 1
        2   # dx = 2
    )
    assert y == result
    assert dy == (alpha + 2 * beta) * dx


def test_coker_vector():

    A = np.array([
        [2, 0, 0,],
        [0, 0, -2],
        [0, 2, 0]
    ])
    b = np.array([1, 1, 1])
    f = kernel(
        arguments=[VectorSpace('x', 3)], implementation=lambda x: A @ x + b
    )

    g = create_opgraph(f)
    x_vec = np.array([1, 2, 3])
    result = f(x_vec)

    assert np.allclose(result, g(x_vec))
    dx = np.array([-1, -1, -1])
    y, dy = g.push_forward(x_vec, dx)
    assert np.allclose(y, result)
    assert np.allclose(dy,  A @ dx)

    assert len(g.layers) == 3



