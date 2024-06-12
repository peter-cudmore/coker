import numpy as np
import pytest

from coker.backends.coker import *
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


def test_trace_depds():
    alpha = 3
    beta = 4
    f = kernel(
        arguments=[Scalar('x')], implementation=lambda x: alpha * x + beta * x * x
    )

    deps, quadtrics, nonlinears = trace_deps(f.tape, f.output)

    assert quadtrics == [{0,4}]
    assert not nonlinears

    assert len(deps[-1]) == 2


def test_edge_graph():
    alpha = 3
    beta = 4
    f = kernel(
        arguments=[Scalar('x')], implementation=lambda x: alpha * x + beta * x * x
    )
    nodes, edges = build_edge_graph(f)

    # graph should have 2 edge that corresponds to
    # y = a x + z_2
    # and z_1 = beta * x
    # graph should have one node
    # corresponding to z_2 = z_1 * x

    assert len(nodes) == 1
    assert len(edges) == 2

    # distance should have
    # input indicies :  2
    # node :            1
    # output indicies : 0

    f_tilde = rewrite_graph(f)

    nodes_2, edges_2, d = assign_layers_to_edge_graph(f_tilde)
    assert len(nodes_2) == 1
    assert len(edges_2) == 1


@pytest.mark.skip
def test_extract_layer_2():
    f_2 = kernel(arguments=[VectorSpace('x', 2)], implementation=f_quadratic_layer)
    # f_2 = x.T A.T @ A @ x + x.T @ A.T @ b + b.T @ A @ x + b.T @ A.T @ A @ b
    layers = extract_layers(f_2.tape, f_2.output)

    assert len(layers) == 1
    layer, = layers

    assert layer.input_dimension == 2
    assert layer.output_dimension == 1
    assert layer.constant.to_numpy() == np.dot(A @ b, A @ b) + c

    lin = 2 * b.reshape((1, 2)) @ A
    assert np.allclose(layer.linear.to_numpy(), lin)

    quad = A.T @ A
    assert np.allclose(layer.quadratic.to_numpy(), quad)

    assert layer.nonlinear_projections is None
    assert layer.nonlinear_function is None
    assert layer.nonlinear_inclusion is None
