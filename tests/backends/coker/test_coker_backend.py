import numpy as np
from coker.backends.coker.core import create_opgraph
from coker.backends.coker.layers import InputLayer
from coker.backends.coker.weights import BilinearWeights, MemorySpec
from coker import function, VectorSpace, Scalar, Dimension
from coker.backends.coker.sparse_tensor import dok_ndarray

A = np.array([[1, 2], [3, 4]])
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
    expected_matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    assert np.allclose(vector_proj_matrix, expected_matrix)

    mapped_input = input_layer(0, np.array([1, 2, 3]))
    assert np.allclose(mapped_input, np.array([0, 1, 2, 3]))


def test_scalar_weights():
    # Scalar input
    spec = MemorySpec(0, 1)

    w = BilinearWeights(
        spec, constant=dok_ndarray((1, 1), {(0, 0): 1}), shape=(1, 1)
    )

    assert w.constant.keys == {(0, 0): 1}
    assert w.constant.toarray() == np.array([1])
    assert w(0) == 1
    assert w(1) == 1


def test_bilinear_numpy_projection_keeps_sparse_coefficient_tensors(
    monkeypatch,
):
    memory = MemorySpec(0, 128)
    weights = BilinearWeights.from_trusted_dok(
        memory,
        shape=(2,),
        constant=dok_ndarray((2,), {(0,): 1.0}),
        linear=dok_ndarray((2, 128), {(0, 7): 2.0}),
        quadratic=dok_ndarray(
            (2, 128, 128),
            {(0, 7, 11): 3.0},
        ),
    )
    dense_round_trips = []
    original_fromarray = dok_ndarray.fromarray

    def record_fromarray(value):
        dense_round_trips.append(value.shape)
        return original_fromarray(value)

    monkeypatch.setattr(dok_ndarray, "fromarray", record_fromarray)
    projected = weights.__rmatmul__(np.array([[1.0, 0.0]]))

    assert dense_round_trips == []
    assert projected.constant.keys == {(0,): 1.0}
    assert projected.linear.keys == {(0, 7): 2.0}
    assert projected.quadratic.keys == {(0, 7, 11): 3.0}


def test_bilinear_matrix_matrix_product_lowers_without_opaque_layer():
    def implementation(x):
        left = np.reshape(x, (2, 2))
        right = np.reshape(x + 1.0, (2, 2))
        return left @ right

    f = function([VectorSpace("x", 4)], implementation)
    graph = create_opgraph(f)

    assert not any(
        getattr(layer, "opaque_programs", ()) for layer in graph.layers
    )
    value = np.array([0.2, -0.5, 1.1, 0.7])
    assert np.allclose(graph(value), f(value))


def test_bilinear_matrix_times_constant_vector_contracts_output_axis():
    memory = MemorySpec(0, 16)
    weights = BilinearWeights.from_trusted_dok(
        memory,
        shape=(2, 3),
        constant=dok_ndarray(
            (2, 3),
            {(0, 1): 2.0, (1, 2): 3.0},
        ),
        linear=dok_ndarray((2, 3, 16), {(0, 1, 7): 4.0}),
        quadratic=dok_ndarray(
            (2, 3, 16, 16),
            {(1, 2, 7, 11): 6.0},
        ),
    )

    projected = weights @ np.array([10.0, 5.0, 2.0])

    assert projected.shape == (2,)
    assert projected.constant.keys == {(0,): 10.0, (1,): 6.0}
    assert projected.linear.keys == {(0, 7): 20.0}
    assert projected.quadratic.keys == {(1, 7, 11): 12.0}


def test_graph_lowering_extends_only_live_bilinear_values(monkeypatch):
    def implementation(x):
        value = x
        for _ in range(20):
            value = np.sin(2.0 * value)
        return value

    symbolic_function = function(
        [VectorSpace("x", 4)],
        implementation=implementation,
    )
    extension_count = 0
    original_extend_memory = BilinearWeights.extend_memory

    def count_extend_memory(value, memory):
        nonlocal extension_count
        extension_count += 1
        return original_extend_memory(value, memory)

    monkeypatch.setattr(
        BilinearWeights,
        "extend_memory",
        count_extend_memory,
    )
    create_opgraph(symbolic_function)

    assert extension_count <= 2 * 20


def test_coker_graph():
    alpha = 3
    beta = 4
    f = function(
        arguments=[Scalar("x")],
        implementation=lambda x: alpha * x + beta * x * x,
    )

    g = create_opgraph(f)

    assert (
        len(g.layers) == 3
    ), f"Expected 3 layers (in, compute, out), got {len(g.layers)}"

    result = f(1)

    assert result == g(1)
    dx = 2
    y, dy = g.push_forward(1, 2)  # x = 1  # dx = 2
    assert y == result
    assert dy == (alpha + 2 * beta) * dx


def test_coker_vector():
    A = np.array(
        [
            [
                2,
                0,
                0,
            ],
            [0, 0, -2],
            [0, 2, 0],
        ]
    )
    b = np.array([1, 1, 1])

    test_functions = [
        (lambda x: A @ x + b, lambda x, dx: A @ dx),
        (lambda x: np.dot(A @ x, b), lambda x, dx: np.dot(A @ dx, b)),
        (lambda x: b + np.cos(x), lambda x, dx: -np.sin(x) * dx),
        (lambda x: np.cos(x), lambda x, dx: -np.sin(x) * dx),
        (
            lambda x: np.dot(b, np.cos(x)),
            lambda x, dx: -b.T @ (np.sin(x) * dx),
        ),
    ]

    for i, (test_function, derivative) in enumerate(test_functions):

        f = function(
            arguments=[VectorSpace("x", 3)], implementation=test_function
        )

        g = create_opgraph(f)

        x_vec = np.array([1, 2, 3])
        result = f(x_vec)

        assert np.allclose(result, g(x_vec))
        dx = np.array([-1, -1, -1])
        y, dy = g.push_forward(x_vec, dx)
        assert np.allclose(y, result)


def test_sin_cos():

    def f_impl(x):
        return np.sin(x[0]) + np.cos(x[1])

    f = function([VectorSpace("x", 2)], f_impl)
    g = create_opgraph(f)

    arg = np.array([1, 2])

    assert abs(f(arg) - g(arg)) < 1e-5


def test_dot_derivative():

    f, df = lambda x: np.dot(x, x), lambda x, dx: 2 * x.T @ dx

    f_function = function([VectorSpace("x", 3)], f)

    g = create_opgraph(f_function)

    assert len(g.intermediate_layers) == 1

    test_values = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 1]),
    ]
    tangent_vector = np.array([0, 1, 0])
    for test_value in test_values:
        f_v = f(test_value)
        df_v = df(test_value, tangent_vector)
        g_v, dg_v = g.push_forward(test_value, tangent_vector)

        assert np.allclose(f_v, g_v)
        assert np.allclose(df_v, dg_v)
