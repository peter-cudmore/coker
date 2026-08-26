import numpy as np
from coker import Scalar, VectorSpace, function, if_then_else
from coker.backends.coker.lowering import create_function_table


def _assert_same(actual, expected):
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=False):
            _assert_same(actual_item, expected_item)
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert np.allclose(actual, expected)
        return
    assert np.allclose(actual, expected)


def _finite_difference(implementation, args, tangents):
    """Differentiate the source NumPy implementation without bytecode."""
    h = 1e-6
    plus = implementation(
        *tuple(np.asarray(value) + h * np.asarray(tangent)
               for value, tangent in zip(args, tangents, strict=True))
    )
    minus = implementation(
        *tuple(np.asarray(value) - h * np.asarray(tangent)
               for value, tangent in zip(args, tangents, strict=True))
    )
    if isinstance(plus, (list, tuple)):
        values = [
            (np.asarray(p) - np.asarray(m)) / (2 * h)
            for p, m in zip(plus, minus, strict=True)
        ]
        return type(plus)(values)
    return (np.asarray(plus) - np.asarray(minus)) / (2 * h)

def _assert_runtime_matches_graph(
    symbolic_function, args, tangents=None, compare_push_forward=True
):
    graph = create_function_table(symbolic_function).entry
    graph_value = graph(*args)
    source_value = symbolic_function(*args)
    _assert_same(graph_value, source_value)

    if not compare_push_forward:
        return
    assert tangents is not None
    graph_push_forward = graph.push_forward(*args, *tangents)
    source_tangent = _finite_difference(symbolic_function, args, tangents)
    _assert_same(graph_push_forward[0], source_value)
    _assert_same(graph_push_forward[1], source_tangent)





def test_runtime_matches_scalar_quadratic_graph():
    symbolic_function = function(
        [Scalar("x")],
        implementation=lambda x: 3.0 * x + 4.0 * x * x,
        backend="coker",
    )
    _assert_runtime_matches_graph(
        symbolic_function, args=(1.25,), tangents=(-0.5,)
    )


def test_runtime_matches_matrix_transpose_and_concatenate_value_graph():
    symbolic_function = function(
        [VectorSpace("A", (2, 2)), VectorSpace("B", (2, 2))],
        implementation=lambda A, B: np.concatenate([A.T, B], axis=1),
        backend="coker",
    )
    args = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=args,
        tangents=(
            np.array([[0.5, -0.5], [1.0, -1.0]]),
            np.array([[0.25, 0.75], [-0.5, 0.5]]),
        ),
    )


def test_runtime_matches_matrix_transpose_push_forward_graph():
    symbolic_function = function(
        [VectorSpace("A", (2, 2))],
        implementation=lambda A: A.T,
        backend="coker",
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([[1.0, 2.0], [3.0, 4.0]]),),
        tangents=(np.array([[0.5, -0.5], [1.0, -1.0]]),),
    )


def test_runtime_matches_comparison_case_graph():
    symbolic_function = function(
        [Scalar("x")],
        implementation=lambda x: if_then_else(
            x <= 0,
            np.array([1.0, x, -2.0]),
            np.array([0.0, x + 1.0, 2.0]),
        ),
        backend="coker",
    )
    _assert_runtime_matches_graph(
        symbolic_function, args=(-2.0,), tangents=(0.25,)
    )
    _assert_runtime_matches_graph(
        symbolic_function, args=(3.0,), tangents=(-0.75,)
    )


def test_runtime_matches_nested_evaluate_graph():
    inner = function(
        [VectorSpace("x", 2)],
        implementation=lambda x: np.dot(x, x) + 1.0,
        backend="coker",
    )
    symbolic_function = function(
        [VectorSpace("x", 2)],
        implementation=lambda x: np.sqrt(inner(x)),
        backend="coker",
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([1.0, 2.0]),),
        tangents=(np.array([-0.5, 0.25]),),
    )


def test_runtime_matches_nested_evaluate_with_constant_argument_graph():
    inner = function(
        [VectorSpace("x", 2), VectorSpace("offset", 2)],
        implementation=lambda x, offset: x + offset,
        backend="coker",
    )
    symbolic_function = function(
        [VectorSpace("x", 2)],
        implementation=lambda x: inner(x, np.array([1.0, -2.0])),
        backend="coker",
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([3.0, 4.0]),),
        tangents=(np.array([-0.25, 0.75]),),
    )


def test_runtime_matches_vector_valued_nested_evaluate_graph():
    inner = function(
        [VectorSpace("x", 2)],
        implementation=lambda x: x + np.array([2.0, -1.0]),
        backend="coker",
    )
    symbolic_function = function(
        [VectorSpace("x", 2)],
        implementation=lambda x: inner(x) * np.array([0.5, -2.0]),
        backend="coker",
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([1.0, -3.0]),),
        tangents=(np.array([0.75, 0.5]),),
    )


def test_runtime_matches_dot_graph():
    symbolic_function = function(
        [VectorSpace("x", 3)],
        implementation=lambda x: np.dot(x, x),
        backend="coker",
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([1.0, -2.0, 0.5]),),
        tangents=(np.array([0.5, 0.25, -1.0]),),
    )

    def implementation(x):
        squared_norm = np.dot(x, x)
        vector = np.concatenate([squared_norm, x])
        return np.dot(vector, vector)

    symbolic_function = function(
        [VectorSpace("x", 3)],
        implementation=implementation,
        backend="coker",
    )
    graph = create_function_table(symbolic_function).entry
    value = np.array([1.0, -2.0, 0.5])
    squared_norm = np.dot(value, value)
    _assert_same(graph(value), squared_norm * (squared_norm + 1.0))

    symbolic_function = function(
        [VectorSpace("x", 3)],
        implementation=lambda x: np.cross(x, np.array([1.0, -2.0, 0.5])),
        backend="coker",
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([1.0, 2.0, 3.0]),),
        tangents=(np.array([-1.0, 0.5, 0.25]),),
    )

    symbolic_function = function(
        [VectorSpace("x", 3)],
        implementation=lambda x: x + np.array([1.0, -2.0, 3.5]),
        backend="coker",
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([4.0, -1.0, 0.25]),),
        tangents=(np.array([0.5, 1.5, -0.5]),),
    )


