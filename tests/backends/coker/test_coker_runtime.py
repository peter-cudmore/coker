import numpy as np

from coker import Scalar, VectorSpace, function, if_then_else
from coker.backends.coker.core import create_opgraph
from coker.backends.coker.runtime import CompiledGraph


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


def _assert_runtime_matches_graph(
    symbolic_function, args, tangents=None, compare_push_forward=True
):
    graph = create_opgraph(symbolic_function)
    compiled_graph = CompiledGraph.compile(graph)

    assert isinstance(compiled_graph.program, bytes)
    assert compiled_graph.program

    graph_value = graph(*args)
    compiled_value = compiled_graph(*args)
    _assert_same(compiled_value, graph_value)

    if not compare_push_forward:
        return

    assert tangents is not None
    graph_push_forward = graph.push_forward(*args, *tangents)
    compiled_push_forward = compiled_graph.push_forward(*args, *tangents)
    _assert_same(compiled_push_forward[0], graph_push_forward[0])
    _assert_same(compiled_push_forward[1], graph_push_forward[1])


def test_runtime_matches_scalar_quadratic_graph():
    symbolic_function = function(
        [Scalar("x")], implementation=lambda x: 3.0 * x + 4.0 * x * x
    )
    _assert_runtime_matches_graph(
        symbolic_function, args=(1.25,), tangents=(-0.5,)
    )


def test_runtime_matches_matrix_transpose_and_concatenate_value_graph():
    symbolic_function = function(
        [VectorSpace("A", (2, 2)), VectorSpace("B", (2, 2))],
        implementation=lambda A, B: np.concatenate([A.T, B], axis=1),
    )
    args = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )
    _assert_runtime_matches_graph(
        symbolic_function, args=args, compare_push_forward=False
    )


def test_runtime_matches_matrix_transpose_push_forward_graph():
    symbolic_function = function(
        [VectorSpace("A", (2, 2))], implementation=lambda A: A.T
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
    )
    _assert_runtime_matches_graph(
        symbolic_function, args=(-2.0,), tangents=(0.25,)
    )
    _assert_runtime_matches_graph(
        symbolic_function, args=(3.0,), tangents=(-0.75,)
    )


def test_runtime_matches_nested_evaluate_graph():
    inner = function(
        [VectorSpace("x", 2)], implementation=lambda x: np.dot(x, x) + 1.0
    )
    symbolic_function = function(
        [VectorSpace("x", 2)], implementation=lambda x: np.sqrt(inner(x))
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
    )
    symbolic_function = function(
        [VectorSpace("x", 2)],
        implementation=lambda x: inner(x, np.array([1.0, -2.0])),
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
    )
    symbolic_function = function(
        [VectorSpace("x", 2)],
        implementation=lambda x: inner(x) * np.array([0.5, -2.0]),
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([1.0, -3.0]),),
        tangents=(np.array([0.75, 0.5]),),
    )

def test_runtime_matches_dot_graph():
    symbolic_function = function(
        [VectorSpace("x", 3)], implementation=lambda x: np.dot(x, x)
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([1.0, -2.0, 0.5]),),
        tangents=(np.array([0.5, 0.25, -1.0]),),
    )


def test_runtime_matches_cross_graph():
    symbolic_function = function(
        [VectorSpace("x", 3)],
        implementation=lambda x: np.cross(x, np.array([1.0, -2.0, 0.5])),
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([1.0, 2.0, 3.0]),),
        tangents=(np.array([-1.0, 0.5, 0.25]),),
    )


def test_runtime_matches_mixed_constant_workspace_graph():
    symbolic_function = function(
        [VectorSpace("x", 3)],
        implementation=lambda x: x + np.array([1.0, -2.0, 3.5]),
    )
    _assert_runtime_matches_graph(
        symbolic_function,
        args=(np.array([4.0, -1.0, 0.25]),),
        tangents=(np.array([0.5, 1.5, -0.5]),),
    )
