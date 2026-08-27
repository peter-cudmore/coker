from dataclasses import dataclass

import numpy as np
import pytest

from coker import Scalar, VectorSpace, function, if_then_else
from coker.backends.coker.lowering import create_function_table
from coker.backends.coker.runtime import CompiledGraph


@dataclass(frozen=True)
class RuntimeObservationPolicy:
    absolute_tolerance: float
    relative_tolerance: float


DESKTOP_RUNTIME_POLICY = RuntimeObservationPolicy(
    absolute_tolerance=1e-5,
    relative_tolerance=1e-5,
)


def _assert_runtime_observation(
    symbolic_function,
    args,
    tangents,
    policy=DESKTOP_RUNTIME_POLICY,
):
    """Compare the current desktop runtime with the existing Coker graph."""
    graph = create_function_table(symbolic_function).entry
    compiled_graph = CompiledGraph.compile(graph)

    expected_value = graph(*args)
    actual_value = compiled_graph(*args)
    np.testing.assert_allclose(
        actual_value,
        expected_value,
        rtol=policy.relative_tolerance,
        atol=policy.absolute_tolerance,
    )
    expected_value, expected_tangent = graph.push_forward(*args, *tangents)
    actual_value, actual_tangent = compiled_graph.push_forward(
        *args, *tangents
    )
    np.testing.assert_allclose(
        actual_value,
        expected_value,
        rtol=policy.relative_tolerance,
        atol=policy.absolute_tolerance,
    )
    np.testing.assert_allclose(
        actual_tangent,
        expected_tangent,
        rtol=policy.relative_tolerance,
        atol=policy.absolute_tolerance,
    )


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
    function_table = create_function_table(symbolic_function)
    graph = function_table.entry
    compiled_graph = CompiledGraph.compile(function_table)

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
        symbolic_function, args=args, compare_push_forward=False
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
    function_table = create_function_table(symbolic_function)
    graph = function_table.entry
    assert not any(
        layer.opaque_programs
        for layer in graph.layers
        if hasattr(layer, "opaque_programs")
    )
    compiled_graph = CompiledGraph.compile(function_table)
    value = np.array([1.0, -2.0, 0.5])
    squared_norm = np.dot(value, value)
    _assert_same(compiled_graph(value), squared_norm * (squared_norm + 1.0))

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

    symbolic_function = function(
        [VectorSpace("x", 3)],
        implementation=lambda x: np.cross(x, np.array([1.0, -2.0, 0.5])),
        backend="coker",
    )

    lowered = symbolic_function.lower()
    program = lowered.compile_bytecode()

    assert isinstance(program, bytes)
    assert program
    assert (
        CompiledGraph.compile(create_function_table(symbolic_function)).program
        == program
    )


def test_runtime_observes_scalar_opcode_surface():
    def implementation(x, y):
        return np.array(
            [
                x,
                np.sin(x),
                np.cos(x),
                np.tan(x),
                np.exp(x),
                np.sqrt(x),
                np.log(x),
                -x,
                np.abs(-x),
                x + y,
                x - y,
                x * y,
                x / y,
                x**3,
                np.arctan2(x, y),
                x == y,
                x < y,
                x <= y,
                if_then_else(x < y, x, y),
            ]
        )

    symbolic_function = function(
        [Scalar("x"), Scalar("y")],
        implementation=implementation,
        backend="coker",
    )
    _assert_runtime_observation(
        symbolic_function,
        args=(1.25, 2.0),
        tangents=(0.25, -0.5),
    )


def test_variable_power_is_not_a_supported_source_operation():
    with pytest.raises(KeyError, match="PWR"):
        function(
            [Scalar("x"), Scalar("y")],
            implementation=lambda x, y: x**y,
            backend="coker",
        )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), -0.0],
)
def test_runtime_observes_special_float_values_without_crashing(value):
    symbolic_function = function(
        [Scalar("x")],
        implementation=lambda x: np.array(
            [np.sin(x), np.sqrt(x), np.log(x), x / x]
        ),
        backend="coker",
    )
    compiled_graph = CompiledGraph.compile(
        create_function_table(symbolic_function)
    )

    output = compiled_graph(value)
    pushed_output, pushed_tangent = compiled_graph.push_forward(value, 1.0)

    assert np.asarray(output).shape == (4,)
    assert np.asarray(pushed_output).shape == (4,)
    assert np.asarray(pushed_tangent).shape == (4,)


def test_runtime_observes_fork_join_unused_branch_and_deep_calls():
    first = function(
        [Scalar("x")],
        implementation=lambda x: x + 1.0,
        backend="coker",
    )
    second = function(
        [Scalar("x")],
        implementation=lambda x: first(x) * 2.0,
        backend="coker",
    )
    third = function(
        [Scalar("x")],
        implementation=lambda x: second(x) - 3.0,
        backend="coker",
    )

    def implementation(x):
        np.sin(x)
        left = third(x)
        right = x * x
        return left + right

    symbolic_function = function(
        [Scalar("x")],
        implementation=implementation,
        backend="coker",
    )
    _assert_runtime_observation(
        symbolic_function,
        args=(1.25,),
        tangents=(0.25,),
    )


def test_runtime_observes_near_identity_bilinear_map():
    symbolic_function = function(
        [Scalar("x")],
        implementation=lambda x: 1.01 * x,
        backend="coker",
    )
    _assert_runtime_observation(
        symbolic_function,
        args=(1.25,),
        tangents=(-0.5,),
        policy=RuntimeObservationPolicy(
            absolute_tolerance=1e-6,
            relative_tolerance=1e-5,
        ),
    )
