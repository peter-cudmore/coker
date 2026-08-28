from dataclasses import dataclass

import numpy as np
import pytest

from coker import Scalar, VectorSpace, function, if_then_else
from coker.algebra.kernel import Function


@dataclass(frozen=True)
class RuntimeObservationPolicy:
    absolute_tolerance: float
    relative_tolerance: float


DESKTOP_RUNTIME_POLICY = RuntimeObservationPolicy(
    absolute_tolerance=1e-5,
    relative_tolerance=1e-5,
)


def _assert_same(actual, expected):
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_same(actual_item, expected_item)
        return
    np.testing.assert_allclose(actual, expected)


def _assert_runtime_observation(
    symbolic_function,
    args,
    tangents=None,
    policy=DESKTOP_RUNTIME_POLICY,
):
    """Compare mapped execution and JVPs against the NumPy tape oracle."""
    oracle = Function(
        symbolic_function.tape, symbolic_function.output, backend="numpy"
    )
    lowered = symbolic_function.lower()
    expected_value = oracle(*args)
    actual_value = lowered(args)
    _assert_same(actual_value, expected_value)
    assert isinstance(lowered.artifact, bytes)
    assert lowered.artifact

    if tangents is None:
        return

    actual_value, actual_tangent = lowered.push_forward(*args, *tangents)
    _assert_same(actual_value, expected_value)
    step = 1.0e-4
    upper = tuple(
        np.asarray(value) + step * np.asarray(tangent)
        for value, tangent in zip(args, tangents, strict=True)
    )
    lower = tuple(
        np.asarray(value) - step * np.asarray(tangent)
        for value, tangent in zip(args, tangents, strict=True)
    )
    expected_tangent = (
        np.asarray(oracle(*upper)) - np.asarray(oracle(*lower))
    ) / (2.0 * step)
    np.testing.assert_allclose(
        actual_tangent,
        expected_tangent,
        rtol=policy.relative_tolerance,
        atol=policy.absolute_tolerance,
    )


def _assert_runtime_matches_graph(
    symbolic_function, args, tangents=None, compare_push_forward=True
):
    _assert_runtime_observation(
        symbolic_function,
        args,
        tangents if compare_push_forward else None,
    )


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

    lowered = symbolic_function.lower()
    program = lowered.artifact

    assert isinstance(program, bytes)
    assert program
    assert lowered.artifact == program


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
    lowered = symbolic_function.lower()
    output = lowered(value)
    pushed_output, pushed_tangent = lowered.push_forward(value, 1.0)

    assert np.asarray(output[0]).shape == (4,)
    assert np.asarray(pushed_output[0]).shape == (4,)
    assert np.asarray(pushed_tangent[0]).shape == (4,)


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
