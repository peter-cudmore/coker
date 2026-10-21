import numpy as np
import pytest

from coker import function, Scalar, if_then_else
from coker.algebra.exceptions import InvalidShape
from ..util import is_close


def test_case(backend):

    def f_impl(x):
        expression = x == 0
        return if_then_else(
            expression,
            np.array([1, 0, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        )

    f = function(
        arguments=[Scalar("x")],
        implementation=f_impl,
        backend=backend,
    )

    test_values = [0, 1]
    for test_value in test_values:
        expected = f_impl(test_value)
        result = f(test_value)
        assert is_close(
            result, expected
        ), f"For x={test_value}, got {result}, expected {expected}"


@pytest.mark.parametrize("value, expected", [(0.0, 0.0), (-2.0, -0.5), (2.0, 0.5)])
def test_sympy_case_selects_value_over_zero_division(value, expected):
    guarded = function(
        [Scalar("x")],
        lambda x: if_then_else(x == 0, 0, 1 / x),
        backend="sympy",
    )

    assert float(guarded(value)) == expected


def test_sympy_zero_division_returns_nan_when_selected():
    reciprocal = function([Scalar("x")], lambda x: 1 / x, backend="sympy")

    assert np.isnan(float(reciprocal(0.0)))


def test_case_mismatched_branches_raises():
    """Branches of different shapes must raise InvalidShape at trace time."""
    with pytest.raises(InvalidShape):
        function(
            arguments=[Scalar("x")],
            implementation=lambda x: if_then_else(
                x == 0,
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0]),
            ),
            backend="numpy",
        )


def test_case_non_comparison_tracer_raises():
    """A Tracer not produced by a comparison operator must raise TypeError."""
    with pytest.raises(TypeError):
        function(
            arguments=[Scalar("x")],
            implementation=lambda x: if_then_else(x, 1.0, 0.0),
            backend="numpy",
        )


def test_case_ambiguous_condition_raises():
    """A multi-element array condition must raise TypeError."""
    with pytest.raises(TypeError):
        if_then_else(np.array([1.0, 0.0]), 1.0, 0.0)
