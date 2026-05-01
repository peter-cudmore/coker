"""Tests for higher-order function composition across backends.

Verifies that a function compiled with the default ("coker") backend can be
passed as a FunctionSpace argument into a "casadi"-compiled outer function
and evaluated correctly.  This pattern appears in Mechanica whenever a
body-level CasADi system calls individual compartment evaluators whose
boundary-flow inputs are Python callables.
"""

import importlib

import numpy as np
import pytest

from coker import function, Scalar, VectorSpace, FunctionSpace

casadi_available = importlib.util.find_spec("casadi") is not None


pytestmark = pytest.mark.skipif(
    not casadi_available, reason="CasADi not available"
)


# ---------------------------------------------------------------------------
# Scalar composition
# ---------------------------------------------------------------------------


def test_scalar_coker_function_in_casadi_outer():
    """coker-backend scalar f passed into casadi-backend outer function."""

    inner = function(
        arguments=[Scalar("x")],
        implementation=lambda x: x**2,
        backend="coker",
    )

    outer = function(
        arguments=[
            FunctionSpace("f", arguments=[Scalar("x")], output=[Scalar("y")]),
            Scalar("x"),
        ],
        implementation=lambda f, x: f(x) + x + 1,
        backend="casadi",
    )

    # f(x) = x^2 + x + 1;  at x=3: 9+3+1 = 13
    result = outer(inner, 3.0)
    assert abs(result - 13.0) < 1e-9, f"Expected 13.0, got {result}"


def test_plain_callable_in_casadi_outer():
    """A plain Python callable (no tape) passed into a casadi-backend outer."""

    def plain_square(x):
        return x**2

    outer = function(
        arguments=[
            FunctionSpace("f", arguments=[Scalar("x")], output=[Scalar("y")]),
            Scalar("x"),
        ],
        implementation=lambda f, x: f(x) + x,
        backend="casadi",
    )

    # f(x) = x^2 + x;  at x=4: 16+4 = 20
    result = outer(plain_square, 4.0)
    assert abs(result - 20.0) < 1e-9, f"Expected 20.0, got {result}"


# ---------------------------------------------------------------------------
# Vector composition
# ---------------------------------------------------------------------------


def test_vector_coker_function_in_casadi_outer():
    """coker-backend vector f(x)->y passed into casadi-backend outer."""

    # inner: R^2 -> R^2, doubles each element
    inner = function(
        arguments=[VectorSpace("x", 2)],
        implementation=lambda x: 2 * x,
        backend="coker",
    )

    outer = function(
        arguments=[
            FunctionSpace(
                "f",
                arguments=[VectorSpace("x", 2)],
                output=[VectorSpace("y", 2)],
            ),
            VectorSpace("x", 2),
        ],
        implementation=lambda f, x: f(x) + x,
        backend="casadi",
    )

    # f(x) = 2x + x = 3x; at x=[1,2]: [3,6]
    x_val = np.array([1.0, 2.0])
    result = outer(inner, x_val)
    expected = np.array([3.0, 6.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# FunctionSpace input that itself calls into the outer symbolic context
# ---------------------------------------------------------------------------


def test_casadi_outer_passes_symbolic_arg_to_inner():
    """The outer function evaluates the inner at a *symbolic* sub-expression.

    This mirrors the Mechanica pattern where the body-level CasADi system
    calls a spatial evaluator with a symbolic parameter vector ``p``, and
    the evaluator internally calls ``u(t)`` with symbolic ``t``.
    """

    # inner: R -> R, computes sin(x)
    inner = function(
        arguments=[Scalar("x")],
        implementation=lambda x: np.sin(x),
        backend="coker",
    )

    # outer: f, a, x -> f(a * x)
    # The inner is called at the *derived* value (a*x), not at x directly.
    outer = function(
        arguments=[
            FunctionSpace("f", arguments=[Scalar("x")], output=[Scalar("y")]),
            Scalar("a"),
            Scalar("x"),
        ],
        implementation=lambda f, a, x: f(a * x),
        backend="casadi",
    )

    a_val, x_val = 2.0, 0.5
    result = outer(inner, a_val, x_val)
    expected = np.sin(a_val * x_val)
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Nested: coker inner called from casadi middle, middle from casadi outer
# ---------------------------------------------------------------------------


def test_three_level_nesting():
    """Three-level nesting: coker leaf inside casadi middle inside
    casadi outer."""

    leaf = function(
        arguments=[Scalar("x")],
        implementation=lambda x: x + 1,
        backend="coker",
    )

    middle = function(
        arguments=[
            FunctionSpace("f", arguments=[Scalar("x")], output=[Scalar("y")]),
            Scalar("x"),
        ],
        implementation=lambda f, x: f(x) * 2,
        backend="casadi",
    )

    outer = function(
        arguments=[
            FunctionSpace("g", arguments=[Scalar("x")], output=[Scalar("y")]),
            Scalar("x"),
        ],
        implementation=lambda g, x: g(x) - x,
        backend="casadi",
    )

    # middle(leaf, x) = (x+1)*2
    # outer(middle(leaf,·), x) = middle(leaf, x) - x = (x+1)*2 - x = x+2
    # at x=5: 7
    result = outer(lambda x: middle(leaf, x), 5.0)
    assert abs(result - 7.0) < 1e-9, f"Expected 7.0, got {result}"


# ---------------------------------------------------------------------------
# Casadi inner with FunctionSpace input called from casadi outer
# (mirrors the Mechanica body-evaluator → spatial-evaluator pattern)
# ---------------------------------------------------------------------------


def test_casadi_inner_with_functionspace_called_from_casadi_outer():
    """casadi-backend inner that itself has a FunctionSpace input, called
    from a casadi-backend outer.

    This mirrors the Mechanica pattern:
      - "inner" ~ spatial evaluator dxdt: (t, x, p, u) -> dx  where u is a
        FunctionSpace (boundary flow callable)
      - "outer" ~ body-level dxdt: calls inner with a closure over its own
        accumulated boundary flow variable
    """

    # inner: (u: R->R, x: R, p: R) -> u(x) * p
    inner = function(
        arguments=[
            FunctionSpace(
                "u", arguments=[Scalar("t")], output=[Scalar("u(t)")]
            ),
            Scalar("x"),
            Scalar("p"),
        ],
        implementation=lambda u, x, p: u(x) * p,
        backend="casadi",
    )

    # outer: (x: R, p: R) -> inner(u=lambda t: t+1, x, p) + x
    # where u is built from outer's own variables (here just a closure over x)
    def outer_impl(x, p):
        u_closure = lambda t: t + 1  # noqa: E731 — plain callable, no tape
        return inner(u_closure, x, p) + x

    outer = function(
        arguments=[Scalar("x"), Scalar("p")],
        implementation=outer_impl,
        backend="casadi",
    )

    # inner(u, x=2, p=3) = u(2)*3 = 3*3 = 9;  outer = 9 + 2 = 11
    result = outer(2.0, 3.0)
    assert abs(result - 11.0) < 1e-9, f"Expected 11.0, got {result}"


def test_casadi_inner_with_functionspace_closure_over_outer_variable():
    """The FunctionSpace argument passed to the inner function closes over a
    variable from the outer function — the closure must carry the symbolic
    value through correctly.
    """

    # inner: (u: R->R, x: R) -> u(x)
    inner = function(
        arguments=[
            FunctionSpace(
                "u", arguments=[Scalar("t")], output=[Scalar("u(t)")]
            ),
            Scalar("x"),
        ],
        implementation=lambda u, x: u(x),
        backend="casadi",
    )

    # outer: (a: R, x: R) -> inner(u=lambda t: a*t, x) + a
    # u closes over `a`, an outer parameter
    def outer_impl(a, x):
        u_closure = lambda t: a * t  # noqa: E731
        return inner(u_closure, x) + a

    outer = function(
        arguments=[Scalar("a"), Scalar("x")],
        implementation=outer_impl,
        backend="casadi",
    )

    # inner(u, x=3) = a*3;  outer = a*3 + a = a*4; at a=2: 8
    result = outer(2.0, 3.0)
    assert abs(result - 8.0) < 1e-9, f"Expected 8.0, got {result}"
