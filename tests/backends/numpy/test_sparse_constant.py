"""
Regression test: sparse VALUE nodes in numpy evaluate_inner.

When a coker function captures a large, sparse numpy constant (density < 0.25),
Tape.try_sparsify converts it to scipy CSC format to save memory.  If that
function is later called with a Tracer argument — which routes through numpy's
evaluate_inner — the CSC sparse ends up as a VALUE node that evaluate_inner
must reshape.  numpy's reshape() did not handle scipy sparse arrays and raised
NotImplementedError.  Similarly, CasADi's to_casadi() failed to convert scipy
sparse arrays before building a ca.Function.
"""

import numpy as np
import pytest
from coker import function, VectorSpace


def _sparse_projection(n_out: int, n_in: int, offset: int) -> np.ndarray:
    """Dense numpy selection matrix: selects n_out rows starting at *offset*.

    With n_out=10, n_in=13, density = 10/130 ≈ 0.077, below the 0.25
    threshold in try_sparsify, so this matrix is stored as CSC sparse in
    the tape.
    """
    proj = np.zeros((n_out, n_in))
    for i in range(n_out):
        proj[i, offset + i] = 1.0
    return proj


def test_sparse_constant_concrete_call(backend):
    """A function with a sparsified constant evaluates correctly on all backends."""
    proj = _sparse_projection(10, 13, 3)

    def impl(x):
        return proj @ x

    f = function([VectorSpace("x", 13)], impl, backend=backend)

    x = np.ones(13)
    result = f(x)
    np.testing.assert_allclose(np.asarray(result).ravel(), proj @ x)


def test_sparse_constant_tracer_call():
    """
    A function with a sparsified VALUE-node constant must be callable in a
    tracing context (Tracer input), which routes through numpy's evaluate_inner.

    This reproduces the regression where numpy's reshape() raised
    NotImplementedError on scipy CSC sparse arrays produced by try_sparsify.
    """
    proj = _sparse_projection(10, 13, 3)

    def inner_impl(x):
        return proj @ x  # proj → CSC sparse VALUE node via try_sparsify

    inner = function([VectorSpace("x", 13)], inner_impl, backend="casadi")

    # Wrapping inner in an outer function causes it to be called with a Tracer,
    # triggering the numpy evaluate_inner path.
    def outer_impl(y):
        return inner(y)

    outer = function([VectorSpace("y", 13)], outer_impl, backend="numpy")

    x = np.ones(13)
    result = outer(x)
    np.testing.assert_allclose(np.asarray(result).ravel(), proj @ x)
