import coker
import sympy as sp
import numpy as np


def test_scalar_lowering():

    def f_impl(x, p):
        return x**2 + p

    f = coker.function(
        [coker.Scalar("x"), coker.Scalar("p")], f_impl, backend="sympy"
    )

    args, out = f.lower()
    assert args == [sp.Symbol("x"), sp.Symbol("p")]
    assert out == sp.Symbol("x") ** 2 + sp.Symbol("p")


def test_vector_lowering():
    A = np.array([[0, 1], [-1, 0]])

    def f_impl(x, p):

        return A @ x + p

    f = coker.function(
        [coker.VectorSpace("x", 2), coker.VectorSpace("p", 2)],
        f_impl,
        backend="sympy",
    )

    args, out = f.lower()
    x_symbol = sp.Array([sp.Symbol("x_0"), sp.Symbol("x_1")])
    p_symbol = sp.Array([sp.Symbol("p_0"), sp.Symbol("p_1")])
    assert args == [x_symbol, p_symbol]
    assert out == sp.Array([x_symbol[1], -x_symbol[0]]) + p_symbol
