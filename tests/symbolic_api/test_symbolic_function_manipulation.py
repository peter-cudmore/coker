from coker import function, Scalar, VectorSpace
import numpy as np

from tests.util import is_close


def test_function_composition(backend):
    sqr = function(
        arguments=[Scalar("x")], implementation=lambda x: x**2, backend=backend
    )

    quadratic = function(
        arguments=[Scalar("x")],
        implementation=lambda x: sqr(x) + x + 1,
        backend=backend,
    )

    assert quadratic(1) == 3


def test_function_composition_with_vector_input(backend):

    norm_sqr = function(
        arguments=[VectorSpace("x", 2)],
        implementation=lambda x: np.dot(x, x),
        backend=backend,
    )

    norm = function(
        arguments=[VectorSpace("x", 2)],
        implementation=lambda x: np.sqrt(norm_sqr(x)),
        backend=backend,
    )
    expected = np.sqrt(5)
    value = norm([1, 2])
    assert is_close(value, expected, tolerance=1e-6)
