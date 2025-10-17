from coker import function, Scalar, VectorSpace, FunctionSpace
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


def test_partial_evaluation(backend):
    sqr = function(
        arguments=[Scalar("x")], implementation=lambda x: x**2, backend=backend
    )

    def evaluator(f, x):
        return f(x)

    evaluator_f = function(
        arguments=[FunctionSpace('f', arguments=[Scalar('x')], output=[Scalar('f(x)')]), Scalar("x")], implementation=evaluator, backend=backend
    )
    assert evaluator_f(sqr, 2) == 4

    def f_sqr_plus_one(f, x):
        f2 = evaluator_f.call_inline(f, x)
        return f2 + 1
    evaluator_f_sqr_plus_one = function(
        arguments=[FunctionSpace('f', arguments=[Scalar('x')], output=[Scalar('f(x)')]), Scalar("x")], implementation=f_sqr_plus_one, backend=backend
    )
    assert evaluator_f_sqr_plus_one(sqr, 2) == 5


