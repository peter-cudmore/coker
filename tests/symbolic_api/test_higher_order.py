import numpy as np

from coker import function, Scalar, VectorSpace, FunctionSpace, Dimension
from ..util import is_close


def test_functional(backend):

    def f_inner(A, b, x):
        return A @ x + b

    def f_outer(f, x):
        A = np.array([[0, 1], [1, 0]], dtype=float)
        b = np.array([0, 0], dtype=float)
        return f(A, b, x)

    f_result = f_outer(f_inner, np.array([2, 3], dtype=float))
    assert is_close(f_result, np.array([3, 2], dtype=float))

    f_coker = function(
        arguments=[
            FunctionSpace(
                name="f_inner",
                arguments=[
                    VectorSpace(name="A", dimension=(2, 2)),
                    VectorSpace(name="b", dimension=2),
                    VectorSpace(name="x", dimension=2),
                ],
                output=[VectorSpace(name="y", dimension=2)],
                signature=None,
            ),
            VectorSpace(name="x", dimension=2),
        ],
        implementation=f_outer,
        backend=backend,
    )

    f_coker_result = f_coker(f_inner, np.array([2, 3], dtype=float))
    assert is_close(f_coker_result, np.array([3, 2], dtype=float))


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
        arguments=[
            FunctionSpace(
                "f", arguments=[Scalar("x")], output=[Scalar("f(x)")]
            ),
            Scalar("x"),
        ],
        implementation=evaluator,
        backend=backend,
    )
    assert evaluator_f(sqr, 2) == 4

    def f_sqr_plus_one(f, x):
        f2 = evaluator_f.call_inline(f, x)
        return f2 + 1

    evaluator_f_sqr_plus_one = function(
        arguments=[
            FunctionSpace(
                "f", arguments=[Scalar("x")], output=[Scalar("f(x)")]
            ),
            Scalar("x"),
        ],
        implementation=f_sqr_plus_one,
        backend=backend,
    )
    assert evaluator_f_sqr_plus_one(sqr, 2) == 5


def test_vector_norm_order_is_preserved(backend):

    norm_1 = function(
        arguments=[VectorSpace("x", 2)],
        implementation=lambda x: np.linalg.norm(x, ord=1),
        backend=backend,
    )

    assert norm_1.output_shape() == (Dimension(None),)
    result = norm_1(np.array([3.0, -4.0]))
    assert isinstance(result, float)
    assert is_close(result, 7.0, tolerance=1e-6)


def test_matrix_norm_order_is_preserved(backend):

    matrix_norm = function(
        arguments=[VectorSpace("A", (2, 2))],
        implementation=lambda A: np.linalg.norm(A, ord=1),
        backend=backend,
    )
    matrix = np.array([[1.0, -2.0], [3.0, 4.0]])
    assert is_close(
        matrix_norm(matrix),
        np.linalg.norm(matrix, ord=1),
        tolerance=1e-6,
    )
