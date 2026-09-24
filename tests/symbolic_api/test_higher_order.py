import numpy as np
import pytest

from coker import function, Scalar, VectorSpace, FunctionSpace, Dimension
from coker.algebra.function import BoundCallable
from coker.algebra.graph import TraceContext
from coker.algebra.ops import Noop, OP
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


def test_bound_callable_capture_is_retained_in_function_table():
    target = function(
        [Scalar("t"), Scalar("a")],
        lambda t, a: a * t,
    )
    public_space = FunctionSpace(
        "scaled",
        arguments=[Scalar("t")],
        output=[Scalar("value")],
    )

    with TraceContext() as tape:
        captured_a = tape.input(Scalar("a"))
        reference = tape._create_function_reference(
            BoundCallable(target, public_space, (captured_a,))
        )
        function_value = tape.insert_function_value(reference)

    entry = tape.nodes._function_table[0]
    op, node_reference, node_capture = tape.nodes[function_value.index]
    assert reference.is_function_reference
    assert reference.target is target
    assert reference.capture_dependencies == (captured_a,)
    assert reference.function_space is public_space
    assert reference.result_dimension == Dimension.scalar()
    assert entry.target is target
    assert entry.capture_dependencies == (captured_a,)
    assert op is OP.FUNCTION_VALUE
    assert node_reference.is_function_reference
    assert node_reference.target is target
    assert node_reference.capture_dependencies == (captured_a,)
    assert node_capture is captured_a
    assert tape.depends_on(function_value, captured_a)

    with pytest.raises(TypeError, match="Function or BoundCallable"):
        tape._create_function_reference(lambda t: t)


def test_bound_callable_counts_only_present_target_inputs():
    target = function(
        [Scalar("x"), None, Noop(), Scalar("a")],
        lambda x, _absent, _noop, a: a * x,
    )
    public_space = FunctionSpace(
        "scaled",
        arguments=[Scalar("x")],
        output=[Scalar("value")],
    )

    with TraceContext() as tape:
        captured_a = tape.input(Scalar("a"))
        bound = BoundCallable(target, public_space, (captured_a,))
        reference = tape._create_function_reference(bound)

    bound_target, bound_arguments = bound.expand_call(2.0)
    assert bound_target is target
    assert bound_arguments == (2.0, captured_a)
    assert reference.capture_dependencies == (captured_a,)
    assert reference.function_space is public_space


def test_function_reference_excludes_absent_target_signature_values():
    target = function(
        [Scalar("x"), None, Noop()],
        lambda x, _absent, _noop: (x + 1, None),
        name="optional_target",
    )

    with TraceContext() as tape:
        reference = tape._create_function_reference(target)
        function_value = tape.insert_function_value(reference)
        raw_reference = tape._create_callable_reference(
            lambda x: x + 1,
            FunctionSpace(
                "raw",
                arguments=[Scalar("x")],
                output=[Scalar("value")],
            ),
        )

    entry = tape.nodes._function_table[0]
    assert [input_spec.space for input_spec in target.signature.inputs] == [
        Scalar("x"),
        None,
        Noop(),
    ]
    assert [output_spec.shape for output_spec in target.signature.outputs] == [
        Dimension.scalar(),
        None,
    ]
    assert reference.function_space.arguments == [Scalar("x")]
    assert reference.function_space.output == [Scalar("output_0")]
    assert reference.result_dimension == Dimension.scalar()
    assert entry.target is target
    assert (
        tape.dim[function_value.index].function_space
        is reference.function_space
    )
    bound = BoundCallable(target, reference.function_space, ())
    bound_target, bound_arguments = bound.expand_call(2.0)
    assert bound_target is target
    assert bound_arguments == (2.0,)
    bound_reference = tape._create_function_reference(bound)
    assert bound_reference.target is target
    assert bound_reference.function_space is reference.function_space

    wrapper = function(
        [reference.function_space, Scalar("wrapper_x")],
        lambda target_value, x: target_value(x),
    )
    composed = function(
        [Scalar("x")],
        lambda x: wrapper(target, x),
    )
    assert composed(2.0) == 3.0

    with pytest.raises(TypeError, match="resolved by a backend"):
        reference(2.0)
    assert raw_reference(2.0) == 3.0


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

    assert norm_1.output_shape() == (Dimension.scalar(),)
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
