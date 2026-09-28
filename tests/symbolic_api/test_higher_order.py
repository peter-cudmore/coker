import numpy as np
import pytest

from coker import function, Scalar, VectorSpace, FunctionSpace, Dimension
from coker.algebra.function import BoundCallable, Function
from coker.algebra.graph import Tape, TraceContext, Tracer
from coker.interfaces import SymbolicCallable
from coker.algebra.ops import Noop, OP
from coker.algebra.exceptions import InvalidArgument, InvalidShape
from coker.backends.backend import create_native_symbol_entry

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


def test_symbol_emission_requires_symbolic_callable_base():
    class DeclaredEmitter(SymbolicCallable):
        def __call__(self, *args):
            raise NotImplementedError

        def lower(self):
            raise NotImplementedError

        def _emit_symbol_value(self, tape):
            return tape.input(Scalar("emitted"))

    class UndeclaredEmitter:
        def _emit_symbol_value(self, tape):
            return tape.input(Scalar("emitted"))

    with TraceContext() as tape:
        emitted = tape.insert_symbol_value(DeclaredEmitter())
        with pytest.raises(TypeError, match="SymbolicCallable"):
            tape.insert_symbol_value(UndeclaredEmitter())

    assert emitted.dim == Dimension.scalar()


def test_bound_callable_emits_base_symbol_and_bind_capture_dependency():
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
        bound = BoundCallable(target, public_space, (captured_a,))
        function_value = tape.insert_symbol_value(bound)

    bind_op, base_value, bind_capture, bind_position = tape.nodes[
        function_value.index
    ]
    function_op, symbol = tape.nodes[base_value.index]
    entry = tape.nodes._symbol_table[0]
    assert function_op is OP.FUNCTION
    assert bind_op is OP.BIND
    assert bind_capture.tape is tape
    assert bind_capture.index == captured_a.index
    assert bind_position == 1
    assert symbol.target is target
    assert symbol.function_space.arguments == [Scalar("t"), Scalar("a")]
    assert entry.target is target
    assert entry.target is not bound
    assert function_value.dim.arguments == [Scalar("t")]
    assert tape.depends_on(function_value, captured_a)


def test_bound_callable_captures_share_symbol_and_evaluate(backend):
    target = function(
        [Scalar("t"), Scalar("a")],
        lambda t, a: a * t,
        backend=backend,
    )
    public_space = FunctionSpace(
        "scaled",
        arguments=[Scalar("t")],
        output=[Scalar("value")],
    )
    tape = Tape(backend)
    captured_a = tape.input(Scalar("a"))
    captured_b = tape.input(Scalar("b"))
    x = tape.input(Scalar("x"))

    first_value = tape.insert_symbol_value(
        BoundCallable(target, public_space, (captured_a,))
    )
    second_value = tape.insert_symbol_value(
        BoundCallable(target, public_space, (captured_b,))
    )
    first_result = Tracer(tape, tape.append(OP.EVALUATE, first_value, x))
    second_result = Tracer(tape, tape.append(OP.EVALUATE, second_value, x))
    composed = Function(
        tape,
        (first_result, second_result),
        backend=backend,
    )

    assert len(tape.nodes._symbol_table) == 1
    assert tape.nodes._symbol_table[0].target is target
    assert (
        sum(
            isinstance(tape.nodes[index], tuple)
            and tape.nodes[index][0] is OP.FUNCTION
            for index in range(len(tape.nodes))
        )
        == 1
    )
    bound_nodes = [
        tape.nodes[function_value.index]
        for function_value in (first_value, second_value)
    ]
    assert bound_nodes[0][1].index == bound_nodes[1][1].index
    for (
        (op, base_value, bind_capture, position),
        function_value,
        capture,
    ) in zip(
        bound_nodes,
        (first_value, second_value),
        (captured_a, captured_b),
    ):
        assert op is OP.BIND
        assert tape.op(base_value.index) is OP.FUNCTION
        assert bind_capture.tape is tape
        assert bind_capture.index == capture.index
        assert position == 1
        assert tape.depends_on(base_value, capture) is False
        assert tape.depends_on(function_value, capture)

    first_result, second_result = composed(2.0, 5.0, 3.0)
    assert float(first_result) == 6.0
    assert float(second_result) == 15.0


def test_bound_callable_uses_one_bind_node_per_capture():
    target = function(
        [Scalar("t"), Scalar("a"), Scalar("b")],
        lambda t, a, b: a * t + b,
    )
    public_space = FunctionSpace(
        "affine",
        arguments=[Scalar("t")],
        output=[Scalar("value")],
    )

    with TraceContext() as tape:
        captured_a = tape.input(Scalar("a"))
        captured_b = tape.input(Scalar("b"))
        function_value = tape.insert_symbol_value(
            BoundCallable(target, public_space, (captured_a, captured_b))
        )

    second_bind_op, first_bound_value, second_capture, second_position = (
        tape.nodes[function_value.index]
    )
    first_bind_op, base_value, first_capture, first_position = tape.nodes[
        first_bound_value.index
    ]
    assert first_bind_op is OP.BIND
    assert second_bind_op is OP.BIND
    assert base_value.dim.arguments == [
        Scalar("t"),
        Scalar("a"),
        Scalar("b"),
    ]
    assert first_bound_value.dim.arguments == [
        Scalar("t"),
        Scalar("b"),
    ]
    assert function_value.dim.arguments == [Scalar("t")]
    assert first_capture.tape is tape
    assert first_capture.index == captured_a.index
    assert first_position == 1
    assert second_capture.tape is tape
    assert second_capture.index == captured_b.index
    assert second_position == 1
    assert tape.depends_on(function_value, captured_a)
    assert tape.depends_on(function_value, captured_b)


def test_bind_validates_position_and_argument_dimension():
    target = function(
        [Scalar("x"), Scalar("a")],
        lambda x, a: a * x,
    )
    with TraceContext() as tape:
        x = tape.input(Scalar("x"))
        vector = tape.input(VectorSpace("v", 2))
        function_value = tape.insert_symbol_value(target)

        with pytest.raises(InvalidArgument, match="outside the callable"):
            tape.append(OP.BIND, function_value, x, -1)
        with pytest.raises(InvalidArgument, match="outside the callable"):
            tape.append(OP.BIND, function_value, x, 2)
        with pytest.raises(TypeError, match="must be an integer"):
            tape.append(OP.BIND, function_value, x, 0.0)
        with pytest.raises(InvalidShape, match="BIND argument 0"):
            tape.append(OP.BIND, function_value, vector, 0)
        signed_symbol = tape.intern_symbol(
            create_native_symbol_entry(
                lambda x, a: a * x,
                FunctionSpace(
                    "signed",
                    arguments=[Scalar("x"), Scalar("a")],
                    output=[Scalar("value")],
                    signature=(1, 2),
                ),
            )
        )
        signed_value = tape.insert_symbol_value(signed_symbol)
        reduced_value = Tracer(
            tape,
            tape.append(OP.BIND, signed_value, x, 0),
        )
        assert reduced_value.dim.signature == (2,)


def test_bind_accepts_function_valued_input():
    mapped_space = FunctionSpace(
        "mapped",
        arguments=[Scalar("x")],
        output=[Scalar("value")],
    )
    target = function(
        [mapped_space, Scalar("x")],
        lambda mapped, x: mapped(x),
    )

    with TraceContext() as tape:
        mapped_input = tape.input(mapped_space)
        target_value = tape.insert_symbol_value(target)
        bound_value = Tracer(
            tape,
            tape.append(OP.BIND, target_value, mapped_input, 0),
        )

    assert bound_value.dim.arguments == [Scalar("x")]
    assert bound_value.dim.output_dimensions() == [Dimension.scalar()]


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
        function_value = tape.insert_symbol_value(bound)

    bind_op, base_value, bind_capture, bind_position = tape.nodes[
        function_value.index
    ]
    assert bind_op is OP.BIND
    assert base_value.dim.arguments == [
        Scalar("x"),
        Scalar("a"),
    ]
    assert bind_capture.tape is tape
    assert bind_capture.index == captured_a.index
    assert bind_position == 1


def test_function_symbols_preserve_absent_target_signature_values():
    target = function(
        [Scalar("x"), None, Noop()],
        lambda x, _absent, _noop: (x + 1, None),
        name="optional_target",
    )

    def native(x):
        return x + 1

    with TraceContext() as tape:
        symbol = tape.intern_symbol(target._symbol_entry())
        function_value = tape.insert_symbol_value(symbol)
        native_symbol = tape.intern_symbol(
            create_native_symbol_entry(
                native,
                FunctionSpace(
                    "raw",
                    arguments=[Scalar("x")],
                    output=[Scalar("value")],
                ),
            )
        )

    entry = tape.nodes._symbol_table[0]
    assert [input_spec.space for input_spec in target.signature.inputs] == [
        Scalar("x"),
        None,
        Noop(),
    ]
    assert [output_spec.shape for output_spec in target.signature.outputs] == [
        Dimension.scalar(),
        None,
    ]
    assert symbol.function_space.arguments == [Scalar("x")]
    assert symbol.function_space.output == [Scalar("output_0")]
    assert entry.target is target
    assert tape.dim[function_value.index] is symbol.function_space
    assert native_symbol.target is native

    wrapper = function(
        [symbol.function_space, Scalar("wrapper_x")],
        lambda target_value, x: target_value(x),
    )
    composed = function(
        [Scalar("x")],
        lambda x: wrapper(target, x),
    )
    assert composed(2.0) == 3.0


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
