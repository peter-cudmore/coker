import pytest

from coker import Scalar, function
from coker.backends import get_backend_by_name
from coker.algebra.ops import OP
from coker.algebra.graph import FunctionSymbol
from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    ResultBundleDimension,
)
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
)


def test_multi_output_native_function_executes_once_per_call():
    calls = []

    def native(x):
        calls.append(x)
        return x + 1, x * 2

    signature = FunctionSignature(
        inputs=(FunctionInputSpec("x", Scalar("x")),),
        outputs=(
            FunctionOutputSpec("incremented", Scalar("incremented")),
            FunctionOutputSpec("doubled", Scalar("doubled")),
        ),
    )
    imported = get_backend_by_name("numpy").import_function(native, signature)

    assert imported(3.0) == (4.0, 6.0)
    assert calls == [3.0]


def test_native_result_bundle_preserves_absent_outputs():
    def native(x):
        return x + 1, None

    signature = FunctionSignature(
        inputs=(FunctionInputSpec("x", Scalar("x")),),
        outputs=(
            FunctionOutputSpec("incremented", Scalar("incremented")),
            FunctionOutputSpec("absent", None),
        ),
    )
    imported = get_backend_by_name("numpy").import_function(native, signature)

    assert imported(3.0) == (4.0, None)


def test_native_function_rejects_cross_backend_composition():
    signature = FunctionSignature(
        inputs=(FunctionInputSpec("x", Scalar("x")),),
        outputs=(FunctionOutputSpec("output", Scalar("output")),),
    )
    imported = get_backend_by_name("numpy").import_function(
        lambda x: x, signature
    )

    with pytest.raises(
        RuntimeError, match="native callable for backend 'numpy'"
    ):
        function([Scalar("x")], lambda x: imported(x), backend="sympy")


def test_native_result_bundle_rejects_wrong_result_count():
    signature = FunctionSignature(
        inputs=(FunctionInputSpec("x", Scalar("x")),),
        outputs=(
            FunctionOutputSpec("incremented", Scalar("incremented")),
            FunctionOutputSpec("doubled", Scalar("doubled")),
        ),
    )
    imported = get_backend_by_name("numpy").import_function(
        lambda x: (x + 1,), signature
    )

    with pytest.raises(ValueError, match="returned 1 results; expected 2"):
        imported(3.0)


def test_native_function_space_preserves_absent_outputs():
    signature = FunctionSignature(
        inputs=(FunctionInputSpec("x", Scalar("x")),),
        outputs=(
            FunctionOutputSpec("present", Scalar("present")),
            FunctionOutputSpec("absent", None),
        ),
    )
    imported = get_backend_by_name("numpy").import_function(
        lambda x: (x, None), signature
    )
    function_value_index, (_, native_symbol) = next(
        (index, node)
        for index, node in enumerate(imported.tape.nodes)
        if isinstance(node, tuple) and node[0] == OP.FUNCTION
    )
    evaluate_index, evaluate_node = next(
        (index, node)
        for index, node in enumerate(imported.tape.nodes)
        if isinstance(node, tuple) and node[0] == OP.EVALUATE
    )

    assert native_symbol.function_space.output == [Scalar("present"), None]
    assert type(native_symbol) is FunctionSymbol
    assert evaluate_node[1].index == function_value_index
    assert isinstance(imported.tape.dim[function_value_index], FunctionSpace)
    assert (
        imported.tape.dim[function_value_index] is native_symbol.function_space
    )
    assert isinstance(imported.tape.dim[evaluate_index], ResultBundleDimension)
    assert imported.tape.dim[evaluate_index].outputs == (
        Dimension.scalar(),
        None,
    )


@pytest.mark.parametrize(
    "backend_name, optional_module",
    [
        ("numpy", None),
        ("pytorch", "torch"),
        ("jax", "jax"),
        ("sympy", "sympy"),
    ],
)
def test_native_result_selection_dispatches_on_every_backend(
    backend_name, optional_module
):
    if optional_module is not None:
        pytest.importorskip(optional_module)

    calls = []

    def native(x):
        calls.append(x)
        return x + 1, x * 2

    signature = FunctionSignature(
        inputs=(FunctionInputSpec("x", Scalar("x")),),
        outputs=(
            FunctionOutputSpec("incremented", Scalar("incremented")),
            FunctionOutputSpec("doubled", Scalar("doubled")),
        ),
    )
    imported = get_backend_by_name(backend_name).import_function(
        native, signature
    )

    result = imported(3.0)

    assert tuple(float(value) for value in result) == (4.0, 6.0)
    assert len(calls) == 1
