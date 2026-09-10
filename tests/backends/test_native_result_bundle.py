import pytest

from coker import Function, Scalar
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
    imported = Function.from_native(native, signature, backend="numpy")

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
    imported = Function.from_native(native, signature, backend="numpy")

    assert imported(3.0) == (4.0, None)


@pytest.mark.parametrize(
    "backend_name, optional_module",
    [
        ("numpy", None),
        ("pytorch", "torch"),
        ("casadi", "casadi"),
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
    imported = Function.from_native(native, signature, backend=backend_name)

    result = imported(3.0)

    assert tuple(float(value) for value in result) == (4.0, 6.0)
    assert len(calls) == 1
