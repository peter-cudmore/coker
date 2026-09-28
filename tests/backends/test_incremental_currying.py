import pytest

from coker import FunctionSpace, Scalar, function
from coker.algebra.function import BoundCallable
from coker.backends import get_backend_by_name
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
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
def test_imported_native_function_supports_incremental_currying(
    backend_name, optional_module
):
    if optional_module is not None:
        pytest.importorskip(optional_module)

    signature = FunctionSignature(
        inputs=(
            FunctionInputSpec("x", Scalar("x")),
            FunctionInputSpec("scale", Scalar("scale")),
            FunctionInputSpec("offset", Scalar("offset")),
        ),
        outputs=(FunctionOutputSpec("value", Scalar("value")),),
    )
    imported = get_backend_by_name(backend_name).import_function(
        lambda x, scale, offset: scale * x + offset,
        signature,
    )
    public_space = FunctionSpace(
        "affine",
        arguments=[Scalar("x")],
        output=[Scalar("value")],
    )
    composed = function(
        [Scalar("scale"), Scalar("offset"), Scalar("x")],
        lambda scale, offset, x: BoundCallable(
            imported, public_space, (scale, offset)
        )(x),
        backend=backend_name,
    )

    assert float(composed(3.0, 4.0, 2.0)) == pytest.approx(10.0)
