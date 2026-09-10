import numpy as np
import pytest

ca = pytest.importorskip("casadi")
from coker import Scalar, function
from coker.backends.casadi import CasadiBackend
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
)



def scalar_signature(name="x"):
    return FunctionSignature(
        inputs=(FunctionInputSpec(name, Scalar(name)),),
        outputs=(FunctionOutputSpec("output_0", Scalar("output_0")),),
    )


def test_imported_casadi_function_composes_symbolically_without_tracers():
    native_x = ca.MX.sym("native_x")
    native = ca.Function("native", [native_x], [2 * native_x + 3])
    imported = CasadiBackend().import_function(native, scalar_signature())
    inner = function([Scalar("x")], lambda x: x + 1, backend="casadi")
    outer = function([Scalar("x")], lambda x: imported(inner(x)), backend="casadi")

    lowered = outer.lower()
    (result,) = lowered.execute([4.0])

    assert isinstance(result, (ca.DM, ca.MX))
    assert np.allclose(np.asarray(result), [[13.0]])


def test_imported_casadi_function_uses_backend_name_and_native_signature():
    x = ca.MX.sym("x")
    imported = CasadiBackend().import_function(
        ca.Function("native", [x], [x]), scalar_signature("argument")
    )

    assert imported.backend == "casadi"
    assert imported.signature == scalar_signature("argument")



def test_imported_casadi_function_derives_its_signature():
    scalar = ca.MX.sym("scalar")
    vector = ca.MX.sym("vector", 2, 1)
    native = ca.Function(
        "native",
        [scalar, vector],
        [scalar + 1, vector * 2],
        ["time", "state"],
        ["offset_time", "scaled_state"],
    )

    imported = CasadiBackend().import_function(native)

    assert [spec.name for spec in imported.signature.inputs] == [
        "time",
        "state",
    ]
    assert [spec.space.dimension if hasattr(spec.space, "dimension") else None
            for spec in imported.signature.inputs] == [None, 2]
    assert [spec.name for spec in imported.signature.outputs] == [
        "offset_time",
        "scaled_state",
    ]
    assert [spec.shape.dim for spec in imported.signature.outputs] == [
        None,
        (2,),
    ]


def test_imported_casadi_function_rejects_cross_backend_lowering():
    x = ca.MX.sym("x")
    imported = CasadiBackend().import_function(
        ca.Function("native", [x], [x]), scalar_signature()
    )

    from coker.backends.lowered import LoweringOptions

    with pytest.raises(RuntimeError, match="native callable for backend 'casadi'"):
        imported.lower(LoweringOptions(backend="numpy"))

def test_functionspace_lowering_keeps_evaluate_fallback_boundary():
    from coker import FunctionSpace

    space = FunctionSpace("f", [Scalar("x")], [Scalar("y")])
    outer = function([space], lambda f: f(3.0), backend="casadi")
    inner = function([Scalar("x")], lambda x: x * 2.0, backend="casadi")

    lowered = outer.lower()
    assert lowered.ca_function is None
    assert lowered.execute([inner]) == (6.0,)
