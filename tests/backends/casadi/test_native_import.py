import numpy as np
import pytest

from coker import FunctionSpace, Scalar, function
from coker.algebra.ops import Noop
from coker.backends.casadi import CasadiBackend
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
    LoweringOptions,
)
from coker.dynamics import DynamicsSpec, VariationalProblemBuilder
from coker.dynamics.system import create_dynamics_from_spec
from coker.parameters import BoundedVariable
from coker.parameters.function_parameters import (
    ClosureParameter,
    FittedFunction,
)
from coker.toolkits.codesign import Minimise

ca = pytest.importorskip("casadi")


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
    outer = function(
        [Scalar("x")], lambda x: imported(inner(x)), backend="casadi"
    )

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

    input_dimensions = [
        spec.space.dimension if hasattr(spec.space, "dimension") else None
        for spec in imported.signature.inputs
    ]
    assert input_dimensions == [None, 2]
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

    with pytest.raises(
        RuntimeError, match="native callable for backend 'casadi'"
    ):
        imported.lower(LoweringOptions(backend="numpy"))


def test_functionspace_lowering_keeps_evaluate_fallback_boundary():

    space = FunctionSpace("f", [Scalar("x")], [Scalar("y")])
    outer = function([space], lambda f: f(3.0), backend="casadi")
    inner = function([Scalar("x")], lambda x: x * 2.0, backend="casadi")

    lowered = outer.lower()
    assert lowered.ca_function is None
    assert lowered.execute([inner]) == (6.0,)


def test_casadi_solution_reconstructs_native_function_parameter():
    response = FunctionSpace(
        "response",
        arguments=[Scalar("inflow")],
        output=[Scalar("rate")],
    )
    scale = ca.MX.sym("scale")
    inflow = ca.MX.sym("inflow")
    native_rate = CasadiBackend().import_function(
        ca.Function("native_rate", [scale, inflow], [scale * inflow]),
        FunctionSignature(
            inputs=(
                FunctionInputSpec("scale", Scalar("scale")),
                FunctionInputSpec("inflow", Scalar("inflow")),
            ),
            outputs=(FunctionOutputSpec("rate", Scalar("rate")),),
        ),
    )
    native_rate.name = "response"
    declaration = ClosureParameter(
        native_rate,
        (
            BoundedVariable(
                "rate_scale", lower_bound=1.0, upper_bound=1.0, guess=1.0
            ),
        ),
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(response,),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (0.0, None),
            dynamics=lambda _t, _x, _z, _u, parameters: parameters[0](0.5),
            constraints=Noop(),
            outputs=lambda _t, state, _z, _u, _p, _q: state,
            quadratures=Noop(),
        ),
        backend="casadi",
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[declaration],
        backend="casadi",
    ) as builder:
        problem = builder.build(
            Minimise((builder.output(builder.t_final)[0] - 0.5) ** 2)
        )

    solution = problem()

    fitted = solution.parameters["response"]
    assert isinstance(fitted, FittedFunction)
    assert fitted(0.25) == pytest.approx(0.25)
