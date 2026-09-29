import importlib.util

import numpy as np
import pytest

from coker import FunctionSpace, Scalar, function
from coker.backends import get_backend_by_name
from coker.parameters import UnboundedVariable
from coker.parameters.function_parameters.base import FittedFunction
from coker.parameters.function_parameters.bound import BoundFunctionParameter
from coker.toolkits.codesign import Minimise, ProblemBuilder


@pytest.mark.parametrize(
    "optimiser_backend",
    [
        "numpy",
        pytest.param(
            "casadi",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("casadi") is None,
                reason="requires casadi",
            ),
        ),
    ],
)
def test_mathematical_program_fits_cubic_bound_function_parameter(
    optimiser_backend,
):
    """Fit a cubic through a bound function parameter."""
    get_backend_by_name(optimiser_backend)
    cubic = function(
        [
            Scalar("constant"),
            Scalar("x"),
            Scalar("linear"),
            Scalar("quadratic"),
            Scalar("cubic"),
        ],
        lambda constant, x, linear, quadratic, cubic: (
            constant + linear * x + quadratic * x**2 + cubic * x**3
        ),
        backend=optimiser_backend,
        name="response",
    )
    response_space = FunctionSpace(
        "response", arguments=[Scalar("x")], output=[Scalar("y")]
    )
    declaration = BoundFunctionParameter(
        cubic,
        {
            4: UnboundedVariable("cubic_coefficient"),
            0: UnboundedVariable("constant_coefficient"),
            3: UnboundedVariable("quadratic_coefficient"),
            2: UnboundedVariable("linear_coefficient"),
        },
    )
    fit_points = np.array([-1.0, 0.0, 1.0, 2.0])

    with ProblemBuilder() as builder:
        response = builder.new_function_parameter(response_space, declaration)
        residuals = [response(x) - x**3 for x in fit_points]
        builder.objective = Minimise(
            sum(residual * residual for residual in residuals)
        )
        builder.outputs = [response(0.5)]
        program = builder.build(optimiser_backend)

    objective, fitted_value = program()
    fitted = program.parameters["response"]

    assert isinstance(fitted, FittedFunction)
    assert objective == pytest.approx(0.0, abs=1e-6)
    assert fitted_value == pytest.approx(0.5**3, abs=1e-6)
    evaluation_points = np.linspace(-1.0, 2.0, 13)
    np.testing.assert_allclose(
        [fitted(point) for point in evaluation_points],
        evaluation_points**3,
        atol=1e-5,
    )
    assert program.solve_info is not None
    assert program.solve_info.success


def test_bound_function_parameter_keeps_each_unbound_input_public():
    source = function(
        [
            Scalar("gain"),
            Scalar("x"),
            Scalar("y"),
            Scalar("offset"),
        ],
        lambda gain, x, y, offset: gain * x + y + offset,
        backend="numpy",
        name="affine",
    )
    target = FunctionSpace(
        "affine",
        arguments=[Scalar("x"), Scalar("y")],
        output=[Scalar("value")],
    )
    declaration = BoundFunctionParameter(
        source,
        [
            (3, UnboundedVariable("offset")),
            (0, UnboundedVariable("gain")),
        ],
    )

    parameterised = declaration.build_function(target, "numpy")

    assert parameterised(2.0, 3.0, 4.0, -1.0) == pytest.approx(10.0)
