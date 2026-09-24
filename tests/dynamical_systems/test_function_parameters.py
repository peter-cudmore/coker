import importlib.util

import pytest

import numpy as np

from coker import FunctionSpace, Scalar, VectorSpace, function
from coker.algebra.function import BoundCallable
from coker.algebra.ops import Noop
from coker.backends.backend import get_backend_by_name
from coker.dynamics import (
    BoundVector,
    BoundedVariable,
    ClosureParameter,
    DenseTensorVariable,
    DynamicsSpec,
    FittedFunction,
    MonotonePiecewiseLinear,
    DenseLayer,
    RadialBasisFunction,
    UnboundedVariable,
    VariationalProblem,
    VariationalProblemBuilder,
)
from coker.dynamics.variational.function_binding import ParameterValueLayout
from coker.dynamics.system import create_dynamics_from_spec
from coker.dynamics.variables import ConstantControlVariable
from coker.toolkits.codesign import Minimise


def test_system_accepts_a_function_valued_positional_parameter():
    function_parameter = FunctionSpace(
        "p_0",
        arguments=[VectorSpace("x", 1)],
        output=[VectorSpace("y", 1)],
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=[
                function_parameter,
                Scalar("p_1"),
                Scalar("p_2"),
            ],
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda _t, x, _z, _u, p: p[0](x) + p[1] + p[2],
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, p, _q: p[0](x) + p[1] + p[2],
            quadratures=Noop(),
        )
    )

    assert isinstance(system.parameters, tuple)

    np.testing.assert_allclose(
        system.dxdt(
            0.0, np.array([2.0]), None, None, lambda x: x * 3, 1.0, 2.0
        ),
        np.array([9.0]),
    )

    np.testing.assert_allclose(
        system.y(
            0.0,
            np.array([2.0]),
            None,
            None,
            lambda x: x * 3,
            1.0,
            2.0,
            None,
        ),
        np.array([9.0]),
    )


def test_function_space_contains_matching_functions():
    space = FunctionSpace(
        "rate", arguments=[Scalar("time")], output=[Scalar("rate")]
    )
    matching = function([Scalar("t")], lambda time: time, backend="numpy")
    incompatible = function(
        [VectorSpace("time", 2)], lambda time: time[0], backend="numpy"
    )
    fitted = get_backend_by_name(
        "numpy", set_current=False
    ).fit_function_parameter(
        RadialBasisFunction(centers=[0.0], width=1.0),
        space,
        [0.0, 1.0],
    )

    assert matching in space
    assert fitted in space
    assert incompatible not in space


def test_function_space_contains_bound_callable():
    space = FunctionSpace("constant", arguments=[], output=[Scalar("rate")])
    bound = BoundCallable(function([], lambda: 1.0), space, ())

    assert bound in space


def test_closure_parameter_binds_declared_scalar_decisions():
    closure = ClosureParameter.bind_callable(
        lambda gain, offset, value: gain * value + offset,
        [
            UnboundedVariable("gain", 0.0),
            UnboundedVariable("offset", 0.0),
        ],
        [Scalar("value")],
        name="affine",
    )
    target = FunctionSpace(
        "forcing", arguments=[Scalar("time")], output=[Scalar("rate")]
    )

    fitted = get_backend_by_name(
        "numpy", set_current=False
    ).fit_function_parameter(closure, target, [2.0, -1.0])

    assert closure.name == "affine"
    assert closure.list_concrete_parameters() == (
        UnboundedVariable("gain", 0.0),
        UnboundedVariable("offset", 0.0),
    )
    assert fitted(3.0) == 5.0


def test_builder_specializes_function_parameter_to_numeric_decisions():
    function_parameter = FunctionSpace(
        "p_0", arguments=[Scalar("x")], output=[Scalar("y")]
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(
                function_parameter,
                Scalar("p_1"),
                Scalar("p_2"),
            ),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (0.0, None),
            dynamics=lambda _t, x, _z, _u, p: p[0](x[0]) + p[1] + p[2],
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        )
    )
    with pytest.raises(ValueError, match="function parameter name"):
        VariationalProblemBuilder(
            system,
            t_final=1.0,
            parameters=[
                MonotonePiecewiseLinear(
                    domain_knots=[-1.0, 0.0, 1.0],
                    lower_bound=0.0,
                    upper_bound=2.0,
                ),
                BoundedVariable("p_1", -1.0, 1.0),
                BoundedVariable("p_2", -1.0, 1.0),
            ],
        )
    declaration = MonotonePiecewiseLinear(
        domain_knots=[-1.0, 0.0, 1.0],
        lower_bound=0.0,
        upper_bound=2.0,
        name="p_0",
    )
    (theta,) = declaration.list_concrete_parameters()
    assert declaration.list_concrete_parameters()[0] is theta
    assert theta.name == "p_0_theta"
    assert isinstance(theta, DenseTensorVariable)
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[
            declaration,
            BoundedVariable("p_1", -1.0, 1.0),
            BoundedVariable("p_2", -1.0, 1.0),
        ],
    ) as builder:
        problem = builder.build(
            Minimise(builder.output(builder.t_final)[0] ** 2)
        )

    assert problem.system.parameters.dimension == 5
    assert [parameter.name for parameter in problem.parameters] == [
        "p_0_theta_0",
        "p_0_theta_1",
        "p_0_theta_2",
        "p_1",
        "p_2",
    ]
    assert all(
        isinstance(parameter, UnboundedVariable)
        for parameter in problem.parameters[:3]
    )


def test_builder_specializes_bound_vector_parameter():
    function_parameter = FunctionSpace(
        "p_0", arguments=[Scalar("x")], output=[Scalar("y")]
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(function_parameter, VectorSpace("gain", 2)),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (0.0, None),
            dynamics=lambda _t, x, _z, _u, p: p[0](x[0]) + p[1][0],
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        )
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[
            MonotonePiecewiseLinear(
                domain_knots=[-1.0, 0.0, 1.0],
                lower_bound=0.0,
                upper_bound=2.0,
                name="p_0",
            ),
            BoundVector(
                "gain",
                lower_bound=[-1.0, -1.0],
                upper_bound=[1.0, 1.0],
                guess=[0.0, 0.0],
            ),
        ],
    ) as builder:
        problem = builder.build(
            Minimise(builder.output(builder.t_final)[0] ** 2)
        )

    assert problem.system.parameters.dimension == 5
    assert [parameter.name for parameter in problem.parameters] == [
        "p_0_theta_0",
        "p_0_theta_1",
        "p_0_theta_2",
        "gain_0",
        "gain_1",
    ]


def test_builder_specializes_dense_tensor_parameter():

    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(VectorSpace("weights", (2, 2)),),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda _t, x, _z, _u, _p: x,
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        )
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[DenseTensorVariable("weights", guess=np.zeros((2, 2)))],
    ) as builder:
        problem = builder.build(
            Minimise(builder.output(builder.t_final)[0] ** 2)
        )

    assert problem.system.parameters.dimension == 4
    assert [parameter.name for parameter in problem.parameters] == [
        "weights_0",
        "weights_1",
        "weights_2",
        "weights_3",
    ]


def test_parameter_layout_reconstructs_public_values():
    response = FunctionSpace(
        "response", arguments=[Scalar("time")], output=[Scalar("rate")]
    )
    response_declaration = RadialBasisFunction(
        centers=[0.0], width=1.0, name="response"
    )
    layout = ParameterValueLayout(
        targets=(
            Scalar("offset"),
            VectorSpace("gain", 2),
            VectorSpace("weights", (2, 2)),
            response,
        ),
        declarations=(
            BoundedVariable("offset", -1.0, 1.0),
            BoundVector("gain", [-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]),
            DenseTensorVariable("weights", np.zeros((2, 2))),
            response_declaration,
        ),
        offsets=((0, 1), (1, 3), (3, 7), (7, 9)),
    )

    parameters = layout.reconstruct(np.arange(1.0, 10.0))

    assert parameters["offset"] == 1.0
    np.testing.assert_array_equal(parameters["gain"], [2.0, 3.0])
    np.testing.assert_array_equal(
        parameters["weights"], [[4.0, 5.0], [6.0, 7.0]]
    )
    fitted = parameters["response"]
    assert isinstance(fitted, FittedFunction)
    assert fitted.specification is response_declaration
    assert fitted.space is response
    np.testing.assert_array_equal(fitted.parameters[0], [8.0, 9.0])
    assert fitted(0.0) == pytest.approx(17.0)


def test_variational_lowers_output_loss_with_control_input(
    variational_backend,
):
    control = FunctionSpace("u", arguments=[Scalar("t")], output=[Scalar("u")])
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=control,
            parameters=None,
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (0.0, None),
            dynamics=lambda _t, x, _z, _u, _p: x * 0,
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        ),
        backend=variational_backend,
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        control=[
            ConstantControlVariable("u", upper_bound=0.0, lower_bound=0.0)
        ],
        backend=variational_backend,
    ) as builder:
        problem = builder.build(
            Minimise(builder.output(builder.t_final)[0] ** 2)
        )

    assert problem().cost == pytest.approx(0.0)


def test_variational_fits_bound_vector_parameter(variational_backend):
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(VectorSpace("gain", 1),),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (0.0, None),
            dynamics=lambda _t, _x, _z, _u, p: p[0],
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        ),
        backend=variational_backend,
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[
            BoundVector(
                "gain",
                lower_bound=[-1.0],
                upper_bound=[1.0],
                guess=[0.0],
            )
        ],
        backend=variational_backend,
    ) as builder:
        problem = builder.build(
            Minimise((builder.output(builder.t_final)[0] - 0.5) ** 2)
        )

    solution = problem()
    np.testing.assert_allclose(solution.parameters["gain"], [0.5], atol=1e-2)
    assert not hasattr(solution, "parameter_blocks")
    assert not hasattr(solution, "parameter_solutions")


@pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None, reason="CasADi not available"
)
def test_casadi_fits_monotone_function_parameter():
    function_parameter = FunctionSpace(
        "p_0", arguments=[Scalar("x")], output=[Scalar("y")]
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(function_parameter, Scalar("p_1")),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (0.0, None),
            dynamics=lambda _t, x, _z, _u, p: p[0](x[0]) + p[1],
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        )
    )
    declaration = MonotonePiecewiseLinear(
        domain_knots=[-1.0, 0.0, 1.0],
        lower_bound=0.0,
        upper_bound=2.0,
        name="p_0",
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[declaration, BoundedVariable("p_1", -1.0, 1.0)],
        backend="casadi",
    ) as builder:
        problem = builder.build(
            Minimise((builder.output(builder.t_final)[0] - 0.5) ** 2)
        )

    solution = problem.get_solver("casadi").solve()

    assert solution.solve_info.success
    assert solution.cost < 1e-4

    f = solution.parameters["p_0"]

    assert isinstance(f, FittedFunction)
    assert f.specification is declaration
    assert f.space is function_parameter
    assert len(f.parameters) == 1
    np.testing.assert_equal(f.parameters[0].shape, (declaration.size,))
    assert callable(f.function)
    assert isinstance(f(0.0), float)


def test_monotone_function_rejects_unimplemented_explicit_constraints():
    with pytest.raises(TypeError, match="constraint_mode"):
        MonotonePiecewiseLinear(
            domain_knots=[-1.0, 1.0],
            lower_bound=0.0,
            upper_bound=2.0,
            constraint_mode="explicit",
        )


def test_dense_layer_declares_and_evaluates_scalar_parameters():
    activation = function([Scalar("hidden")], lambda hidden: hidden)
    target = FunctionSpace(
        "response", arguments=[Scalar("state")], output=[Scalar("rate")]
    )
    declaration = DenseLayer(1, activation, name="response")
    weight, bias = declaration.list_concrete_parameters()

    assert isinstance(weight, UnboundedVariable)
    assert isinstance(bias, UnboundedVariable)
    assert weight.name == "response_weight"
    assert bias.name == "response_bias"
    assert declaration.validate_target(target) is target
    assert declaration.evaluate((2.0, -1.0), 3.0) == 5.0
    assert declaration.build_function(target, "numpy")(3.0, 2.0, -1.0) == 5.0


def test_dense_layer_declares_and_evaluates_vector_parameters():
    activation = function(
        [VectorSpace("hidden", 2)],
        lambda hidden: hidden,
    )
    target = FunctionSpace(
        "response",
        arguments=[VectorSpace("state", 2)],
        output=[VectorSpace("rate", 2)],
    )
    declaration = DenseLayer(2, activation, name="response")
    weights, bias = declaration.list_concrete_parameters()
    assert declaration.list_concrete_parameters() == (weights, bias)
    assert declaration.list_concrete_parameters()[0] is weights
    assert declaration.list_concrete_parameters()[1] is bias

    assert isinstance(weights, DenseTensorVariable)
    assert isinstance(bias, DenseTensorVariable)
    assert weights.name == "response_weights"
    assert weights.shape == (2, 2)
    assert bias.name == "response_bias"
    assert bias.shape == (2,)
    assert declaration.validate_target(target) is target

    parameters = (
        np.array([[1.0, -1.0], [0.5, 0.25]]),
        np.array([0.0, 0.25]),
    )
    expected = np.array([1.0, 1.5])
    np.testing.assert_allclose(
        declaration.evaluate(parameters, np.array([2.0, 1.0])), expected
    )
    np.testing.assert_allclose(
        declaration.build_function(target, "numpy")(
            np.array([2.0, 1.0]), *parameters
        ),
        expected,
    )
    bound_activation = BoundCallable(
        activation,
        FunctionSpace(
            "activation",
            arguments=[VectorSpace("hidden", 2)],
            output=[VectorSpace("rate", 2)],
        ),
        (),
    )
    np.testing.assert_allclose(
        DenseLayer(2, bound_activation).evaluate(
            parameters, np.array([2.0, 1.0])
        ),
        expected,
    )
    with pytest.raises(ValueError, match="output must have width 2"):
        declaration.validate_target(
            FunctionSpace(
                "response",
                arguments=[VectorSpace("state", 2)],
                output=[VectorSpace("rate", 1)],
            )
        )
    with pytest.raises(ValueError, match="argument must have width 2"):
        declaration.validate_target(
            FunctionSpace(
                "response",
                arguments=[VectorSpace("state", 1)],
                output=[VectorSpace("rate", 2)],
            )
        )


def test_radial_basis_function_evaluates_scalar_input():
    declaration = RadialBasisFunction(
        centers=[-1.0, 1.0],
        width=0.5,
        guess=[2.0, -1.0, 0.25],
        name="response",
    )
    (coefficients,) = declaration.list_concrete_parameters()
    assert declaration.list_concrete_parameters()[0] is coefficients

    assert coefficients.name == "response_coefficients"
    expected = (
        0.25
        + 2.0 * np.exp(-0.5 * ((0.0 + 1.0) / 0.5) ** 2)
        - np.exp(-0.5 * ((0.0 - 1.0) / 0.5) ** 2)
    )
    assert declaration.evaluate((coefficients.guess,), 0.0) == pytest.approx(
        expected
    )


@pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None, reason="CasADi not available"
)
@pytest.mark.parametrize(
    ("declaration", "space", "basis", "argument", "expected"),
    [
        (
            DenseLayer(
                2,
                function(
                    [VectorSpace("hidden", 1)],
                    lambda hidden: hidden,
                ),
            ),
            VectorSpace("x", 2),
            (np.array([[1.0, -1.0]]), np.array([0.0])),
            np.array([2.0, 1.0]),
            np.array([1.0]),
        ),
        (
            RadialBasisFunction([0.0], 1.0),
            Scalar("x"),
            (np.array([2.0, 0.5]),),
            0.0,
            2.5,
        ),
    ],
)
def test_function_declarations_lower_to_casadi(
    declaration, space, basis, argument, expected
):
    compiled = function(
        [space],
        lambda value: declaration.evaluate(basis, value),
        backend="casadi",
    )

    actual = compiled(argument)

    np.testing.assert_allclose(
        np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
    )


@pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None, reason="CasADi not available"
)
def test_variational_problem_specializes_radial_basis_parameter():
    function_parameter = FunctionSpace(
        "response", arguments=[Scalar("time")], output=[Scalar("rate")]
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(function_parameter,),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (0.0, None),
            dynamics=lambda time, _x, _z, _u, p: p[0](time),
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        ),
        backend="casadi",
    )
    problem = VariationalProblem(
        loss=lambda solution, parameters: (solution(1.0, parameters)[0] - 0.5)
        ** 2,
        system=system,
        parameters=[
            RadialBasisFunction(centers=[0.0], width=1.0, name="response")
        ],
        t_final=1.0,
        backend="casadi",
    )

    solution = problem()

    assert solution.solve_info.success
    assert solution.cost < 1e-4


def _function_parameter_system(rate: FunctionSpace, backend: str):
    return create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(Scalar("offset"), rate),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda time, _state, _z, _u, parameters: parameters[0]
            + parameters[1](time),
            constraints=Noop(),
            outputs=lambda _t, state, _z, _u, _p, _q: state,
            quadratures=Noop(),
        ),
        backend=backend,
    )


@pytest.mark.parametrize(
    "integration_backend",
    [
        "numpy",
        pytest.param(
            "casadi",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("casadi") is None,
                reason="CasADi not available",
            ),
        ),
        pytest.param(
            "pytorch",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("torchdiffeq") is None,
                reason="torchdiffeq not available",
            ),
        ),
    ],
)
@pytest.mark.parametrize("source", ("callable", "function", "fitted"))
def test_system_integrates_function_valued_parameter(
    integration_backend, source
):
    rate = FunctionSpace(
        "rate",
        arguments=[Scalar("t")],
        output=[Scalar("rate")],
    )
    system = _function_parameter_system(rate, integration_backend)

    def affine_rate(time):
        return 1.0 + time

    def affine_integral(time):
        return 3 * time + time**2 / 2

    def constant_integral(time):
        return 3 * time

    if source == "callable":
        rate_value = affine_rate
        integral = affine_integral
    elif source == "function":
        rate_value = function(
            [Scalar("time")],
            affine_rate,
            backend=integration_backend,
        )
        integral = affine_integral
    else:
        rate_value = get_backend_by_name(
            "numpy", set_current=False
        ).fit_function_parameter(
            RadialBasisFunction(centers=[0.0], width=1.0),
            rate,
            [0.0, 1.0],
        )
        integral = constant_integral

    timeline = np.array([0.0, 0.5, 1.0])
    trajectory = system(timeline, 2.0, rate_value)

    np.testing.assert_allclose(
        trajectory.reshape(-1), integral(timeline), rtol=1e-5
    )
    np.testing.assert_allclose(
        system(1.0, 2.0, rate_value), integral(1.0), rtol=1e-5
    )


def test_input_system_integrates_heterogeneous_parameters():
    input_signal = FunctionSpace(
        "input",
        arguments=[Scalar("t")],
        output=[Scalar("u")],
    )
    rate = FunctionSpace(
        "rate",
        arguments=[Scalar("t")],
        output=[Scalar("rate")],
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=input_signal,
            parameters=(Scalar("gain"), rate),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda time, _state, _z, input_value, parameters: (
                parameters[0] * input_value(time) + parameters[1](time)
            ),
            constraints=Noop(),
            outputs=lambda _t, state, _z, _u, _p, _q: state,
            quadratures=Noop(),
        ),
        backend="numpy",
    )

    timeline = np.array([0.0, 0.5, 1.0])
    trajectory = system(timeline, lambda _time: 2.0, 3.0, lambda time: time)

    np.testing.assert_allclose(
        trajectory.reshape(-1), 6 * timeline + timeline**2 / 2, rtol=1e-5
    )


def test_direct_function_parameters_reject_undeclared_input_sentinel():
    rate = FunctionSpace(
        "rate",
        arguments=[Scalar("t")],
        output=[Scalar("rate")],
    )
    system = _function_parameter_system(rate, "numpy")

    with pytest.raises(ValueError, match=r"expected 3, got 4"):
        system(1.0, Noop(), 2.0, lambda _time: 1.0)


def test_direct_function_parameters_validate_the_declared_space():
    rate = FunctionSpace(
        "rate",
        arguments=[Scalar("t")],
        output=[Scalar("rate")],
    )
    system = _function_parameter_system(rate, "numpy")
    incompatible_rate = function(
        [VectorSpace("time", 2)],
        lambda time: time[0],
        backend="numpy",
    )

    with pytest.raises(
        ValueError,
        match=r"Function-valued parameter 1 does not match declared "
        r"FunctionSpace 'rate'",
    ):
        system(1.0, 2.0, incompatible_rate)
