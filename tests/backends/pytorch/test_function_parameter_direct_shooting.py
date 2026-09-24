from contextlib import contextmanager

import numpy as np
import torch

from coker import FunctionSpace, Scalar, VectorSpace, function
from coker.algebra.ops import Noop
from coker.backends import get_backend_by_name
from coker.dynamics import DynamicsSpec, VariationalProblemBuilder
from coker.dynamics.system import create_dynamics_from_spec
from coker.parameters import DenseTensorVariable, UnboundedVariable
from coker.parameters.function_parameters import (
    ClosureParameter,
    DenseLayer,
    FittedFunction,
    MonotonePiecewiseLinear,
    RadialBasisFunction,
)
from coker.toolkits.codesign import Minimise


def _use_cpu_float64(monkeypatch):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)


def _identity_activation(width):
    return function(
        [VectorSpace("hidden", width)],
        lambda hidden: hidden,
        backend="pytorch",
    )


def _scalar_identity_activation():
    return function(
        [Scalar("hidden")],
        lambda hidden: hidden,
        backend="pytorch",
    )


@contextmanager
def _scalar_response_problem(declaration, arguments, targets):
    response = FunctionSpace(
        "response",
        arguments=[Scalar("argument")],
        output=[Scalar("rate")],
    )
    basis = np.eye(len(arguments))
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(response,),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (
                np.zeros(len(arguments)),
                None,
            ),
            dynamics=lambda _t, _state, _z, _u, p: sum(
                (
                    p[0](argument) * basis[index]
                    for index, argument in enumerate(arguments)
                ),
                np.zeros(len(arguments)),
            ),
            constraints=Noop(),
            outputs=lambda _t, state, _z, _u, _p, _q: state,
            quadratures=Noop(),
        ),
        backend="pytorch",
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[declaration],
        backend="pytorch",
    ) as builder:
        terminal = builder.output(builder.t_final)
        residual = terminal - np.asarray(targets)
        problem = builder.build(
            Minimise(
                sum(
                    residual[index] * residual[index]
                    for index in range(len(arguments))
                )
            )
        )
    yield problem


@contextmanager
def _vector_response_problem(declaration, arguments, targets):
    response = FunctionSpace(
        "response",
        arguments=[VectorSpace("argument", 2)],
        output=[VectorSpace("rate", 2)],
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(response,),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (
                np.zeros(2 * len(arguments)),
                None,
            ),
            dynamics=lambda _t, _state, _z, _u, p: np.concatenate(
                [p[0](argument) for argument in arguments]
            ),
            constraints=Noop(),
            outputs=lambda _t, state, _z, _u, _p, _q: state,
            quadratures=Noop(),
        ),
        backend="pytorch",
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[declaration],
        backend="pytorch",
    ) as builder:
        terminal = builder.output(builder.t_final)
        residual = terminal - np.asarray(targets).reshape(-1)
        problem = builder.build(
            Minimise(
                sum(
                    residual[index] * residual[index]
                    for index in range(2 * len(arguments))
                )
            )
        )
    yield problem


def _assert_scalar_response(fitted, arguments, targets, *, atol=2e-3):
    assert isinstance(fitted, FittedFunction)
    assert callable(fitted)
    actual = torch.stack(
        [
            fitted(torch.tensor(argument, dtype=torch.float64))
            for argument in arguments
        ]
    )
    torch.testing.assert_close(
        actual,
        torch.tensor(targets, dtype=torch.float64),
        rtol=0,
        atol=atol,
    )


def _assert_vector_response(fitted, arguments, targets, *, atol=2e-3):
    assert isinstance(fitted, FittedFunction)
    assert callable(fitted)
    actual = torch.stack(
        [
            fitted(torch.tensor(argument, dtype=torch.float64))
            for argument in arguments
        ]
    )
    torch.testing.assert_close(
        actual,
        torch.tensor(targets, dtype=torch.float64),
        rtol=0,
        atol=atol,
    )


def test_pytorch_direct_shooting_optimises_scalar_dense_function_parameter(
    monkeypatch,
):
    _use_cpu_float64(monkeypatch)
    arguments = (-1.0, 1.0)
    targets = (-0.25, 0.75)
    declaration = DenseLayer(
        1,
        _scalar_identity_activation(),
        name="response",
    )

    with _scalar_response_problem(declaration, arguments, targets) as problem:
        solution = problem.get_solver("pytorch").solve()

    assert solution.solve_info.success
    assert solution.cost < 1e-4
    fitted = solution.parameters["response"]
    assert isinstance(fitted.parameters, tuple)
    assert [parameter.shape for parameter in fitted.parameters] == [(), ()]
    _assert_scalar_response(fitted, arguments, targets)


def test_pytorch_direct_shooting_optimises_rbf_function_parameter(monkeypatch):
    _use_cpu_float64(monkeypatch)
    arguments = (0.0, 1.0)
    targets = (1.5, 0.5)
    declaration = RadialBasisFunction(
        centers=[0.0],
        width=1.0,
        name="response",
    )

    with _scalar_response_problem(declaration, arguments, targets) as problem:
        solution = problem.get_solver("pytorch").solve()

    assert solution.solve_info.success
    assert solution.cost < 1e-4
    fitted = solution.parameters["response"]
    assert isinstance(fitted.parameters, tuple)
    assert [parameter.shape for parameter in fitted.parameters] == [(2,)]
    _assert_scalar_response(fitted, arguments, targets)


def test_pytorch_direct_shooting_optimises_monotone_function_parameter(
    monkeypatch,
):
    _use_cpu_float64(monkeypatch)
    arguments = (-1.0, 0.0, 1.0)
    targets = (0.5, 1.0, 1.5)
    declaration = MonotonePiecewiseLinear(
        domain_knots=arguments,
        lower_bound=0.0,
        upper_bound=2.0,
        name="response",
    )

    with _scalar_response_problem(declaration, arguments, targets) as problem:
        solution = problem.get_solver("pytorch").solve()

    assert solution.solve_info.success
    assert solution.cost < 1e-4
    fitted = solution.parameters["response"]
    assert isinstance(fitted.parameters, tuple)
    assert [parameter.shape for parameter in fitted.parameters] == [(3,)]
    _assert_scalar_response(fitted, arguments, targets)
    values = torch.stack(
        [
            fitted(torch.tensor(argument, dtype=torch.float64))
            for argument in arguments
        ]
    )
    assert torch.all(values >= 0.0)
    assert torch.all(values <= 2.0)
    assert torch.all(values[1:] >= values[:-1])


def test_pytorch_direct_shooting_optimises_scalar_closure_function_parameter(
    monkeypatch,
):
    _use_cpu_float64(monkeypatch)
    arguments = (-1.0, 2.0)
    targets = (-0.5, 1.0)
    declaration = ClosureParameter.bind_callable(
        lambda gain, offset, argument: gain * argument + offset,
        [
            UnboundedVariable("response_gain", guess=0.0),
            UnboundedVariable("response_offset", guess=0.0),
        ],
        [Scalar("argument")],
        name="response",
        backend="pytorch",
    )

    with _scalar_response_problem(declaration, arguments, targets) as problem:
        solution = problem.get_solver("pytorch").solve()

    assert solution.solve_info.success
    assert solution.cost < 1e-4
    fitted = solution.parameters["response"]
    assert isinstance(fitted.parameters, tuple)
    assert [parameter.shape for parameter in fitted.parameters] == [(), ()]
    _assert_scalar_response(fitted, arguments, targets)


def test_pytorch_direct_shooting_optimises_vector_dense_function_parameter(
    monkeypatch,
):
    _use_cpu_float64(monkeypatch)
    arguments = (
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    targets = ((1.0, 0.5), (-0.5, 1.0), (0.0, 0.0))
    declaration = DenseLayer(
        2,
        _identity_activation(2),
        name="response",
    )

    with _vector_response_problem(declaration, arguments, targets) as problem:
        solution = problem.get_solver("pytorch").solve()

    assert solution.solve_info.success
    assert solution.cost < 1e-4
    fitted = solution.parameters["response"]
    assert isinstance(fitted.parameters, tuple)
    assert [parameter.shape for parameter in fitted.parameters] == [
        (2, 2),
        (2,),
    ]
    _assert_vector_response(fitted, arguments, targets)


def test_pytorch_direct_shooting_preserves_bounded_rbf_block_bounds(
    monkeypatch,
):
    _use_cpu_float64(monkeypatch)
    arguments = (-1.0, 0.0, 1.0)
    coefficients = (0.5, 0.25, 0.1)
    targets = tuple(
        coefficients[0] * np.exp(-0.5 * (argument + 1.0) ** 2)
        + coefficients[1] * np.exp(-0.5 * argument**2)
        + coefficients[2]
        for argument in arguments
    )
    declaration = RadialBasisFunction(
        centers=[-1.0, 0.0],
        width=1.0,
        lower_bound=-0.25,
        upper_bound=0.75,
        guess=[0.0, 0.0, 0.0],
        name="response",
    )

    with _scalar_response_problem(declaration, arguments, targets) as problem:
        solution = problem.get_solver("pytorch").solve()

    assert solution.solve_info.success
    assert solution.cost < 1e-4
    fitted = solution.parameters["response"]
    assert isinstance(fitted.parameters, tuple)
    assert [parameter.shape for parameter in fitted.parameters] == [(3,)]
    assert torch.all(fitted.parameters[0] >= -0.25)
    assert torch.all(fitted.parameters[0] <= 0.75)
    _assert_scalar_response(fitted, arguments, targets)


def test_pytorch_direct_shooting_restores_dense_closure_block_shape(
    monkeypatch,
):
    _use_cpu_float64(monkeypatch)
    arguments = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    targets = ((0.75, -0.25), (0.5, 1.25))
    declaration = ClosureParameter.bind_callable(
        lambda matrix, argument: matrix @ argument,
        [DenseTensorVariable("response_matrix", np.zeros((2, 2)))],
        [VectorSpace("argument", 2)],
        name="response",
        backend="pytorch",
    )

    with _vector_response_problem(declaration, arguments, targets) as problem:
        solution = problem.get_solver("pytorch").solve()

    assert solution.solve_info.success
    assert solution.cost < 1e-4
    fitted = solution.parameters["response"]
    assert isinstance(fitted.parameters, tuple)
    assert [parameter.shape for parameter in fitted.parameters] == [(2, 2)]
    _assert_vector_response(fitted, arguments, targets)


def test_pytorch_direct_shooting_keeps_fixed_flattened_function_decision(
    monkeypatch,
):
    _use_cpu_float64(monkeypatch)
    arguments = (
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    targets = ((0.7, -0.05), (-0.1, 0.2), (0.5, 0.45))
    declaration = DenseLayer(
        2,
        _identity_activation(2),
        name="response",
    )

    with _vector_response_problem(declaration, arguments, targets) as problem:
        solution = problem.get_solver("pytorch").solve(response_weights_0=0.6)

    assert solution.solve_info.success
    assert solution.cost < 1e-4
    fitted = solution.parameters["response"]
    assert isinstance(fitted.parameters, tuple)
    assert [parameter.shape for parameter in fitted.parameters] == [
        (2, 2),
        (2,),
    ]
    torch.testing.assert_close(
        fitted.parameters[0][0, 0],
        torch.tensor(0.6, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    _assert_vector_response(fitted, arguments, targets)
