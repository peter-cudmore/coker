import numpy as np
import torch

import warnings

from coker import FunctionSpace, Scalar, VectorSpace, function
from coker.algebra.ops import Noop
from coker.backends import get_backend_by_name
from coker.dynamics import (
    BoundVector,
    DynamicsSpec,
    FittedFunction,
    Perceptron,
    RadialBasisFunction,
    VariationalProblemBuilder,
)
from coker.dynamics.system import create_dynamics_from_spec
from coker.toolkits.codesign import Minimise


def test_pytorch_fits_bound_vector_parameter(monkeypatch):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)
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
        backend="pytorch",
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
        backend="pytorch",
    ) as builder:
        problem = builder.build(
            Minimise((builder.output(builder.t_final)[0] - 0.5) ** 2)
        )

    solution = problem()
    np.testing.assert_allclose(solution.parameters["gain"], [0.5], atol=1e-2)
    assert not hasattr(solution, "parameter_blocks")


def test_pytorch_lowers_function_parameter_declarations():
    perceptron = Perceptron(2)
    radial_basis = RadialBasisFunction([0.0], 1.0)
    vector_function = function(
        [VectorSpace("x", 2)],
        lambda x: perceptron.evaluate([1.0, -1.0, 0.0], x),
        backend="pytorch",
    )
    scalar_function = function(
        [VectorSpace("x", 1)],
        lambda x: radial_basis.evaluate([2.0, 0.5], x[0]),
        backend="pytorch",
    )

    vector = torch.tensor([2.0, 1.0], dtype=torch.float64)
    scalar = torch.tensor([0.0], dtype=torch.float64)

    assert torch.allclose(
        vector_function(vector),
        torch.tensor(1.0 / (1.0 + np.exp(-1.0)), dtype=torch.float64),
    )
    assert torch.allclose(
        scalar_function(scalar), torch.tensor(2.5, dtype=torch.float64)
    )


def test_pytorch_variational_solver_lowers_perceptron_parameter(monkeypatch):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)
    function_parameter = FunctionSpace(
        "response",
        arguments=[VectorSpace("state", 2)],
        output=[Scalar("rate")],
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(function_parameter,),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.zeros(2), None),
            dynamics=lambda _t, state, _z, _u, p: np.ones(2) * p[0](state),
            constraints=Noop(),
            outputs=lambda _t, state, _z, _u, _p, _q: state,
            quadratures=Noop(),
        ),
        backend="pytorch",
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[Perceptron(2, name="response")],
        backend="pytorch",
    ) as builder:
        problem = builder.build(
            Minimise((builder.output(builder.t_final)[0] - 0.5) ** 2)
        )

    solution = problem()

    assert solution.cost < 1e-4


def test_pytorch_solution_reconstructs_mapped_function_parameter(
    monkeypatch,
):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)
    function_parameter = FunctionSpace(
        "response",
        arguments=[VectorSpace("state", 2)],
        output=[Scalar("rate")],
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(function_parameter,),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.zeros(2), None),
            dynamics=lambda _t, _state, _z, _u, _p: np.zeros(2),
            constraints=Noop(),
            outputs=lambda _t, state, _z, _u, _p, _q: state,
            quadratures=Noop(),
        ),
        backend="pytorch",
    )
    parameter_map = np.diag([2.0, 3.0, 4.0])
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[Perceptron(2, name="response")],
        system_parameter_map=parameter_map,
        backend="pytorch",
    ) as builder:
        problem = builder.build(Minimise(builder.output(builder.t_final)[0]))

    raw_basis = np.array([0.1, -0.2, 0.3])
    solver = problem.get_solver()
    solution = solver.solve(**dict(zip(solver.parameters, raw_basis)))

    argument = np.array([0.7, -0.4])
    mapped_basis = parameter_map @ raw_basis
    expected = 1.0 / (
        1.0
        + np.exp(
            -(
                mapped_basis[0] * argument[0]
                + mapped_basis[1] * argument[1]
                + mapped_basis[2]
            )
        )
    )
    raw_value = 1.0 / (
        1.0
        + np.exp(
            -(
                raw_basis[0] * argument[0]
                + raw_basis[1] * argument[1]
                + raw_basis[2]
            )
        )
    )

    fitted = solution.parameters["response"]
    assert isinstance(fitted, FittedFunction)
    assert isinstance(fitted.parameters, torch.Tensor)
    assert callable(fitted.function)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        actual = fitted(torch.as_tensor(argument, dtype=torch.float64))
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected)
    assert not np.isclose(expected, raw_value)
