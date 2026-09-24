from contextlib import contextmanager

import numpy as np
import pytest
import torch

import warnings

from coker import FunctionSpace, Scalar, VectorSpace, function
from coker.algebra.function import BoundCallable
from coker.algebra.ops import Noop
from coker.backends import get_backend_by_name
from coker.dynamics import (
    DynamicsSpec,
    VariationalProblemBuilder,
    direct_sum,
)
from coker.parameters.function_parameters import (
    ClosureParameter,
    DenseLayer,
    FittedFunction,
    RadialBasisFunction,
)
from coker.dynamics.system import create_dynamics_from_spec
from coker.toolkits.codesign import Minimise
from coker.parameters import BoundedVariable, BoundVector


def _identity_activation(backend, width=1):
    return function(
        [VectorSpace("hidden", width)], lambda hidden: hidden, backend=backend
    )


def _relu_scalar_activation(backend):
    signature_source = function(
        [Scalar("hidden")],
        lambda hidden: hidden,
        backend="pytorch",
    )
    return backend.import_function(torch.relu, signature_source.signature)


@pytest.mark.parametrize("bind_activation", (False, True))
def test_pytorch_fitted_dense_layer_preserves_nested_relu_activation(
    bind_activation,
):
    backend = get_backend_by_name("pytorch", set_current=False)
    response = FunctionSpace(
        "response",
        arguments=[Scalar("inflow")],
        output=[Scalar("rate")],
    )
    activation = _relu_scalar_activation(backend)
    if bind_activation:
        activation = BoundCallable(
            activation,
            FunctionSpace(
                "relu",
                arguments=[Scalar("hidden")],
                output=[Scalar("output")],
            ),
            (),
        )
    declaration = DenseLayer(1, activation, name="response")

    fitted = backend.fit_function_parameter(
        declaration,
        response,
        np.array([1.1796, -0.3112]),
    )

    assert isinstance(fitted, FittedFunction)
    assert fitted(torch.tensor(0.0, dtype=torch.float64)).item() == 0.0
    assert fitted(
        torch.tensor(1.0, dtype=torch.float64)
    ).item() == pytest.approx(0.8684)


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
    dense_layer = DenseLayer(2, _identity_activation("pytorch"))
    radial_basis = RadialBasisFunction([0.0], 1.0)
    vector_function = function(
        [VectorSpace("x", 2)],
        lambda x: dense_layer.evaluate(
            (np.array([[1.0, -1.0]]), np.array([0.0])), x
        ),
        backend="pytorch",
    )
    scalar_function = function(
        [VectorSpace("x", 1)],
        lambda x: radial_basis.evaluate((np.array([2.0, 0.5]),), x[0]),
        backend="pytorch",
    )

    vector = torch.tensor([2.0, 1.0], dtype=torch.float64)
    scalar = torch.tensor([0.0], dtype=torch.float64)

    assert torch.allclose(
        vector_function(vector), torch.tensor([1.0], dtype=torch.float64)
    )
    assert torch.allclose(
        scalar_function(scalar), torch.tensor(2.5, dtype=torch.float64)
    )


def test_pytorch_variational_solver_lowers_dense_layer_parameter(monkeypatch):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)
    function_parameter = FunctionSpace(
        "response",
        arguments=[VectorSpace("state", 2)],
        output=[VectorSpace("rate", 2)],
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(function_parameter,),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.zeros(2), None),
            dynamics=lambda _t, state, _z, _u, p: p[0](state),
            constraints=Noop(),
            outputs=lambda _t, state, _z, _u, _p, _q: state,
            quadratures=Noop(),
        ),
        backend="pytorch",
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[
            DenseLayer(2, _identity_activation("pytorch", 2), name="response")
        ],
        backend="pytorch",
    ) as builder:
        problem = builder.build(
            Minimise((builder.output(builder.t_final)[0] - 0.5) ** 2)
        )

    solution = problem()

    assert solution.cost < 1e-4


def test_pytorch_solver_reconstructs_dense_relu_function_parameter(
    monkeypatch,
):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)
    response = FunctionSpace(
        "response",
        arguments=[Scalar("inflow")],
        output=[Scalar("rate")],
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(response,),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (0.0, None),
            dynamics=lambda _t, state, _z, _u, p: p[0](state[0]),
            constraints=Noop(),
            outputs=lambda _t, state, _z, _u, _p, _q: state,
            quadratures=Noop(),
        ),
        backend="pytorch",
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[
            DenseLayer(1, _relu_scalar_activation(backend), name="response")
        ],
        backend="pytorch",
    ) as builder:
        problem = builder.build(Minimise(builder.output(builder.t_final)[0]))

    solver = problem.get_solver()
    solution = solver.solve(**dict(zip(solver.parameters, (1.1796, -0.3112))))

    fitted = solution.parameters["response"]
    assert isinstance(fitted, FittedFunction)
    assert fitted(torch.tensor(0.0, dtype=torch.float64)).item() == 0.0


def test_pytorch_solution_reconstructs_mapped_function_parameter(
    monkeypatch,
):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)
    function_parameter = FunctionSpace(
        "response",
        arguments=[VectorSpace("state", 2)],
        output=[VectorSpace("rate", 1)],
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
        parameters=[
            DenseLayer(2, _identity_activation("pytorch"), name="response")
        ],
        system_parameter_map=parameter_map,
        backend="pytorch",
    ) as builder:
        problem = builder.build(Minimise(builder.output(builder.t_final)[0]))

    raw_basis = np.array([0.1, -0.2, 0.3])
    solver = problem.get_solver()
    solution = solver.solve(**dict(zip(solver.parameters, raw_basis)))

    argument = np.array([0.7, -0.4])
    mapped_basis = parameter_map @ raw_basis
    expected = np.array(
        [
            mapped_basis[0] * argument[0]
            + mapped_basis[1] * argument[1]
            + mapped_basis[2]
        ]
    )
    raw_value = (
        raw_basis[0] * argument[0] + raw_basis[1] * argument[1] + raw_basis[2]
    )

    fitted = solution.parameters["response"]
    assert isinstance(fitted, FittedFunction)
    assert isinstance(fitted.parameters, tuple)
    assert [parameter.shape for parameter in fitted.parameters] == [
        (1, 2),
        (1,),
    ]
    assert callable(fitted.function)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        actual = fitted(torch.as_tensor(argument, dtype=torch.float64))
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected)
    assert not np.isclose(expected[0], raw_value)


def _quadratic_rate_declaration(*, upper_bound=2.0):
    return ClosureParameter.bind_callable(
        lambda scale, inflow: scale * inflow**2,
        [
            BoundedVariable(
                "rate_scale",
                lower_bound=0.0,
                upper_bound=upper_bound,
                guess=0.5,
            )
        ],
        [Scalar("inflow")],
        name="rate",
        backend="pytorch",
    )


def _two_batch_transfer_system():
    rate = FunctionSpace(
        "rate",
        arguments=[Scalar("inflow")],
        output=[Scalar("transfer_rate")],
    )

    def batch_system():
        return create_dynamics_from_spec(
            DynamicsSpec(
                inputs=Noop(),
                parameters=(rate, Scalar("initial_inflow")),
                algebraic=None,
                initial_conditions=lambda _z, _u, p: (
                    p[1] * np.array([1.0, 0.0]),
                    None,
                ),
                dynamics=lambda _t, state, _z, _u, p: p[0](state[0])
                * np.array([-1.0, 1.0]),
                constraints=Noop(),
                outputs=lambda _t, state, _z, _u, _p, _q: state,
                quadratures=Noop(),
            ),
            backend="pytorch",
        )

    return direct_sum(batch_system(), batch_system(), backend="pytorch")[0]


@contextmanager
def _two_batch_transfer_problem(rate_declarations, output_projection=None):
    initial_inflows = (1.0, 2.0)
    t_final = 0.2
    targets = np.concatenate(
        [
            (
                np.array(
                    [
                        inflow / (1.0 + inflow * t_final),
                        inflow - inflow / (1.0 + inflow * t_final),
                    ]
                )
            )
            for inflow in initial_inflows
        ]
    )
    fixed_initials = [
        BoundedVariable(
            f"initial_inflow_{index}",
            lower_bound=inflow,
            upper_bound=inflow,
            guess=inflow,
        )
        for index, inflow in enumerate(initial_inflows)
    ]
    with VariationalProblemBuilder(
        _two_batch_transfer_system(),
        t_final=t_final,
        parameters=[
            rate_declarations[0],
            fixed_initials[0],
            rate_declarations[1],
            fixed_initials[1],
        ],
        backend="pytorch",
    ) as builder:
        terminal = builder.output(builder.t_final)
        if output_projection is not None:
            terminal = output_projection @ terminal
            targets = output_projection @ targets
        residual = terminal - targets
        problem = builder.build(
            Minimise(
                sum(
                    residual[index] * residual[index]
                    for index in range(targets.size)
                )
            )
        )
    yield problem, initial_inflows, targets


def test_pytorch_fits_shared_function_parameter_across_fixed_batches(
    monkeypatch,
):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)
    shared_rate = _quadratic_rate_declaration()

    with _two_batch_transfer_problem((shared_rate, shared_rate)) as (
        problem,
        initial_inflows,
        targets,
    ):
        solution = problem()

        assert solution.solve_info.success
        assert solution.cost < 1e-6
        np.testing.assert_allclose(solution.state(0.2), targets, atol=2e-3)
        fitted = solution.parameters["rate"]
        assert isinstance(fitted, FittedFunction)
        torch.testing.assert_close(
            torch.stack(
                [
                    fitted(torch.tensor(inflow, dtype=torch.float64))
                    for inflow in initial_inflows
                ]
            ),
            torch.tensor(
                [inflow**2 for inflow in initial_inflows],
                dtype=torch.float64,
            ),
            rtol=0,
            atol=2e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pytorch_cuda_fits_shared_function_parameter_with_projections(
    monkeypatch,
):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cuda"))
    monkeypatch.setattr(backend, "dtype", torch.float64)
    output_projection = np.array(
        [
            [1.0, 0.1, 0.2, 0.3],
            [0.2, 1.0, 0.3, 0.1],
            [0.3, 0.2, 1.0, 0.1],
            [0.1, 0.3, 0.2, 1.0],
        ]
    )
    shared_rate = _quadratic_rate_declaration()

    with _two_batch_transfer_problem(
        (shared_rate, shared_rate),
        output_projection=output_projection,
    ) as (problem, _, _):
        solution = problem()

    assert solution.solve_info.success
    assert np.isfinite(solution.cost)


def test_pytorch_shares_compatible_duplicate_function_declarations(
    monkeypatch,
):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)

    with _two_batch_transfer_problem(
        (_quadratic_rate_declaration(), _quadratic_rate_declaration())
    ) as (problem, _, targets):
        solution = problem.get_solver().solve(rate_scale=1.0)

        assert solution.solve_info.success
        np.testing.assert_allclose(solution.state(0.2), targets, atol=2e-3)


def test_pytorch_rejects_conflicting_duplicate_function_declarations():
    with (
        pytest.raises(
            ValueError, match="(?i)conflicting concrete parameter declarations"
        ),
        _two_batch_transfer_problem(
            (
                _quadratic_rate_declaration(upper_bound=2.0),
                _quadratic_rate_declaration(upper_bound=3.0),
            )
        ),
    ):
        pass
