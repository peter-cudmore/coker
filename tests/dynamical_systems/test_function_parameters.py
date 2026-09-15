import importlib.util

import pytest

import numpy as np

from coker import FunctionSpace, Scalar, VectorSpace
from coker.algebra.ops import Noop
from coker.dynamics import (
    BoundVector,
    DenseTensorVariable,
    BoundedVariable,
    DynamicsSpec,
    MonotonePiecewiseLinear,
    VariationalProblemBuilder,
)
from coker.dynamics.system import create_dynamics_from_spec
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
    declaration = MonotonePiecewiseLinear(
        knots=[-1.0, 0.0, 1.0], lower_bound=0.0, upper_bound=2.0
    )

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
                knots=[-1.0, 0.0, 1.0],
                lower_bound=0.0,
                upper_bound=2.0,
            ),
            BoundVector("gain", guess=[0.0, 0.0]),
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
        parameters=[BoundVector("gain", guess=[0.0])],
        backend=variational_backend,
    ) as builder:
        problem = builder.build(
            Minimise((builder.output(builder.t_final)[0] - 0.5) ** 2)
        )

    solution = problem()
    np.testing.assert_allclose(
        solution.parameter_blocks["gain"],
        [0.5],
        atol=1e-2,
    )


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
        knots=[-1.0, 0.0, 1.0], lower_bound=0.0, upper_bound=2.0
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


def test_monotone_function_rejects_unimplemented_explicit_constraints():
    with pytest.raises(TypeError, match="constraint_mode"):
        MonotonePiecewiseLinear(
            knots=[-1.0, 1.0],
            lower_bound=0.0,
            upper_bound=2.0,
            constraint_mode="explicit",
        )
