import importlib.util

import pytest

import numpy as np

from coker import FunctionSpace, Scalar, VectorSpace
from coker.algebra.ops import Noop
from coker.dynamics import (
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
            parameters=[function_parameter, Scalar("p_1")],
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda _t, x, _z, _u, p: p[0](x) + p[1],
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        )
    )

    assert isinstance(system.parameters, tuple)

    np.testing.assert_allclose(
        system.dxdt(0.0, np.array([2.0]), None, None, lambda x: x * 3, 1.0),
        np.array([7.0]),
    )


def test_builder_specializes_function_parameter_to_numeric_decisions():
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
    ) as builder:
        problem = builder.build(
            Minimise(builder.output(builder.t_final)[0] ** 2)
        )

    assert problem.system.parameters.dimension == 4
    assert [parameter.name for parameter in problem.parameters] == [
        "p_0_theta_0",
        "p_0_theta_1",
        "p_0_theta_2",
        "p_1",
    ]


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
