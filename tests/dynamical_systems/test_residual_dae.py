from dataclasses import replace

import numpy as np
import pytest

from coker import Dimension, FunctionSpace, Scalar, VectorSpace, function
from coker.backends.numpy import NumpyBackend
from coker.algebra.ops import Noop
from coker.dynamics import (
    DynamicsSpec,
    ResidualDynamicalSystem,
    to_residual_dynamical_system,
)
from coker.dynamics.system import create_dynamics_from_spec


def _residual_model(backend, *, F=None, x0=None, y=None):
    if F is None:
        F = function(
            [
                Scalar("t"),
                VectorSpace("w", 1),
                VectorSpace("wdot", 1),
                VectorSpace("z", 0),
                Noop(),
                None,
            ],
            lambda _t, _w, _wdot, _z, _u, _p: np.zeros(1),
            backend=backend,
        )
    if x0 is None:
        x0 = function(
            [None, Noop(), None],
            lambda _z, _u, _p: (np.array([1.0]), None),
            backend=backend,
        )
    if y is None:
        y = function(
            [Scalar("t"), VectorSpace("x", 1), None, Noop(), None, None],
            lambda _t, x, _z, _u, _p, _q: x,
            backend=backend,
        )
    return ResidualDynamicalSystem(
        inputs=Noop(),
        parameters=None,
        x0=x0,
        F=F,
        y=y,
        differential=Dimension(1),
        algebraic=None,
        quadrature=None,
    )


def test_residual_ode_rows_are_state_rate_minus_public_dynamics(backend):
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=None,
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([1.0, 2.0]), None),
            dynamics=lambda t, x, _z, _u, _p: np.array([x[0] + t, x[1] - t]),
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        ),
        backend=backend,
    )

    residual = to_residual_dynamical_system(system)

    np.testing.assert_allclose(
        residual.F(
            0.5,
            np.array([2.0, 3.0]),
            np.array([6.0, 9.0]),
            np.empty((0,)),
            None,
            None,
        ),
        np.array([3.5, 6.5]),
    )


def test_residual_rejects_invalid_residual_arity(backend):
    callback = function(
        [
            Scalar("t"),
            VectorSpace("w", 1),
            VectorSpace("wdot", 1),
            VectorSpace("z", 0),
            Noop(),
        ],
        lambda _t, _w, _wdot, _z, _u: np.zeros(1),
        backend=backend,
    )

    with pytest.raises(TypeError, match="residual callback has 5 inputs; expected 6"):
        replace(_residual_model(backend), F=callback)


def test_numpy_validates_initial_condition_dimensions_at_evaluation():
    residual = replace(
        _residual_model("numpy"),
        x0=function(
            [None, Noop(), None],
            lambda _z, _u, _p: (np.zeros(2), None),
            backend="numpy",
        ),
    )

    with pytest.raises(
        ValueError,
        match="differential initial condition has 2 values; expected 1",
    ):
        residual(0.25)


def test_numpy_defers_residual_row_count_validation_to_evaluation():
    residual = replace(
        _residual_model("numpy"),
        F=function(
            [
                Scalar("t"),
                VectorSpace("w", 1),
                VectorSpace("wdot", 1),
                VectorSpace("z", 0),
                Noop(),
                None,
            ],
            lambda _t, _w, _wdot, _z, _u, _p: np.zeros(2),
            backend="numpy",
        ),
    )

    with pytest.raises(ValueError, match="implicit residual must return one row"):
        NumpyBackend().evaluate_integrals(
            residual,
            [np.array([1.0]), None, None],
            0.25,
            [None, None],
        )


def test_residual_lowers_function_valued_controls_on_demand(backend):
    control = FunctionSpace(
        "u", arguments=[Scalar("time")], output=[VectorSpace("value", 1)]
    )
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=control,
            parameters=None,
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([0.0]), None),
            dynamics=lambda t, x, _z, u, _p: x + u(t),
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        ),
        backend=backend,
    )

    residual = to_residual_dynamical_system(system)

    np.testing.assert_allclose(
        residual.F(
            0.5,
            np.array([2.0]),
            np.array([5.0]),
            np.empty((0,)),
            lambda time: np.array([time + 1.0]),
            None,
        ),
        np.array([1.5]),
    )


def test_residual_lowers_native_casadi_function_parameter_symbolically():
    ca = pytest.importorskip("casadi")
    from coker.backends.casadi import CasadiBackend
    from coker.dynamics.variational.function_binding import (
        specialize_system_parameters,
    )
    from coker.parameters import BoundedVariable
    from coker.parameters.function_parameters import ClosureParameter

    response = FunctionSpace(
        "response",
        arguments=[Scalar("inflow")],
        output=[Scalar("rate")],
    )
    scale = ca.MX.sym("scale")
    inflow = ca.MX.sym("inflow")
    native_response = CasadiBackend().import_function(
        ca.Function("native_residual_rate", [scale, inflow], [scale * inflow])
    )
    native_response.name = "response"
    declaration = ClosureParameter(
        native_response,
        (BoundedVariable("rate_scale", lower_bound=1.0, upper_bound=1.0, guess=1.0),),
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
    specialized_system, _, _ = specialize_system_parameters(system, [declaration])
    residual = to_residual_dynamical_system(specialized_system)

    time = ca.MX.sym("time")
    w = ca.MX.sym("w", 1)
    wdot = ca.MX.sym("wdot", 1)
    parameter = ca.MX.sym("parameter", 1)
    (symbolic_residual,) = CasadiBackend().evaluate(
        residual.F,
        (
            time,
            w,
            wdot,
            ca.MX.zeros(0, 1),
            Noop(),
            parameter,
        ),
    )

    assert isinstance(symbolic_residual, ca.MX)
    residual_function = ca.Function(
        "residual",
        [time, w, wdot, parameter],
        [symbolic_residual],
    )
    assert float(residual_function(0.0, 0.0, 5.0, 3.0)) == pytest.approx(3.5)


def test_residual_preserves_scalar_quadrature_rate_row_shape(backend):
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=None,
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.zeros(2), None),
            dynamics=lambda t, x, _z, _u, _p: np.array([x[0] + t, x[1] - t]),
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=lambda t, x, _z, _u, _p: x[0] - t,
        ),
        backend=backend,
    )

    residual = to_residual_dynamical_system(system)

    actual = residual.F(
        0.25,
        np.array([3.0, 5.0, 11.0]),
        np.array([20.0, 30.0, 40.0]),
        np.empty((0,)),
        None,
        None,
    )
    assert actual.shape == (3,)
    np.testing.assert_allclose(actual, np.array([16.75, 25.25, 37.25]))


def test_residual_orders_state_quadrature_and_algebraic_rows(backend):
    def initial(z, _u, p):
        return np.array([p, 2.0 * p]), z

    def dynamics(t, x, z, _u, p):
        return np.array([x[0] + z[0] + t + p, x[1] - z[1]])

    def quadrature(t, x, z, _u, p):
        return np.array([x[0] * z[0], t + p])

    def constraints(_t, x, z, _u, p):
        return np.array([z[0] - x[0], z[1] + p])

    def output(_t, x, z, _u, p, q):
        return np.concatenate((x, z, np.array([p]), q))

    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=Scalar("gain"),
            algebraic=VectorSpace("z", 2),
            initial_conditions=initial,
            dynamics=dynamics,
            constraints=constraints,
            outputs=output,
            quadratures=quadrature,
        ),
        backend=backend,
    )
    residual = to_residual_dynamical_system(system)

    t = 0.25
    x = np.array([3.0, 5.0])
    z = np.array([7.0, -2.0])
    p = 4.0
    w = np.concatenate((x, np.array([11.0, 13.0])))
    wdot = np.array([20.0, 30.0, 40.0, 50.0])
    expected = np.concatenate(
        (
            wdot[:2] - dynamics(t, x, z, None, p),
            wdot[2:] - quadrature(t, x, z, None, p),
            constraints(t, x, z, None, p),
        )
    )
    np.testing.assert_allclose(residual.F(t, w, wdot, z, None, p), expected)
