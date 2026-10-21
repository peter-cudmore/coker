import numpy as np
import pytest

from coker import Dimension, Scalar, VectorSpace, function
from coker.algebra.ops import Noop
from coker.dynamics import ResidualDynamicalSystem


@pytest.fixture(scope="module")
def casadi_backend():
    pytest.importorskip("casadi")
    from coker.backends.casadi import CasadiBackend

    return CasadiBackend()


def _coupled_residual_system():
    return ResidualDynamicalSystem(
        inputs=Noop(),
        parameters=None,
        x0=function(
            [VectorSpace("z", 1), Noop(), None],
            lambda z, _u, _p: (np.array([0.0]), z),
            backend="casadi",
        ),
        F=function(
            [
                Scalar("t"),
                VectorSpace("w", 2),
                VectorSpace("wdot", 2),
                VectorSpace("z", 1),
                Noop(),
                None,
            ],
            lambda _t, w, wdot, z, _u, _p: np.array(
                [
                    wdot[0] + z[0] - 1.0,
                    wdot[1] - z[0],
                    w[0] + 2.0 * wdot[0] - z[0],
                ]
            ),
            backend="casadi",
        ),
        y=function(
            [
                Scalar("t"),
                VectorSpace("x", 1),
                VectorSpace("z", 1),
                Noop(),
                None,
                VectorSpace("q", 1),
            ],
            lambda _t, x, z, _u, _p, q: x + z + q,
            backend="casadi",
        ),
        differential=Dimension(1),
        algebraic=Dimension(1),
        quadrature=Dimension(1),
    )


def _nonlinear_state_rate_residual_system():
    return ResidualDynamicalSystem(
        inputs=Noop(),
        parameters=None,
        x0=function(
            [None, Noop(), None],
            lambda _z, _u, _p: (np.array([1.0]), None),
            backend="casadi",
        ),
        F=function(
            [
                Scalar("t"),
                VectorSpace("w", 1),
                VectorSpace("wdot", 1),
                VectorSpace("z", 0),
                Noop(),
                None,
            ],
            lambda _t, w, wdot, _z, _u, _p: np.array([w[0] * wdot[0] - 1.0]),
            backend="casadi",
        ),
        y=function(
            [Scalar("t"), VectorSpace("x", 1), None, Noop(), None, None],
            lambda _t, x, _z, _u, _p, _q: x,
            backend="casadi",
        ),
        differential=Dimension(1),
        algebraic=None,
        quadrature=None,
    )


def _coupled_continuous_values(times):
    exponential = np.exp(-np.asarray(times, dtype=float) / 3.0)
    return (
        1.0 - exponential,
        1.0 - exponential / 3.0,
        np.asarray(times, dtype=float) - 1.0 + exponential,
    )


@pytest.mark.parametrize(
    ("end_point", "shape"),
    [
        (0.75, (1, 1)),
        (np.array([0.25, 0.75]), (1, 2)),
    ],
)
def test_casadi_integrates_independent_coupled_residual_system(
    end_point, shape, casadi_backend
):

    x, z, q = casadi_backend.evaluate_integrals(
        _coupled_residual_system(),
        [np.array([0.0]), np.array([2.0 / 3.0]), np.array([0.0])],
        end_point,
        [None, None],
    )

    expected_x, expected_z, expected_q = _coupled_continuous_values(
        np.atleast_1d(end_point)
    )
    expected_x = expected_x.reshape(shape)
    expected_z = expected_z.reshape(shape)
    expected_q = expected_q.reshape(shape)

    assert x.shape == shape
    assert z.shape == shape
    assert q.shape == shape
    np.testing.assert_allclose(np.asarray(x), expected_x, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(z), expected_z, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(q), expected_q, rtol=1e-5, atol=1e-7)


def test_casadi_returns_independent_residual_initial_values_at_zero(casadi_backend):
    x, z, q = casadi_backend.evaluate_integrals(
        _coupled_residual_system(),
        [np.array([0.0]), np.array([2.0 / 3.0]), np.array([0.0])],
        0.0,
        [None, None],
    )

    assert x.shape == (1, 1)
    assert z.shape == (1, 1)
    assert q.shape == (1, 1)
    np.testing.assert_allclose(np.asarray(x), [[0.0]])
    np.testing.assert_allclose(np.asarray(z), [[2.0 / 3.0]])
    np.testing.assert_allclose(np.asarray(q), [[0.0]])


def test_casadi_rejects_non_monotonic_independent_residual_time_grid(casadi_backend):
    with pytest.raises(ValueError, match="must be strictly monotonic"):
        casadi_backend.evaluate_integrals(
            _coupled_residual_system(),
            [np.array([0.0]), np.array([2.0 / 3.0]), np.array([0.0])],
            np.array([0.0, 0.75, 0.25]),
            [None, None],
        )


def test_casadi_integrates_nonlinear_state_rate_residual(casadi_backend):

    x, z, q = casadi_backend.evaluate_integrals(
        _nonlinear_state_rate_residual_system(),
        [np.array([1.0]), None, None],
        0.5,
        [None, None],
    )

    assert x.shape == (1, 1)
    np.testing.assert_allclose(np.asarray(x), [[np.sqrt(2.0)]], rtol=1e-5)
    assert z is None
    assert q is None


def test_casadi_binds_scalar_parameter_for_independent_residual(casadi_backend):
    rate = Scalar("rate")
    system = ResidualDynamicalSystem(
        inputs=Noop(),
        parameters=(rate,),
        x0=function(
            [None, Noop(), rate],
            lambda _z, _u, _rate: (np.array([0.0]), None),
            backend="casadi",
        ),
        F=function(
            [
                Scalar("t"),
                VectorSpace("w", 1),
                VectorSpace("wdot", 1),
                None,
                Noop(),
                rate,
            ],
            lambda _t, _w, wdot, _z, _u, parameter: np.array([wdot[0] - parameter]),
            backend="casadi",
        ),
        y=function(
            [Scalar("t"), VectorSpace("x", 1), None, Noop(), rate, None],
            lambda _t, x, _z, _u, _parameter, _q: x,
            backend="casadi",
        ),
        differential=Dimension(1),
        algebraic=None,
        quadrature=None,
    )

    x, z, q = casadi_backend.evaluate_integrals(
        system,
        [np.array([0.0]), None, None],
        np.array([0.25, 0.75]),
        [None, 2.0],
    )

    assert x.shape == (1, 2)
    np.testing.assert_allclose(np.asarray(x), [[0.5, 1.5]], rtol=1e-5, atol=1e-7)
    assert z is None
    assert q is None


def test_casadi_selects_variational_residual_solver(casadi_backend):
    from coker.backends.casadi import (
        CasadiResidualSolver,
        CasadiResidualSolverOptions,
    )

    x, z, q = casadi_backend.evaluate_integrals(
        _coupled_residual_system(),
        [np.array([0.0]), np.array([2.0 / 3.0]), np.array([0.0])],
        0.75,
        [None, None],
        CasadiResidualSolverOptions(CasadiResidualSolver.VARIATIONAL),
    )

    expected_x, expected_z, expected_q = _coupled_continuous_values([0.75])
    np.testing.assert_allclose(
        np.asarray(x), expected_x.reshape((1, 1)), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(z), expected_z.reshape((1, 1)), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(q), expected_q.reshape((1, 1)), rtol=1e-5, atol=1e-7
    )
