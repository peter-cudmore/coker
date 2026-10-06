import numpy as np
import pytest

from coker import Dimension, Scalar, VectorSpace, function
from coker.algebra.ops import Noop
from coker.backends.numpy import NumpyBackend
from coker.dynamics import ResidualDynamicalSystem


def _coupled_residual_system():
    return ResidualDynamicalSystem(
        inputs=Noop(),
        parameters=None,
        x0=function(
            [VectorSpace("z", 1), Noop(), None],
            lambda z, _u, _p: (np.array([0.0]), z),
            backend="numpy",
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
            backend="numpy",
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
            backend="numpy",
        ),
        differential=Dimension(1),
        algebraic=Dimension(1),
        quadrature=Dimension(1),
    )


def _state_residual_system(residual):
    return ResidualDynamicalSystem(
        inputs=Noop(),
        parameters=None,
        x0=function(
            [None, Noop(), None],
            lambda _z, _u, _p: (np.array([1.0]), None),
            backend="numpy",
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
            residual,
            backend="numpy",
        ),
        y=function(
            [Scalar("t"), VectorSpace("x", 1), None, Noop(), None, None],
            lambda _t, x, _z, _u, _p, _q: x,
            backend="numpy",
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
        (0.75, (1,)),
        (np.array([0.25, 0.75]), (1, 2)),
        (np.array([0.0, -0.25, -0.5]), (1, 3)),
    ],
)
def test_numpy_solves_coupled_residual_as_implicit_state_steps(
    end_point, shape
):
    x, z, q = NumpyBackend().evaluate_integrals(
        _coupled_residual_system(),
        [np.array([0.0]), np.array([2.0 / 3.0]), np.array([0.0])],
        end_point,
        [None, None],
    )

    expected_x, expected_z, expected_q = _coupled_continuous_values(
        np.atleast_1d(end_point)
    )
    if isinstance(end_point, np.ndarray):
        expected_x = expected_x[np.newaxis, :]
        expected_z = expected_z[np.newaxis, :]
        expected_q = expected_q[np.newaxis, :]

    assert x.shape == shape
    assert z.shape == shape
    assert q.shape == shape
    np.testing.assert_allclose(x, expected_x, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(z, expected_z, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(q, expected_q, rtol=1e-5, atol=1e-7)


def test_numpy_requires_coupled_residual_algebraic_initial_condition():
    with pytest.raises(ValueError, match="algebraic initial condition"):
        NumpyBackend().evaluate_integrals(
            _coupled_residual_system(),
            [np.array([0.0]), None, np.array([0.0])],
            0.25,
            [None, None],
        )


def test_numpy_solves_implicit_state_at_the_step_endpoint():
    system = _state_residual_system(
        lambda _t, w, wdot, _z, _u, _p: np.array([w[0] * wdot[0] - 1.0])
    )

    x, z, q = NumpyBackend().evaluate_integrals(
        system,
        [np.array([1.0]), None, None],
        0.5,
        [None, None],
    )

    np.testing.assert_allclose(x, [np.sqrt(2.0)], rtol=1e-5)
    assert z is None
    assert q is None


def test_numpy_rejects_nonzero_implicit_state_step_residuals():
    system = _state_residual_system(
        lambda _t, _w, wdot, _z, _u, _p: np.array([wdot[0] ** 2 + 1.0])
    )

    with pytest.raises(RuntimeError, match="state-step solve failed at time"):
        NumpyBackend().evaluate_integrals(
            system,
            [np.array([1.0]), None, None],
            0.25,
            [None, None],
        )


@pytest.mark.parametrize(
    ("end_point", "message"),
    [
        (np.array([np.nan]), "must be finite"),
        (np.array([0.0, 0.25, 0.25]), "must be strictly monotonic"),
    ],
)
def test_numpy_validates_implicit_residual_evaluation_times(
    end_point, message
):
    system = _state_residual_system(
        lambda _t, _w, wdot, _z, _u, _p: np.array([wdot[0] - 1.0])
    )

    with pytest.raises(ValueError, match=message):
        NumpyBackend().evaluate_integrals(
            system,
            [np.array([1.0]), None, None],
            end_point,
            [None, None],
        )
