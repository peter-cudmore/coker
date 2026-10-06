from dataclasses import replace

import numpy as np
import pytest


from coker.algebra.ops import Noop
from coker.backends.numpy import NumpyBackend
from coker.dynamics import DynamicsSpec, to_residual_dynamical_system
from coker.dynamics.residual import _normalise_direct_integration_functions
from coker.dynamics.system import create_dynamics_from_spec


def _semi_explicit_system(quadratures=None):
    return create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=None,
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (np.array([1.0]), None),
            dynamics=lambda _t, x, _z, _u, _p: x,
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop() if quadratures is None else quadratures,
        )
    )


def test_normalises_semi_explicit_system():
    system = _semi_explicit_system()

    rate, algebraic, quadrature = _normalise_direct_integration_functions(
        system
    )

    assert rate is system.dxdt
    assert isinstance(algebraic, Noop)
    assert isinstance(quadrature, Noop)


def test_normalises_certified_converted_residual():
    system = _semi_explicit_system()
    residual = to_residual_dynamical_system(system)

    rate, algebraic, quadrature = _normalise_direct_integration_functions(
        residual
    )

    assert residual.legacy is not None
    assert rate is residual.legacy.rate
    assert isinstance(algebraic, Noop)
    assert isinstance(quadrature, Noop)


def test_normalises_raw_callback_triple():
    def rate(_t, x, _z, _u, _p):
        return x

    callbacks = [rate, Noop(), Noop()]

    normalised_rate, algebraic, quadrature = (
        _normalise_direct_integration_functions(callbacks)
    )

    assert normalised_rate is rate
    assert isinstance(algebraic, Noop)
    assert isinstance(quadrature, Noop)


def test_rejects_independent_residual_direct_integration():
    residual = replace(
        to_residual_dynamical_system(_semi_explicit_system()), legacy=None
    )

    with pytest.raises(
        NotImplementedError,
        match=(
            "independent residual systems without a semi-explicit certificate"
        ),
    ):
        _normalise_direct_integration_functions(residual)


def test_numpy_integrates_certified_residual_directly():
    residual = to_residual_dynamical_system(_semi_explicit_system())

    x, z, q = NumpyBackend().evaluate_integrals(
        residual,
        [np.array([1.0]), None, None],
        0.25,
        [None, None],
    )

    np.testing.assert_allclose(x, np.array([np.exp(0.25)]), rtol=1e-5)
    assert z is None
    assert q is None


@pytest.mark.parametrize(
    ("end_point", "expected_x", "expected_q"),
    [
        (0.25, np.array([np.exp(0.25)]), np.array([0.25])),
        (
            np.array([0.125, 0.25]),
            np.array([[np.exp(0.125), np.exp(0.25)]]),
            np.array([[0.125, 0.25]]),
        ),
    ],
)
def test_numpy_integrates_certified_residual_quadrature_directly(
    end_point, expected_x, expected_q
):
    residual = to_residual_dynamical_system(
        _semi_explicit_system(lambda _t, _x, _z, _u, _p: np.array([1.0]))
    )

    x, z, q = NumpyBackend().evaluate_integrals(
        residual,
        [np.array([1.0]), None, np.array([0.0])],
        end_point,
        [None, None],
    )

    np.testing.assert_allclose(x, expected_x, rtol=1e-5)
    assert z is None
    np.testing.assert_allclose(q, expected_q, rtol=1e-5)
