import numpy as np
import pytest

from coker import VectorSpace
from coker.algebra.ops import Noop
from coker.backends.backend import get_backend_by_name
from coker.dynamics import DynamicsSpec, to_residual_dynamical_system
from coker.dynamics.system import create_dynamics_from_spec


@pytest.mark.parametrize("end_point", [0.25, np.array([0.25, 0.5])])
def test_casadi_integrates_algebraic_constraint(end_point):
    """An algebraic residual selects IDAS even without quadrature dynamics."""
    pytest.importorskip("casadi")
    backend = get_backend_by_name("casadi", set_current=False)

    x_final, z_final, q_final = backend.evaluate_integrals(
        [
            lambda _t, _x, z, _u, _p: z,
            lambda _t, x, z, _u, _p: z - x,
            Noop(),
        ],
        [np.array([1.0]), np.array([1.0]), None],
        end_point,
        [None, None],
    )

    expected = np.exp(np.asarray(end_point, dtype=float))
    np.testing.assert_allclose(
        np.asarray(x_final).reshape(-1), expected.reshape(-1), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(z_final).reshape(-1), expected.reshape(-1), rtol=1e-5
    )
    assert q_final is None


def test_casadi_integrates_quadrature_without_algebraic_constraint():
    pytest.importorskip("casadi")
    backend = get_backend_by_name("casadi", set_current=False)

    x_final, z_final, q_final = backend.evaluate_integrals(
        [
            lambda _t, x, _z, _u, _p: x * 0,
            Noop(),
            lambda _t, x, _z, _u, _p: x,
        ],
        [np.array([2.0]), None, np.array([0.0])],
        0.5,
        [None, None],
    )

    np.testing.assert_allclose(np.asarray(x_final), np.array([[2.0]]))
    assert z_final is None
    np.testing.assert_allclose(np.asarray(q_final), np.array([[1.0]]))


def test_casadi_system_integrates_residual_dae():
    """A public system reaches IDAS through its residual representation."""
    pytest.importorskip("casadi")
    backend = get_backend_by_name("casadi", set_current=False)
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=None,
            algebraic=VectorSpace("z", 1),
            initial_conditions=lambda _z, _u, _p: (
                np.array([1.0]),
                np.array([1.0]),
            ),
            dynamics=lambda _t, _x, z, _u, _p: z,
            constraints=lambda _t, x, z, _u, _p: z - x,
            outputs=lambda _t, _x, z, _u, _p, _q: z,
            quadratures=Noop(),
        ),
        backend="casadi",
    )

    explicit = backend.evaluate_integrals(
        system,
        [np.array([1.0]), np.array([1.0]), None],
        0.25,
        [None, None],
    )
    residual = backend.evaluate_integrals(
        to_residual_dynamical_system(system),
        [np.array([1.0]), np.array([1.0]), None],
        0.25,
        [None, None],
    )
    for explicit_value, residual_value in zip(explicit, residual):
        if explicit_value is not None:
            np.testing.assert_allclose(explicit_value, residual_value)
    np.testing.assert_allclose(
        np.asarray(explicit[0]).reshape(-1),
        np.array([np.exp(0.25)]),
        rtol=1e-5,
    )
