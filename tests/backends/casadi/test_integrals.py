import numpy as np
import pytest

from coker.algebra.ops import Noop
from coker.backends.backend import get_backend_by_name


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
