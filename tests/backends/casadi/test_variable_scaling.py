from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from coker.backends.casadi.variational.solver import (
    _accepts_small_search_direction,
)
from coker.backends.casadi.variational.variable_scaling import (
    _VariableScaling,
    _derive_constraint_scaling,
    _derive_variable_scaling,
)


def test_two_sided_interval_uses_midpoint_and_half_span():
    scaling = _derive_variable_scaling([-2.0], [0.0], [6.0])
    np.testing.assert_allclose(scaling.offset, [2.0])
    np.testing.assert_allclose(scaling.scale, [4.0])
    np.testing.assert_allclose(scaling.encode([2.0]), [0.0])
    np.testing.assert_allclose(scaling.decode([-1.0]), [-2.0])


def test_one_sided_and_unbounded_coordinates_use_guess_distance():
    scaling = _derive_variable_scaling(
        [-3.0, -np.inf, -np.inf], [5.0, -4.0, 0.25], [np.inf, np.inf, np.inf]
    )
    np.testing.assert_allclose(scaling.offset, [-3.0, -4.0, 0.25])
    np.testing.assert_allclose(scaling.scale, [8.0, 4.0, 1.0])


def test_fixed_coordinate_is_identity_around_fixed_value():
    scaling = _derive_variable_scaling([3.0], [3.0], [3.0])
    np.testing.assert_allclose(scaling.offset, [3.0])
    np.testing.assert_allclose(scaling.scale, [1.0])
    np.testing.assert_allclose(scaling.decode(scaling.encode([3.0])), [3.0])


def test_bound_conversion_preserves_infinities_and_supports_casadi():
    scaling = _VariableScaling(np.array([2.0, -1.0]), np.array([4.0, 2.0]))
    lower, upper = scaling.encode_bounds([-np.inf, -5.0], [np.inf, 7.0])
    assert np.isneginf(lower[0]) and np.isposinf(upper[0])
    np.testing.assert_allclose(lower[1:], [-2.0])
    np.testing.assert_allclose(upper[1:], [4.0])

    y = ca.MX.sym("y", 2)
    physical = scaling.decode(y)
    assert isinstance(physical, ca.MX)
    assert physical.shape == (2, 1)


@pytest.mark.parametrize(
    "lower, guess, upper",
    [
        ([2.0], [1.0], [0.0]),
        ([np.nan], [0.0], [1.0]),
        ([-np.inf], [np.inf], [np.inf]),
        ([-np.inf], [0.0], [np.nan]),
    ],
)
def test_rejects_invalid_inputs(lower, guess, upper):
    with pytest.raises(ValueError):
        _derive_variable_scaling(lower, guess, upper)


def test_rejects_wrong_vector_lengths():
    with pytest.raises(ValueError):
        _derive_variable_scaling([0.0, 0.0], [0.0], [1.0, 1.0])


def test_constraint_scaling_normalizes_row_sensitivities_and_bounds():
    decision = ca.MX.sym("decision", 2)
    residual = ca.vertcat(1e6 * decision[0], 1e-6 * decision[1])
    lower = np.array([-1e3, -2e-9])
    upper = np.array([1e3, 2e-9])

    scaling = _derive_constraint_scaling(
        residual,
        decision,
        np.array([0.0, 0.0]),
        lower,
        upper,
    )

    np.testing.assert_allclose(scaling, [1e6, 1e-6])
    np.testing.assert_allclose(lower / scaling, [-1e-3, -2e-3])
    np.testing.assert_allclose(upper / scaling, [1e-3, 2e-3])


def test_small_search_direction_uses_per_row_scaled_tolerance():
    solve_info = SimpleNamespace(
        return_status="Search_Direction_Becomes_Too_Small"
    )

    assert not _accepts_small_search_direction(
        solve_info,
        {"f": ca.DM(0.0), "g": ca.DM([1e-6])},
        ca.DM([0.0]),
        ca.DM([0.0]),
        tolerance=ca.DM([1e-7]),
    )
