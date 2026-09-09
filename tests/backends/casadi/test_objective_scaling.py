import math

import numpy as np
import pytest

from coker.backends.casadi.variational.objective_scaling import (
    ObjectiveScaling,
    derive_objective_scaling,
)


@pytest.mark.parametrize(
    ("nominal", "tolerance", "expected"),
    [
        (12.0, 1e-3, 12.0),
        (-12.0, 1e-3, 12.0),
        (0.0, 1e-3, 1.0),
        (0.0, 0.0, 1.0),
        (float("nan"), 1e-3, 1.0),
        (float("inf"), 1e-3, 1.0),
        (1.0, float("nan"), 1.0),
        (1.0, float("inf"), 1.0),
    ],
)
def test_derive_objective_scaling(nominal, tolerance, expected):
    scaling = derive_objective_scaling(nominal, tolerance)
    assert scaling.scale == expected


def test_scale_and_unscale_preserve_negative_physical_cost():
    scaling = derive_objective_scaling(-25.0, 1e-3)

    normalized = scaling.scale_cost(-50.0)

    assert normalized == pytest.approx(-2.0)
    assert scaling.unscale_cost(normalized) == pytest.approx(-50.0)


def test_scaling_accepts_numpy_scalar_and_vector_costs():
    scaling = derive_objective_scaling(np.float64(4.0), np.float64(0.5))
    physical = np.array([-8.0, 2.0])

    np.testing.assert_allclose(
        scaling.unscale_cost(scaling.scale_cost(physical)), physical
    )


def test_identity_fallback_round_trips_cost():
    scaling = derive_objective_scaling(math.nan, 1e-3)

    assert scaling.scale == 1.0
    assert scaling.scale_cost(-3.5) == -3.5
    assert scaling.unscale_cost(-3.5) == -3.5


def test_encode_decode_are_directional_aliases():
    scaling = ObjectiveScaling(4.0)

    assert scaling.encode_cost(-12.0) == -3.0
    assert scaling.decode_cost(-3.0) == -12.0
