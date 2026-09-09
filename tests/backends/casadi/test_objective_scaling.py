import math

import pytest

from coker.backends.casadi.variational.objective_scaling import (
    derive_objective_scale,
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
def test_derive_objective_scale(nominal, tolerance, expected):
    assert derive_objective_scale(nominal, tolerance) == expected


def test_scale_preserves_negative_objective_direction():
    scale = derive_objective_scale(-25.0, 1e-3)

    assert -50.0 / scale == pytest.approx(-2.0)
    assert (-50.0 / scale) * scale == pytest.approx(-50.0)


def test_identity_scale_handles_invalid_values():
    assert derive_objective_scale(math.nan, 1e-3) == 1.0
