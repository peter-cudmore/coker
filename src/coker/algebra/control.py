"""Numerical normalisation over graph values."""

from typing import Union

import numpy as np

from coker.algebra.graph import Tracer


def normalise(v: Union[np.ndarray, Tracer]):
    if isinstance(v, np.ndarray):
        if all(v_i == 0 for v_i in v):
            return np.zeros_like(v), 0
        else:
            r = np.linalg.norm(v)
            return v / r, r

    assert isinstance(v, Tracer), f"Expected Tracer got {type(v)}"

    unit_v = v.normalise()
    norm_v = v.norm()

    return unit_v, norm_v
