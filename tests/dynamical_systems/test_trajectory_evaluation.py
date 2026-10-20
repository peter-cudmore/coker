"""Trajectory evaluation must reuse an already-constructed system's trace."""

import cProfile

import numpy as np
import pytest

from coker import Function, VectorSpace
from coker.dynamics import create_autonomous_ode


@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_repeated_ordinary_evaluation_does_not_trace_functions(backend_name):
    if backend_name == "pytorch":
        pytest.importorskip("torch")
    system = create_autonomous_ode(
        x0=lambda p: p,
        xdot=lambda x, _p: -x,
        parameters=VectorSpace("p", 1),
        backend=backend_name,
    )
    with cProfile.Profile() as profile:
        values = [system(0.0, np.array([value])) for value in (1.0, 2.0, 3.0)]

    for actual, expected in zip(values, (1.0, 2.0, 3.0)):
        np.testing.assert_allclose(actual, [expected])
    traces = sum(
        entry.callcount
        for entry in profile.getstats()
        if entry.code is Function.__init__.__code__
    )
    assert traces == 0, "Trajectory evaluation must not construct new Functions"
