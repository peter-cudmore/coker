import numpy as np
import pytest

from coker.toolkits.codesign import ProblemBuilder, Minimise
from coker import function, VectorSpace, Dimension


def quadratic(x, p, z):
    # solution should be |x| = 0, |p| = 0 z = 0
    return x.T @ x + p.T @ p + z ** 2



def test_optimisation(backend):
    # define decision variables
    # objective
    # constraints
    if backend not in {'casadi'}:
        return

    with ProblemBuilder() as builder:
        assert not builder.arguments

        x = builder.new_variable(name='x', shape=(3,))
        p = builder.new_variable(name='p', shape=(2,))
        z = builder.new_variable(name='z')
        builder.objective = Minimise(quadratic(x, p, z))
        e_y = np.array([[0], [1], [0]], dtype=float)
        builder.constraints = [
            1 < (z ** 2),
            # numpy implementation doesn't like nonlinear constraints
            x[0] > 1,
            2 < e_y.T @ x
        ]
        builder.outputs = [x, p, z]
        builder.initial_conditions = [
            2 * np.ones(x.shape), np.ones(p.shape), 4
        ]
        problem = builder.build(backend)

    assert not problem.input_shape
    assert problem.output_shape == (Dimension((3,)), Dimension((2,)), Dimension(None))

    soln = problem()

    assert len(soln) == 3
    x_val, p_val, z_val = soln

    x_expected = np.array([1, 2, 0], dtype=float)
    p_expected = np.array([0, 0], dtype=float)
    assert np.allclose(x_val, x_expected, atol=1e-6)
    assert np.allclose(p_val, p_expected, atol=1e-6)
    assert 1 - 1e-6 < abs(z_val) < 1 + 1e-6