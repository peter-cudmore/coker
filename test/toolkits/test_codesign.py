import numpy as np

from coker.toolkits.codesign import ProblemBuilder, Minimise
from coker import kernel, VectorSpace, Dimension


def quadratic(x, p, z):
    # solution should be |x| = 0, |p| = 0 z = 0
    return x.T @ x + p.T @ p + z ** 2


def test_optimisation():
    # define decision variables
    # objective
    # constraints

    with ProblemBuilder() as builder:
        assert not builder.arguments

        x = builder.new_variable(name='x', shape=(3,))
        p = builder.new_variable(name='p', shape=(2,))
        z = builder.new_variable(name='z')
        builder.objective = Minimise(quadratic(x, p, z))
        e_y = np.array([[0], [1], [0]], dtype=float)
        builder.constraints = [
#            1 < (z ** 2),
            # numpy implementation doesn't like nonlinear constraints
            x[0] > 1,
            2 < e_y.T @ x
        ]
        builder.outputs = [x, p, z]
        builder.initial_conditions = [
            2 * np.ones(x.shape), np.ones(p.shape), 4
        ]
        problem = builder.build()

    assert not problem.input_shape
    assert problem.output_shape == (Dimension((3,)), Dimension((2,)), Dimension(None))

    soln = problem()

    assert len(soln) == 3
    x_val, p_val, z_val = soln
    x_expected = np.array([1, 2, 0], dtype=float)

    assert isinstance(x_val, np.ndarray) and (abs(x_val - x_expected) < 1e-3).all()
    assert isinstance(p_val, np.ndarray) and (abs(p_val) < 1e-3).all()
    assert abs(z_val ) < 1e-3
