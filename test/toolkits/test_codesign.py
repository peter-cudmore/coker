import numpy as np

from coker.toolkits.codesign import ProblemBuilder, Minimise
from coker import kernel, VectorSpace, Dimension


def quadratic(x, p):
    # solution should be |x| = 0, |p| = 0
    return x.T @ x + p.T @ p


def test_optimisation():

    # define decision variables
    # objective
    # constraints

    with ProblemBuilder() as builder:
        assert not builder.arguments

        x = builder.new_variable(name='x', shape=(3,))
        p = builder.new_variable(name='p', shape=(2,))

        builder.objective = Minimise(quadratic(x, p))
        builder.constraints = [
            x[0] > 1
        ]
        builder.outputs = [x, p]
        builder.initial_conditions = [

            2*np.ones(x.shape), np.ones(p.shape)
        ]
        problem = builder.build()

    assert not problem.input_shape
    assert problem.output_shape == (Dimension((3,)), Dimension((2,)))

    soln = problem()

    assert len(soln) == 2
    x_val, p_val = soln
    x_expected = np.array([1,0,0],dtype=float)

    assert isinstance(x_val, np.ndarray) and (abs(x_val -x_expected) < 1e-3).all()
    assert isinstance(p_val, np.ndarray) and (abs(p_val) < 1e-3).all()


