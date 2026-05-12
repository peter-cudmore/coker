import numpy as np
import pytest

from coker.toolkits.codesign import (
    ProblemBuilder,
    Minimise,
    norm as codesign_norm,
)
from coker import Dimension, SolveFailure, VectorSpace


def quadratic(x, p, z):
    # solution should be |x| = 0, |p| = 0 z = 0
    return x.T @ x + p.T @ p + z**2


def test_optimisation_zero_input_problem(variational_backend):

    with ProblemBuilder() as builder:
        assert not builder.arguments

        x = builder.new_variable(name="x", shape=(3,))
        p = builder.new_variable(name="p", shape=(2,))
        z = builder.new_variable(name="z")
        builder.objective = Minimise(quadratic(x, p, z))
        e_y = np.array([0, 1, 0], dtype=float)
        builder.constraints = [
            1 < (z**2),
            x[0] > 1,
            2 < np.dot(e_y, x),
        ]
        builder.outputs = [x, p, z]
        builder.initial_conditions = [
            2 * np.ones(x.shape),
            np.ones(p.shape),
            4,
        ]
        problem = builder.build(variational_backend)

    assert not problem.input_shape
    assert problem.output_shape == (
        Dimension((3,)),
        Dimension((2,)),
        Dimension(None),
    )

    soln = problem()

    assert len(soln) == 3
    x_val, p_val, z_val = soln

    x_expected = np.array([1, 2, 0], dtype=float)
    p_expected = np.array([0, 0], dtype=float)
    assert np.allclose(x_val, x_expected, atol=1e-6)
    assert np.allclose(p_val, p_expected, atol=1e-6)
    assert 1 - 1e-5 < abs(z_val) < 1 + 1e-5
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_optimisation_accepts_runtime_parameters(variational_backend):

    with ProblemBuilder(arguments=[VectorSpace("target", 2)]) as builder:
        (target,) = builder.arguments
        x = builder.new_variable(
            name="x", shape=(2,), initial_value=np.ones(2)
        )
        delta = x - target
        builder.objective = Minimise(np.dot(delta, delta))
        builder.outputs = [x]
        problem = builder.build(variational_backend)

    assert problem.input_shape == (Dimension((2,)),)
    assert problem.output_shape == (Dimension((2,)),)

    (x_val,) = problem(np.array([3.0, -1.0]))
    assert x_val.shape == (2,)
    assert np.allclose(x_val, np.array([3.0, -1.0]), atol=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_optimisation_norm_helper(variational_backend):

    with ProblemBuilder() as builder:
        x = builder.new_variable(
            name="x",
            shape=(2,),
            initial_value=np.array([2.0, 1.0]),
        )
        builder.objective = Minimise(
            codesign_norm(x - np.array([1.0, 0.0]), order=2)
        )
        builder.constraints = [x[0] > 1.5]
        builder.outputs = [x]
        problem = builder.build(variational_backend)

    (x_val,) = problem()
    assert np.allclose(x_val, np.array([1.5, 0.0]), atol=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_optimisation_raises_on_infeasible_solve(variational_backend):

    with ProblemBuilder() as builder:
        x = builder.new_variable(name="x", shape=(2,))
        builder.objective = Minimise(np.dot(x, x))
        builder.constraints = [x[0] > 1, x[0] < 0]
        builder.outputs = [x]
        problem = builder.build(variational_backend)

    with pytest.raises(SolveFailure) as exc_info:
        problem()

    assert problem.solve_info is not None
    assert not problem.solve_info.success
    assert exc_info.value.solve_info == problem.solve_info


def test_optimisation_supports_nonlinear_constraints(variational_backend):

    with ProblemBuilder() as builder:
        x = builder.new_variable(
            name="x",
            shape=(2,),
            initial_value=np.array([1.0, 0.0]),
        )
        target = np.array([1.0, 0.0])
        delta = x - target
        builder.objective = Minimise(np.dot(x, x))
        builder.constraints = [np.dot(delta, delta) < 0.25]
        builder.outputs = [x]
        problem = builder.build(variational_backend)

    (x_val,) = problem()
    assert np.allclose(x_val, np.array([0.5, 0.0]), atol=5e-4)
    assert problem.solve_info is not None
    assert problem.solve_info.success
