import numpy as np
import pytest

from coker import Dimension, FunctionSpace, Scalar, VectorSpace, function
from coker.toolkits.codesign import (
    MathematicalProgram,
    Minimise,
    ProblemBuilder,
    SolveFailure,
    bounded,
    norm as codesign_norm,
)
from coker.parameters.function_parameters import (
    DenseLayer,
    FittedFunction,
    MonotonePiecewiseLinear,
)


def quadratic(x, p, z):
    # solution should be |x| = 0, |p| = 0 z = 0
    return x.T @ x + p.T @ p + z**2


def test_mathematical_program_composes_symbolically_and_compiles():
    program = MathematicalProgram(
        input_shape=(Dimension.scalar(),),
        output_shape=(Dimension.scalar(),),
        implementation=lambda x: (x**2, x + 1),
    )

    composed = function(
        arguments=[Scalar("x")],
        implementation=lambda x: program(x)[1] * 2,
        backend="numpy",
    )

    assert not hasattr(program, "impl")
    assert composed(3) == 8

    compiled = program.lower()
    assert compiled.backend == "numpy"
    assert compiled(3) == (9.0, 4.0)


def test_symbolic_program_results_share_one_invocation():
    calls = 0

    def implementation(x):
        nonlocal calls
        calls += 1
        return x**2, x + 1, x + 2

    program = MathematicalProgram(
        input_shape=(Dimension.scalar(),),
        output_shape=(Dimension.scalar(), Dimension.scalar()),
        implementation=implementation,
    )
    composed = function(
        [Scalar("x")], lambda x: sum(program(x)), backend="numpy"
    )

    assert composed(3) == 18
    assert calls == 1


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
        Dimension.scalar(),
    )

    soln = problem()

    assert len(soln) == 4
    objective, x_val, p_val, z_val = soln

    x_expected = np.array([1, 2, 0], dtype=float)
    p_expected = np.array([0, 0], dtype=float)
    assert objective == pytest.approx(6.0, abs=1e-5)
    assert np.allclose(x_val, x_expected, atol=1e-6)
    assert np.allclose(p_val, p_expected, atol=1e-6)
    assert 1 - 1e-5 < abs(z_val) < 1 + 1e-5
    np.testing.assert_allclose(problem.parameters["x"], x_expected, atol=1e-6)
    np.testing.assert_allclose(problem.parameters["p"], p_expected, atol=1e-6)
    assert isinstance(problem.parameters["z"], float)
    assert abs(problem.parameters["z"]) == pytest.approx(abs(z_val), abs=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_optimisation_accepts_runtime_parameters(variational_backend):

    with ProblemBuilder(arguments=[VectorSpace("target", 3)]) as builder:
        (target,) = builder.arguments
        x = builder.new_variable(
            name="x", shape=(2,), initial_value=np.ones(2)
        )
        delta = x - target[:2]
        builder.objective = Minimise(np.dot(delta, delta))
        builder.outputs = [x]
        problem = builder.build(variational_backend)

    assert problem.input_shape == (Dimension((3,)),)
    assert problem.output_shape == (Dimension((2,)),)

    objective, x_val = problem(np.array([3.0, -1.0, 7.0]))
    assert objective == pytest.approx(0.0, abs=1e-6)
    assert x_val.shape == (2,)
    assert np.allclose(x_val, np.array([3.0, -1.0]), atol=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_mathematical_program_reconstructs_function_parameter(
    variational_backend,
):
    rate = FunctionSpace(
        "rate",
        arguments=[VectorSpace("state", 1)],
        output=[VectorSpace("rate", 1)],
    )
    activation = function(
        [VectorSpace("hidden", 1)],
        lambda hidden: hidden,
        backend=variational_backend,
    )
    declaration = DenseLayer(1, activation, name="response")
    with ProblemBuilder() as builder:
        response = builder.new_function_parameter(rate, declaration)
        value = response(np.array([1.0]))[0]
        builder.objective = Minimise((value - 0.5) ** 2)
        builder.outputs = [value]
        problem = builder.build(variational_backend)

    result = problem()

    assert len(result) == 2
    objective, value = result
    assert objective == pytest.approx(0.0, abs=1e-6)
    assert value == pytest.approx(0.5, abs=1e-6)

    fitted = problem.parameters["response"]
    assert isinstance(fitted, FittedFunction)
    assert fitted.specification is declaration
    assert fitted.space is rate
    assert callable(fitted.function)
    assert [parameter.shape for parameter in fitted.parameters] == [
        (1, 1),
        (1,),
    ]
    assert fitted(np.array([1.0]))[0] == pytest.approx(value, abs=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_mathematical_program_fits_monotone_piecewise_linear_cubic(
    variational_backend,
):
    """A piecewise-linear fit matches cubic knots but not the cubic between them."""
    knots = np.linspace(0.0, 2.0, 17)
    targets = knots**3
    response_space = FunctionSpace(
        "response",
        arguments=[Scalar("x")],
        output=[Scalar("y")],
    )
    declaration = MonotonePiecewiseLinear(
        domain_knots=knots,
        lower_bound=-0.1,
        upper_bound=8.1,
        name="response",
    )

    with ProblemBuilder() as builder:
        response = builder.new_function_parameter(response_space, declaration)
        residuals = [
            response(knot) - target for knot, target in zip(knots, targets)
        ]
        builder.objective = Minimise(
            sum(residual * residual for residual in residuals)
        )
        builder.outputs = [response(knots[0])]
        problem = builder.build(variational_backend)

    objective, _ = problem()
    fitted = problem.parameters["response"]
    knot_values = np.asarray([fitted(knot) for knot in knots], dtype=float)
    np.testing.assert_allclose(knot_values, targets, atol=1e-3)

    evaluation_points = np.linspace(0.0, 2.0, 101)
    fitted_values = np.asarray(
        [fitted(point) for point in evaluation_points],
        dtype=float,
    )
    errors = fitted_values - evaluation_points**3
    max_error = np.max(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    assert np.isfinite(objective)
    assert np.all(np.isfinite(fitted_values))
    assert max_error < 2.5e-2
    assert rmse < 1e-2
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_casadi_optimisation_passes_options_at_solver_construction(
    monkeypatch,
):
    pytest.importorskip("casadi")
    from coker.backends.casadi import CasadiNLPSolverOptions
    import coker.backends.casadi.optimiser as casadi_optimiser

    nlpsol = casadi_optimiser.ca.nlpsol
    calls = []

    def capture_nlpsol(*args, **kwargs):
        calls.append((args, kwargs))
        return nlpsol(*args, **kwargs)

    monkeypatch.setattr(casadi_optimiser.ca, "nlpsol", capture_nlpsol)
    solver_options = CasadiNLPSolverOptions(
        optimiser_options={
            "ipopt.max_iter": 50,
            "ipopt.max_cpu_time": 30.0,
        }
    )

    with ProblemBuilder(solver_options=solver_options) as builder:
        x = builder.new_variable(name="x", initial_value=0.0)
        builder.objective = Minimise((x - 1.0) ** 2)
        builder.outputs = [x]
        problem = builder.build("casadi")

    objective, x_value = problem()

    assert objective == pytest.approx(0.0, abs=1e-6)
    assert x_value == pytest.approx(1.0, abs=1e-6)
    assert len(calls) == 1
    casadi_options = calls[0][0][3]
    assert casadi_options == solver_options.optimiser_options
    assert casadi_options is not solver_options.optimiser_options


def test_optimisation_supports_parameter_dependent_constraint_bounds(
    variational_backend,
):
    with ProblemBuilder(arguments=[Scalar("target")]) as builder:
        (target,) = builder.arguments
        x = builder.new_variable(name="x", initial_value=0.0)
        builder.objective = Minimise(x**2)
        builder.constraints = [bounded(x, target, target)]
        builder.outputs = [x]
        problem = builder.build(variational_backend)

    for target in (3.0, -2.0):
        _objective, x_value = problem(target)
        assert x_value == pytest.approx(target, abs=1e-6)


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

    objective, x_val = problem()
    assert objective == pytest.approx(0.5, abs=1e-6)
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

    objective, x_val = problem()
    assert objective == pytest.approx(0.25, abs=5e-4)
    assert np.allclose(x_val, np.array([0.5, 0.0]), atol=5e-4)
    assert problem.solve_info is not None
    assert problem.solve_info.success
