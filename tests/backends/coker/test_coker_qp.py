import numpy as np
import pytest

from coker import VectorSpace
from coker.backends.coker.core import CokerBackend
from coker.backends.coker import qp_extract as coker_qp_optimisation
from coker.backends.coker.qp_extract import extract_qp_program
from coker.backends.coker.runtime import RuntimeQpProgram
from coker.backends.optimisation import build_problem_bindings
from coker.toolkits.codesign import (
    Minimise,
    ProblemBuilder,
    SolveFailure,
    SolveInfo,
    bounded,
    weighted_norm,
)


def _build_bindings(cost, parameters):
    return build_problem_bindings(
        cost.tape, [parameter.index for parameter in parameters]
    )


def _compile_parameterized_runtime_qp() -> RuntimeQpProgram:
    with ProblemBuilder(arguments=[VectorSpace("target", 2)]) as builder:
        (target,) = builder.arguments
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        cost = np.dot(x - target, x - target)
        bindings = _build_bindings(cost, [target])

    extracted = extract_qp_program(
        cost,
        [],
        [x],
        bindings.decision_indices,
        bindings.decision_bindings,
        bindings.parameter_bindings,
    )
    return RuntimeQpProgram.compile(extracted)


def test_extracts_constant_qp_coefficients():
    with ProblemBuilder() as builder:
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        cost = np.dot(x - np.array([1.0, -2.0]), x - np.array([1.0, -2.0]))
        constraints = [x[0] >= 0.0, x[1] <= 3.0]
        bindings = _build_bindings(cost, [])

    extracted = extract_qp_program(
        cost,
        constraints,
        [x],
        bindings.decision_indices,
        bindings.decision_bindings,
        bindings.parameter_bindings,
    )

    assert extracted.n == 2
    assert extracted.m == 2
    assert extracted.cost_node == cost.index
    assert len(extracted.residual_nodes) == 2
    assert all(isinstance(node, int) for node in extracted.residual_nodes)
    assert len(extracted.lower_nodes) == 2
    assert len(extracted.upper_nodes) == 2


def test_qp_preserves_structural_off_diagonal_hessian_entries():
    with ProblemBuilder() as builder:
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        cost = (x[0] + x[1]) ** 2 + x[0] ** 2 + x[1] ** 2
        bindings = _build_bindings(cost, [])

    extracted = extract_qp_program(
        cost,
        [],
        [x],
        bindings.decision_indices,
        bindings.decision_bindings,
        bindings.parameter_bindings,
    )

    assert extracted.cost_node == cost.index
    assert extracted.residual_nodes == []
    RuntimeQpProgram.compile(extracted)


def test_qp_validation_rejects_cubic_objective():
    with ProblemBuilder() as builder:
        x = builder.new_variable("x", initial_value=0.0)
        cost = x**3
        bindings = _build_bindings(cost, [])

    with pytest.raises(ValueError, match="at most quadratic"):
        extract_qp_program(
            cost,
            [],
            [x],
            bindings.decision_indices,
            bindings.decision_bindings,
            bindings.parameter_bindings,
        )


def test_qp_validation_rejects_nonlinear_constraint():
    with ProblemBuilder() as builder:
        x = builder.new_variable("x", initial_value=0.0)
        cost = x**2
        constraints = [x**2 <= 1.0]
        bindings = _build_bindings(cost, [])

    with pytest.raises(ValueError, match="affine"):
        extract_qp_program(
            cost,
            constraints,
            [x],
            bindings.decision_indices,
            bindings.decision_bindings,
            bindings.parameter_bindings,
        )


def test_extracts_parameterized_qp_coefficients():
    with ProblemBuilder(arguments=[VectorSpace("target", 2)]) as builder:
        (target,) = builder.arguments
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        cost = np.dot(x - target, x - target)
        constraints = [x >= -np.ones(2)]
        bindings = _build_bindings(cost, [target])

    extracted = extract_qp_program(
        cost,
        constraints,
        [x],
        bindings.decision_indices,
        bindings.decision_bindings,
        bindings.parameter_bindings,
    )

    assert extracted.m == 2
    assert extracted.cost_node == cost.index
    assert len(extracted.residual_nodes) == 1
    assert isinstance(extracted.residual_nodes[0], int)
    assert len(extracted.lower_nodes) == 1
    assert len(extracted.upper_nodes) == 1


def test_runtime_qp_compile_load_solve_round_trip():
    compiled = _compile_parameterized_runtime_qp()
    loaded = RuntimeQpProgram(compiled.program)

    target_value = np.array([3.0, -1.0])
    solution, solve_info = loaded.solve((target_value,), warm_start=None)

    assert np.allclose(solution, target_value, atol=1e-6)
    assert solve_info.backend == "coker"
    assert solve_info.solver == "osqp"
    assert solve_info.success


def test_qp_artifact_path_does_not_retain_legacy_graph_lowering():
    """QP coefficient compilation must be entirely Rust-owned."""
    assert "create_opgraph" not in vars(coker_qp_optimisation)
    compiled = _compile_parameterized_runtime_qp()
    loaded = RuntimeQpProgram(bytes(compiled.program))

    solution, solve_info = loaded.solve(
        (np.array([3.0, -1.0]),), warm_start=None
    )

    assert np.allclose(solution, np.array([3.0, -1.0]), atol=1e-6)
    assert solve_info.success


def test_runtime_qp_solve_into_reuses_caller_buffer_after_source_lifecycle():
    compiled = _compile_parameterized_runtime_qp()
    source = bytes(compiled.program)
    loaded = RuntimeQpProgram(source)
    del source

    target_value = np.array([3.0, -1.0], dtype=np.float32)
    output = np.empty(2, dtype=np.float64)
    success, status = loaded._runtime.solve_into([target_value], output, None)

    assert success
    assert status == "Solved"
    assert np.allclose(output, target_value, atol=1e-6)


def test_runtime_qp_push_forward_matches_parameter_only_contract():
    compiled = _compile_parameterized_runtime_qp()
    loaded = RuntimeQpProgram(compiled.program)

    target_value = np.array([3.0, -1.0])
    target_tangent = np.array([-0.25, 0.5])

    try:
        solution, tangent = loaded.push_forward(target_value, target_tangent)
    except ValueError as error:
        assert str(error) == (
            "QP push-forward is unsupported: differentiated "
            "KKT solve support is not implemented"
        )
    else:
        assert np.allclose(solution, target_value, atol=1e-6)
        assert np.allclose(tangent, target_tangent, atol=1e-6)


def test_coker_backend_builder_returns_casadi_style_solver(monkeypatch):
    class FakeRuntimeQpProgram:
        def __init__(self):
            self.warm_starts = []

        def solve(self, runtime_args, *, warm_start):
            _ = runtime_args
            call_index = len(self.warm_starts)
            solution = (
                np.array([1.5, -0.5])
                if call_index < 2
                else np.array([2.0, -1.0])
            )
            self.warm_starts.append(np.asarray(warm_start, dtype=float).copy())
            return solution, SolveInfo(
                backend="coker",
                solver="osqp",
                success=True,
                return_status="solved",
            )

    fake_runtime_qp = FakeRuntimeQpProgram()
    monkeypatch.setattr(
        coker_qp_optimisation,
        "compile_qp_problem",
        lambda *args, **kwargs: fake_runtime_qp,
    )

    with ProblemBuilder(arguments=[VectorSpace("target", 2)]) as builder:
        (target,) = builder.arguments
        x = builder.new_variable(
            "x", shape=(2,), initial_value=np.array([-4.0, 5.0])
        )
        builder.objective = Minimise(np.dot(x - target, x - target))
        builder.outputs = [x]

    solver = CokerBackend().build_optimisation_problem(
        builder.objective.expression,
        builder.constraints,
        builder.arguments,
        [builder.objective.expression, *builder.outputs],
        builder.initial_conditions,
    )

    assert solver.last_solve_info is None

    first_objective, first_solution = solver(np.array([3.0, -1.0]))
    first_info = solver.last_solve_info
    second_objective, second_solution = solver(np.array([2.0, -1.0]))
    second_info = solver.last_solve_info

    assert first_objective == pytest.approx(2.5, abs=1e-6)
    assert second_objective == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(first_solution, np.array([1.5, -0.5]), atol=1e-6)
    assert np.allclose(second_solution, np.array([2.0, -1.0]), atol=1e-6)
    assert first_info is not None
    assert second_info is not None
    assert first_info.success
    assert second_info.success
    assert np.allclose(fake_runtime_qp.warm_starts[0], np.array([-4.0, 5.0]))
    assert np.allclose(fake_runtime_qp.warm_starts[2], first_solution)


def test_coker_qp_solves_fixed_problem():
    with ProblemBuilder() as builder:
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        builder.objective = Minimise(
            np.dot(x - np.array([1.0, -2.0]), x - np.array([1.0, -2.0]))
        )
        builder.constraints = [x[0] >= 0.0, x[1] <= 3.0]
        builder.outputs = [x]
        problem = builder.build("coker")

    objective, x_val = problem()
    assert objective == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(x_val, np.array([1.0, -2.0]), atol=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.backend == "coker"
    assert problem.solve_info.success


def test_coker_qp_solves_parameterized_problem():
    with ProblemBuilder(arguments=[VectorSpace("target", 2)]) as builder:
        (target,) = builder.arguments
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        builder.objective = Minimise(np.dot(x - target, x - target))
        builder.constraints = [x >= -np.ones(2)]
        builder.outputs = [x]
        problem = builder.build("coker")

    target_value = np.array([3.0, -1.0])
    objective, x_val = problem(target_value)
    assert objective == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(x_val, target_value, atol=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_coker_qp_rejects_dense_weighted_norm():
    with ProblemBuilder() as builder:
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        cost = weighted_norm(np.eye(2), x)
        bindings = _build_bindings(cost, [])

    with pytest.raises(ValueError, match="SparseMatrixBuilder"):
        extract_qp_program(
            cost,
            [],
            [x],
            bindings.decision_indices,
            bindings.decision_bindings,
            bindings.parameter_bindings,
        )


def test_coker_qp_updates_warm_start_between_solves():
    with ProblemBuilder(arguments=[VectorSpace("target", 2)]) as builder:
        (target,) = builder.arguments
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        builder.objective = Minimise(np.dot(x - target, x - target))
        builder.constraints = [x >= np.zeros(2)]
        builder.outputs = [x]
        problem = builder.build("coker")

    first_objective, first_solution = problem(np.array([1.0, 2.0]))
    first_info = problem.solve_info
    second_objective, second_solution = problem(np.array([0.5, 0.25]))
    second_info = problem.solve_info

    assert first_objective == pytest.approx(0.0, abs=1e-6)
    assert second_objective == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(first_solution, np.array([1.0, 2.0]), atol=1e-6)
    assert np.allclose(second_solution, np.array([0.5, 0.25]), atol=1e-6)
    assert first_info is not None
    assert second_info is not None
    assert first_info.success
    assert second_info.success
    assert second_info is not first_info


def test_coker_qp_reports_infeasible_solve_info():
    with ProblemBuilder() as builder:
        x = builder.new_variable("x", shape=(1,), initial_value=np.zeros(1))
        builder.objective = Minimise(np.dot(x, x))
        builder.constraints = [x[0] >= 1.0, x[0] <= 0.0]
        builder.outputs = [x]
        problem = builder.build("coker")

    with pytest.raises(SolveFailure) as exc_info:
        problem()

    assert problem.solve_info is not None
    assert not problem.solve_info.success
    assert exc_info.value.solve_info == problem.solve_info


def test_coker_qp_solves_unconstrained_problem():
    with ProblemBuilder() as builder:
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        builder.objective = Minimise(
            np.dot(x - np.array([2.0, -3.0]), x - np.array([2.0, -3.0]))
        )
        builder.outputs = [x]
        problem = builder.build("coker")

    objective, x_val = problem()
    assert objective == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(x_val, np.array([2.0, -3.0]), atol=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_coker_qp_accepts_parameterized_objective_and_two_sided_bounds():
    """The caller supplies targets and bounds, never a preassembled Hessian."""
    with ProblemBuilder(
        arguments=[
            VectorSpace("target", 2),
            VectorSpace("lower", 2),
            VectorSpace("upper", 2),
        ]
    ) as builder:
        target, lower, upper = builder.arguments
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        builder.objective = Minimise(np.dot(x - target, x - target))
        builder.constraints = [bounded(x, lower, upper)]
        builder.outputs = [x]
        problem = builder.build("coker")

    objective, solution = problem(
        np.array([3.0, -1.0]),
        np.array([-1.0, -2.0]),
        np.array([2.0, 1.0]),
    )
    assert objective == pytest.approx(1.0, abs=1e-3)

    assert problem.solve_info is not None
    assert problem.solve_info.success
    assert np.allclose(solution, np.array([2.0, -1.0]), atol=1.0e-3)
