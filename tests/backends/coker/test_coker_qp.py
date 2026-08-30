import numpy as np
import pytest

from coker import VectorSpace
from coker.backends.coker.core import CokerBackend
from coker.backends.coker import qp_extract as coker_qp_optimisation
from coker.backends.coker.qp_extract import extract_qp_program
from coker.backends.optimisation import build_problem_bindings
from coker.toolkits.codesign import Minimise, ProblemBuilder, SolveInfo


def _build_bindings(cost, parameters):
    return build_problem_bindings(
        cost.tape, [parameter.index for parameter in parameters]
    )


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
