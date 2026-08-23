import numpy as np
import pytest

from coker import SolveFailure, SolveInfo, VectorSpace
from coker.backends.coker.core import CokerBackend
from coker.backends.coker import optimisation as coker_qp_optimisation
from coker.backends.coker.optimisation import extract_qp_program
from coker.backends.coker.runtime import RuntimeQpProgram
from coker.backends.optimisation import build_problem_bindings
from coker.toolkits.codesign import Minimise, ProblemBuilder


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


def _extract_single_qp_program_payload(extracted):
    payload = extracted.export_payload()
    assert set(payload) == {"functions", "qp_programs"}
    assert len(payload["qp_programs"]) == 1
    qp_payload = payload["qp_programs"][0]
    coefficient_function = next(
        function_payload
        for function_payload in payload["functions"]
        if function_payload["function_id"]
        == qp_payload["coefficient_function_id"]
    )
    return payload, coefficient_function, qp_payload


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

    payload, coefficient_function, qp_payload = (
        _extract_single_qp_program_payload(extracted)
    )
    embedded_plan = qp_payload["embedded_plan"]
    assert extracted.n == 2
    assert extracted.m == 2
    assert extracted.p_indptr == [0, 1, 3]
    assert extracted.p_indices == [0, 0, 1]
    assert extracted.a_indptr == [0, 2, 4]
    assert extracted.a_indices == [0, 1, 0, 1]
    assert extracted.coefficient_slices["q"].length == 2
    assert "program" not in payload
    assert qp_payload["function_id"] != qp_payload["coefficient_function_id"]
    assert (
        qp_payload["input_specs"]
        == coefficient_function["program"]["input_layer"]["inputs"]
    )
    assert qp_payload["output_spec"] == {
        "memory": {"location": 0, "count": extracted.n}
    }
    assert qp_payload["p_pattern"] == {
        "nrows": extracted.n,
        "ncols": extracted.n,
        "indptr": extracted.p_indptr,
        "indices": extracted.p_indices,
    }
    assert qp_payload["a_pattern"] == {
        "nrows": extracted.m,
        "ncols": extracted.n,
        "indptr": extracted.a_indptr,
        "indices": extracted.a_indices,
    }
    assert (
        qp_payload["coefficient_outputs"]["q"]
        == extracted.coefficient_slices["q"].to_export_dict()
    )
    assert embedded_plan["qdldl_plan"]["p_pattern"] == qp_payload["p_pattern"]
    assert embedded_plan["qdldl_plan"]["a_pattern"] == qp_payload["a_pattern"]


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
    assert extracted.p_indptr == [0, 1, 3]
    assert extracted.a_indptr == [0, 2, 4]
    assert extracted.coefficient_slices["px"].length == 3
    assert extracted.coefficient_slices["ax"].length == 4
    _payload, coefficient_function, qp_payload = (
        _extract_single_qp_program_payload(extracted)
    )
    assert "parameter_inputs" not in qp_payload
    assert "settings" not in qp_payload
    assert qp_payload["required_primal_workspace_size"] == extracted.n
    assert qp_payload["required_tangent_workspace_size"] == extracted.n
    assert (
        qp_payload["input_specs"]
        == coefficient_function["program"]["input_layer"]["inputs"]
    )
    assert (
        qp_payload["coefficient_outputs"]["px"]
        == extracted.coefficient_slices["px"].to_export_dict()
    )
    assert (
        qp_payload["coefficient_outputs"]["ax"]
        == extracted.coefficient_slices["ax"].to_export_dict()
    )
    assert qp_payload["embedded_plan"]["settings"]["warm_start"] is True
    assert qp_payload["embedded_plan"]["settings"]["linsys_solver"] == "Qdldl"


def test_runtime_qp_compile_load_solve_round_trip():
    compiled = _compile_parameterized_runtime_qp()
    loaded = RuntimeQpProgram(compiled.program)

    target_value = np.array([3.0, -1.0])
    solution, solve_info = loaded.solve((target_value,), warm_start=None)

    assert np.allclose(solution, target_value, atol=1e-6)
    assert solve_info.backend == "coker"
    assert solve_info.solver == "osqp"
    assert solve_info.success


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
        builder.outputs,
        builder.initial_conditions,
    )

    assert solver.last_solve_info is None

    [first_solution] = solver(np.array([3.0, -1.0]))
    first_info = solver.last_solve_info
    [second_solution] = solver(np.array([2.0, -1.0]))
    second_info = solver.last_solve_info

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

    (x_val,) = problem()
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
    (x_val,) = problem(target_value)
    assert np.allclose(x_val, target_value, atol=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.success


def test_coker_qp_parameterized_box_coefficients_and_solution():
    from coker.backends.coker.optimisation import _build_coefficient_function

    with ProblemBuilder(
        arguments=[
            VectorSpace("target", 2),
            VectorSpace("lower", 2),
            VectorSpace("upper", 2),
        ]
    ) as builder:
        target, lower, upper = builder.arguments
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        cost = np.dot(x - target, x - target)
        constraints = [x >= lower, x <= upper]
        bindings = _build_bindings(cost, [target, lower, upper])
        coefficient_function, slices = _build_coefficient_function(
            cost.tape,
            cost,
            constraints,
            bindings.decision_bindings,
            bindings.parameter_bindings,
        )

    parameters = (
        np.array([3.0, -4.0]),
        np.array([-1.0, -2.0]),
        np.array([2.0, 1.0]),
    )
    coefficients = np.asarray(coefficient_function(*parameters), dtype=float)
    lower_slice = slices["l"]
    upper_slice = slices["u"]
    assert np.allclose(
        coefficients[
            lower_slice.start : lower_slice.start + lower_slice.length
        ],
        [-1.0, -2.0, -2.0, -1.0],
    )
    assert np.allclose(
        coefficients[
            upper_slice.start : upper_slice.start + upper_slice.length
        ],
        [1.0e30, 1.0e30, 1.0e30, 1.0e30],
    )
    matrix_slice = slices["ax"]
    assert np.allclose(
        coefficients[
            matrix_slice.start : matrix_slice.start + matrix_slice.length
        ],
        [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0],
    )
    runtime_qp = RuntimeQpProgram.compile(
        extract_qp_program(
            cost,
            constraints,
            [x],
            bindings.decision_indices,
            bindings.decision_bindings,
            bindings.parameter_bindings,
        )
    )
    solution, info = runtime_qp.solve(parameters, warm_start=np.zeros(2))
    assert info.success
    assert np.allclose(solution, [2.0, -2.0], atol=1.0e-3)


def test_coker_qp_updates_warm_start_between_solves():
    with ProblemBuilder(arguments=[VectorSpace("target", 2)]) as builder:
        (target,) = builder.arguments
        x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
        builder.objective = Minimise(np.dot(x - target, x - target))
        builder.constraints = [x >= np.zeros(2)]
        builder.outputs = [x]
        problem = builder.build("coker")

    (first_solution,) = problem(np.array([1.0, 2.0]))
    first_info = problem.solve_info
    (second_solution,) = problem(np.array([0.5, 0.25]))
    second_info = problem.solve_info

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

    (x_val,) = problem()
    assert np.allclose(x_val, np.array([2.0, -3.0]), atol=1e-6)
    assert problem.solve_info is not None
    assert problem.solve_info.success
