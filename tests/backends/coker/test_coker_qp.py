import numpy as np
import pytest

from coker import VectorSpace
from coker.backends.coker.optimisation import extract_qp_program
from coker.backends.optimisation import build_problem_bindings
from coker.toolkits.codesign import ProblemBuilder


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
    assert [(entry.row, entry.col) for entry in extracted.p_structure] == [
        (0, 0),
        (0, 1),
        (1, 1),
    ]
    assert [(entry.row, entry.col) for entry in extracted.a_structure] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]
    assert extracted.coefficient_slices["q"].length == 2
    assert extracted.coefficient_slices["l"].length == 2


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
    assert extracted.coefficient_slices["px"].length == 3
    assert extracted.coefficient_slices["ax"].length == 4
    assert extracted.export_payload()["program"]["parameter_inputs"] == [
        {"length": 2}
    ]
