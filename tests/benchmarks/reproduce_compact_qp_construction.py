"""Reproduce compact sparse QP construction without Hexapod dependencies."""

from __future__ import annotations

import time

import numpy as np

from coker import SparseMatrixBuilder, VectorSpace
from coker.toolkits.codesign import (
    Minimise,
    ProblemBuilder,
    bounded,
    weighted_norm,
)
from coker_backend.optimisation import extract_qp_program
from coker_backend.runtime import RuntimeQpProgram
from coker.backends.optimisation import build_problem_bindings


DECISIONS = 24
OBJECTIVE_ROWS = 78
CONSTRAINT_ROWS = 66
ROW_SUPPORT = 9


def _fixed_pattern(rows: int, columns: int) -> np.ndarray:
    """Return a fixed nine-column-per-row sparse pattern."""
    pattern = np.zeros((rows, columns), dtype=bool)
    for row in range(rows):
        first = (3 * row) % columns
        pattern[
            row,
            [(first + offset) % columns for offset in range(ROW_SUPPORT)],
        ] = True
    return pattern


def build_compact_qp():
    """Build the direct weighted-norm QP used to isolate construction cost."""
    objective_pattern = _fixed_pattern(OBJECTIVE_ROWS, DECISIONS)
    constraint_pattern = _fixed_pattern(CONSTRAINT_ROWS, DECISIONS)
    objective_data = SparseMatrixBuilder(objective_pattern)
    objective_weight = SparseMatrixBuilder(
        np.column_stack(
            [objective_pattern, np.ones(OBJECTIVE_ROWS, dtype=bool)]
        )
    )
    constraint_data = SparseMatrixBuilder(constraint_pattern)

    with ProblemBuilder(
        arguments=[
            objective_data.data_space("objective_values"),
            VectorSpace("objective_target", OBJECTIVE_ROWS),
            constraint_data.data_space("constraint_values"),
            VectorSpace("lower", CONSTRAINT_ROWS),
            VectorSpace("upper", CONSTRAINT_ROWS),
        ]
    ) as builder:
        objective_values, target, constraint_values, lower, upper = (
            builder.arguments
        )
        decision = builder.new_variable(
            "decision", shape=(DECISIONS,), initial_value=np.zeros(DECISIONS)
        )
        weight = objective_weight.matrix(
            np.concatenate([objective_values, -target])
        )
        constraints = constraint_data.matrix(constraint_values) @ decision
        augmented_decision = np.concatenate([decision, np.ones(1)])
        builder.objective = Minimise(weighted_norm(weight, augmented_decision))
        builder.constraints = [bounded(constraints, lower, upper)]
        builder.outputs = [decision]
        bindings = build_problem_bindings(
            builder.objective.expression.tape,
            [argument.index for argument in builder.arguments],
        )
        print("extracting compact QP coefficients", flush=True)
        extracted = extract_qp_program(
            builder.objective.expression,
            builder.constraints,
            builder.outputs,
            bindings.decision_indices,
            bindings.decision_bindings,
            bindings.parameter_bindings,
        )
        print("compiling compact QP module", flush=True)
        return RuntimeQpProgram.compile(extracted)


if __name__ == "__main__":
    start = time.perf_counter()
    program = build_compact_qp()
    elapsed = time.perf_counter() - start
    if elapsed >= 90.0:
        raise RuntimeError(
            f"compact QP construction exceeded 90s: {elapsed:.3f}s"
        )
    print(f"compact QP module bytes: {len(program.program)}")
    print(f"compact QP construction: {elapsed:.3f}s")
