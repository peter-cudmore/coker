"""Coker QP construction, structural planning, and compilation."""

from __future__ import annotations

import numpy as np

from coker.algebra.kernel import Tape, Tracer
from coker.algebra.sparse import SparseMatrixPattern
from coker.backends.optimisation import (
    InputBinding,
    build_initial_guess,
    build_problem_bindings,
    decision_degree,
)
from coker.backends.coker.qp_types import (
    CokerSolver,
    ExtractedQpProgram,
    OutputSlice,
    RuntimeQpProgram,
    OSQP_INFINITY,
)


def build_optimisation_problem(
    backend,
    cost: Tracer,
    constraints: list[Tracer],
    parameters: list[Tracer],
    outputs: list[Tracer],
    initial_conditions: dict[int, object],
):
    tape = cost.tape
    if tape is None:
        raise ValueError("Optimisation cost must belong to a tape")
    assert all(constraint.tape == tape for constraint in constraints)
    assert all(parameter.tape == tape for parameter in parameters)
    assert all(output.tape == tape for output in outputs)

    bindings = build_problem_bindings(
        tape, [parameter.index for parameter in parameters]
    )
    extracted = extract_qp_program(
        cost,
        constraints,
        outputs,
        bindings.decision_indices,
        bindings.decision_bindings,
        bindings.parameter_bindings,
    )
    runtime_qp = compile_qp_problem(
        backend,
        cost,
        constraints,
        outputs,
        bindings.decision_indices,
        bindings.decision_bindings,
        bindings.parameter_bindings,
        initial_conditions,
        extracted=extracted,
    )
    return CokerSolver(
        tape=tape,
        backend=backend,
        decision_bindings=bindings.decision_bindings,
        parameter_bindings=bindings.parameter_bindings,
        outputs=outputs,
        initial_guess=build_initial_guess(
            bindings.decision_bindings, initial_conditions
        ),
        runtime_qp=runtime_qp,
        extracted_qp=extracted,
    )


def validate_qp_problem(
    tape: Tape,
    cost: Tracer,
    constraints: list[Tracer],
    decision_indices: list[int],
) -> None:
    weighted_norm = getattr(cost, "weighted_norm", None)
    if weighted_norm is not None and not isinstance(
        getattr(weighted_norm.weight, "sparse_matrix_pattern", None),
        SparseMatrixPattern,
    ):
        raise ValueError(
            "Coker weighted-norm QP objectives require a "
            "SparseMatrixBuilder weight"
        )
    if not cost.dim.is_scalar():
        raise ValueError("Coker QP objective must be scalar")

    decision_index_set = set(decision_indices)
    if decision_degree(cost, tape, decision_index_set) > 2:
        raise ValueError("Coker QP objective must be at most quadratic")

    for input_index in tape.input_indicies:
        if tape.dim[input_index] is None:
            raise ValueError("Coker QP inputs must have known dimensions")

    for constraint in constraints:
        residual, _lower_bound, _upper_bound = constraint.as_halfplane_bound()
        if decision_degree(residual, tape, decision_index_set) > 1:
            raise ValueError(
                "Coker QP constraints must be affine in decision variables"
            )


def extract_qp_program(
    cost: Tracer,
    constraints: list[Tracer],
    outputs: list[Tracer],
    decision_indices: list[int],
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
    *,
    warm_start: bool = True,
) -> ExtractedQpProgram:
    tape = cost.tape
    if tape is None:
        raise ValueError("Optimisation cost must belong to a tape")

    validate_qp_problem(tape, cost, constraints, decision_indices)

    n = decision_bindings[-1].stop if decision_bindings else 0
    constraint_rows = [
        constraint.as_halfplane_bound()[0].dim.flat()
        for constraint in constraints
    ]
    row_offsets = [0]
    for row_count in constraint_rows:
        row_offsets.append(row_offsets[-1] + row_count)
    m = row_offsets[-1]
    coefficient_slices = {}
    p_indptr, p_indices = [], []
    a_indptr, a_indices = [], []

    def source_bound(value):
        if isinstance(value, Tracer):
            return value.index
        return np.asarray(value, dtype=float).reshape(-1, order="C").tolist()

    bound_data = [
        constraint.as_halfplane_bound() for constraint in constraints
    ]
    return ExtractedQpProgram(
        n=n,
        m=m,
        parameter_bindings=parameter_bindings,
        decision_bindings=decision_bindings,
        decision_indices=decision_indices,
        constraint_row_offsets=row_offsets,
        p_indptr=p_indptr,
        p_indices=p_indices,
        a_indptr=a_indptr,
        a_indices=a_indices,
        coefficient_slices=coefficient_slices,
        warm_start=warm_start,
        source_tape=tape,
        cost_node=cost.index,
        residual_nodes=[residual.index for residual, _, _ in bound_data],
        lower_nodes=[source_bound(lower) for _, lower, _ in bound_data],
        upper_nodes=[source_bound(upper) for _, _, upper in bound_data],
    )


def _pack_csc(
    nrows: int, ncols: int, entries: set[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    indptr = [0]
    indices: list[int] = []
    for column in range(ncols):
        indices.extend(
            row
            for row, entry_column in sorted(entries)
            if entry_column == column
        )
        indptr.append(len(indices))
    return indptr, indices


def compile_qp_problem(
    backend,
    cost: Tracer,
    constraints: list[Tracer],
    outputs: list[Tracer],
    decision_indices: list[int],
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
    initial_conditions: dict[int, object],
    *,
    extracted: ExtractedQpProgram | None = None,
) -> RuntimeQpProgram:
    """Compile one explicit QP payload through the Rust compiler.

    ``extracted`` is supplied by the public builder so coefficient lowering is
    performed exactly once.  The fallback keeps this low-level helper
    compatible for callers that already use it directly.
    """
    _ = backend, initial_conditions
    if extracted is None:
        extracted = extract_qp_program(
            cost,
            constraints,
            outputs,
            decision_indices,
            decision_bindings,
            parameter_bindings,
        )
    from coker.backends.coker.runtime import RuntimeQpProgram

    return RuntimeQpProgram.compile(extracted)
