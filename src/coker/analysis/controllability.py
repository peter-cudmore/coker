"""Generic local accessibility analysis for control-affine ODEs."""

from __future__ import annotations

from itertools import combinations

import sympy as sp

from .model import AnalysisResult, AnalysisStatus
from .symbolic import SymbolicSystem, UnsupportedSystemError, lower_system


def _zero_matrix(rows: int) -> sp.ImmutableMatrix:
    return sp.ImmutableMatrix.zeros(rows, 0)


def _as_field(values) -> sp.ImmutableMatrix:
    return sp.ImmutableMatrix(len(values), 1, values)


def _simplify_field(field: sp.MatrixBase) -> sp.ImmutableMatrix:
    return sp.ImmutableMatrix(
        field.rows,
        field.cols,
        [sp.simplify(entry) for entry in field],
    )


def _is_zero(expression: sp.Expr) -> bool:
    return sp.simplify(expression) == 0


def _inconclusive(
    reason: str,
    *,
    required_rank: int = 0,
    matrix: sp.ImmutableMatrix | None = None,
    generators: tuple[sp.ImmutableMatrix, ...] = (),
) -> AnalysisResult:
    if matrix is None:
        matrix = _zero_matrix(required_rank)
    rank = matrix.rank()
    return AnalysisResult(
        status=AnalysisStatus.INCONCLUSIVE,
        rank=rank,
        required_rank=required_rank,
        matrix=matrix,
        generators=generators,
        generic_conditions=_rank_witnesses(matrix, rank),
        reason=reason,
    )


def _rank_witnesses(
    matrix: sp.ImmutableMatrix, rank: int
) -> tuple[sp.Expr, ...]:
    """Return a nonzero maximal minor which witnesses the generic rank."""
    if rank == 0:
        return ()

    column_indices = tuple(range(matrix.cols))
    for row_indices in combinations(range(matrix.rows), rank):
        minor = sp.simplify(matrix.extract(row_indices, column_indices).det())
        if not _is_zero(minor):
            return (minor,)

    raise RuntimeError("A rank-positive matrix must have a nonzero minor")


def _matrix_from_fields(
    fields: tuple[sp.ImmutableMatrix, ...], state_dimension: int
) -> sp.ImmutableMatrix:
    if not fields:
        return _zero_matrix(state_dimension)
    return sp.ImmutableMatrix.hstack(*fields)


def _append_if_independent(
    fields: tuple[sp.ImmutableMatrix, ...],
    rank: int,
    candidate: sp.ImmutableMatrix,
    state_dimension: int,
) -> tuple[tuple[sp.ImmutableMatrix, ...], int, bool]:
    """Append ``candidate`` exactly when it raises the generic field rank."""
    candidate = _simplify_field(candidate)
    matrix = _matrix_from_fields((*fields, candidate), state_dimension)
    candidate_rank = matrix.rank()
    if candidate_rank <= rank:
        return fields, rank, False
    return (*fields, candidate), candidate_rank, True


def _extract_control_affine_fields(
    system: SymbolicSystem,
) -> tuple[sp.ImmutableMatrix, tuple[sp.ImmutableMatrix, ...]] | None:
    """Split dynamics into its drift and input vector fields.

    Returning ``None`` means that the dynamics are not affine in every
    control.  The first- and second-derivative checks also reject a control
    coefficient which depends on another control.
    """
    controls = system.controls
    dynamics = system.dynamics
    zero_controls = {control: sp.S.Zero for control in controls}

    try:
        control_fields = tuple(
            _as_field(
                tuple(sp.diff(component, control) for component in dynamics)
            )
            for control in controls
        )
        for field in control_fields:
            for component in field:
                for control in controls:
                    if not _is_zero(sp.diff(component, control)):
                        return None

        drift = _as_field(
            tuple(
                sp.simplify(component.subs(zero_controls))
                for component in dynamics
            )
        )
        for index, component in enumerate(dynamics):
            reconstructed = drift[index] + sum(
                field[index] * control
                for field, control in zip(control_fields, controls)
            )
            if not _is_zero(component - reconstructed):
                return None
    except (NotImplementedError, TypeError, ValueError):
        return None

    return _simplify_field(drift), tuple(
        _simplify_field(field) for field in control_fields
    )


def _lie_bracket(
    left: sp.ImmutableMatrix,
    right: sp.ImmutableMatrix,
    state: tuple[sp.Symbol, ...],
) -> sp.ImmutableMatrix:
    """Compute ``[left, right] = D(right) left - D(left) right``."""
    state_vector = sp.ImmutableMatrix(state)
    return _simplify_field(
        right.jacobian(state_vector) * left
        - left.jacobian(state_vector) * right
    )


def _result(
    status: AnalysisStatus,
    fields: tuple[sp.ImmutableMatrix, ...],
    state_dimension: int,
    reason: str | None = None,
) -> AnalysisResult:
    matrix = _matrix_from_fields(fields, state_dimension)
    rank = matrix.rank()
    return AnalysisResult(
        status=status,
        rank=rank,
        required_rank=state_dimension,
        matrix=matrix,
        generators=fields,
        generic_conditions=_rank_witnesses(matrix, rank),
        reason=reason,
    )


def analyse_controllability(
    system, *, max_order: int | None = None
) -> AnalysisResult:
    """Analyse generic local accessibility through the Lie algebra rank test.

    This establishes only generic local accessibility.  It deliberately does
    not make a global controllability claim.  ``max_order`` bounds the Lie
    bracket depth; control fields seed the distribution and the drift
    participates in the recursive brackets.
    """
    try:
        symbolic = lower_system(system)
    except UnsupportedSystemError as error:
        return _inconclusive(
            str(error) or "The system cannot be represented symbolically."
        )

    state_dimension = len(symbolic.state)
    if not isinstance(max_order, int) or isinstance(max_order, bool):
        if max_order is not None:
            return _inconclusive(
                "max_order must be a non-negative integer or None.",
                required_rank=state_dimension,
            )
    elif max_order < 0:
        return _inconclusive(
            "max_order must be a non-negative integer or None.",
            required_rank=state_dimension,
        )

    affine_fields = _extract_control_affine_fields(symbolic)
    if affine_fields is None:
        return _inconclusive(
            "Dynamics are not affine in the controls.",
            required_rank=state_dimension,
        )

    drift, controls = affine_fields
    base_fields = (drift, *controls)
    fields: tuple[sp.ImmutableMatrix, ...] = ()
    rank = 0
    frontier: tuple[sp.ImmutableMatrix, ...] = ()
    for field in controls:
        fields, rank, added = _append_if_independent(
            fields, rank, field, state_dimension
        )
        if added:
            frontier = (*frontier, field)

    order = 0
    while True:
        if rank == state_dimension:
            return _result(AnalysisStatus.ACCESSIBLE, fields, state_dimension)
        if not frontier:
            return _result(
                AnalysisStatus.NOT_ACCESSIBLE, fields, state_dimension
            )
        if max_order is not None and order >= max_order:
            return _result(
                AnalysisStatus.INCONCLUSIVE,
                fields,
                state_dimension,
                "Lie-bracket expansion reached max_order before closure.",
            )

        next_frontier: tuple[sp.ImmutableMatrix, ...] = ()
        for field in frontier:
            for base_field in base_fields:
                bracket = _lie_bracket(field, base_field, symbolic.state)
                fields, rank, added = _append_if_independent(
                    fields, rank, bracket, state_dimension
                )
                if added:
                    next_frontier = (*next_frontier, bracket)

        if not next_frontier:
            return _result(
                AnalysisStatus.NOT_ACCESSIBLE, fields, state_dimension
            )
        frontier = next_frontier
        order += 1
