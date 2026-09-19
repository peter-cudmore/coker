"""Private helpers shared by symbolic rank analyses."""

from __future__ import annotations

import sympy as sp

from .model import AnalysisResult, AnalysisStatus


def empty_matrix(
    rows: int, columns: int, *, immutable: bool = False
) -> sp.MatrixBase:
    """Create an empty matrix while preserving the requested matrix kind."""
    if immutable:
        return sp.ImmutableMatrix.zeros(rows, columns)
    return sp.zeros(rows, columns)


def generic_rank_witnesses(
    matrix: sp.MatrixBase, rank: int
) -> tuple[sp.Expr, ...]:
    """Return a nonzero maximal minor that witnesses ``matrix``'s rank."""
    if rank == 0:
        return ()

    _reduced, column_indices = matrix.rref()
    column_indices = list(column_indices)
    selected_columns = matrix[:, column_indices]
    _reduced, row_indices = selected_columns.T.rref()
    row_indices = list(row_indices)
    witness = sp.factor(matrix.extract(row_indices, column_indices).det())
    if sp.simplify(witness) == 0:
        raise RuntimeError("A rank-positive matrix must have a nonzero minor")
    return (witness,)


def result(
    status: AnalysisStatus,
    matrix: sp.MatrixBase,
    generators: tuple[sp.Expr | sp.MatrixBase, ...],
    required_rank: int,
    *,
    reason: str | None = None,
) -> AnalysisResult:
    """Construct a symbolic rank-analysis result with a generic witness."""
    rank = matrix.rank()
    return AnalysisResult(
        status=status,
        rank=rank,
        required_rank=required_rank,
        matrix=matrix,
        generators=generators,
        generic_conditions=generic_rank_witnesses(matrix, rank),
        reason=reason,
    )


def inconclusive(
    reason: str,
    *,
    required_rank: int = 0,
    matrix: sp.MatrixBase | None = None,
    generators: tuple[sp.Expr | sp.MatrixBase, ...] = (),
) -> AnalysisResult:
    """Construct an inconclusive result with any available rank data."""
    matrix = empty_matrix(0, required_rank) if matrix is None else matrix
    return result(
        AnalysisStatus.INCONCLUSIVE,
        matrix,
        generators,
        required_rank,
        reason=reason,
    )
