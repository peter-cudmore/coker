"""Result construction for dynamical-system rank analyses."""

from __future__ import annotations

import sympy as sp

from coker.backends.sympy.analysis import empty_matrix, generic_rank_witnesses

from .model import AnalysisResult, AnalysisStatus


def result(
    status: AnalysisStatus,
    matrix: sp.MatrixBase,
    generators: tuple[sp.Expr | sp.MatrixBase, ...],
    required_rank: int,
    *,
    reason: str | None = None,
    matrix_rank: int | None = None,
) -> AnalysisResult:
    """Construct a symbolic rank-analysis result with a generic witness."""
    rank = matrix.rank() if matrix_rank is None else matrix_rank
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
    return result(
        AnalysisStatus.INCONCLUSIVE,
        empty_matrix(0, required_rank) if matrix is None else matrix,
        generators,
        required_rank,
        reason=reason,
    )
