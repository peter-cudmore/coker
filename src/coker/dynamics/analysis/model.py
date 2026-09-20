"""Shared result types for symbolic system analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from coker.backends.sympy.analysis import empty_matrix, generic_rank_witnesses

__all__ = ("AnalysisStatus", "AnalysisResult")


class AnalysisStatus(str, Enum):
    """The outcome of a generic local system-analysis rank test."""

    ACCESSIBLE = "accessible"
    NOT_ACCESSIBLE = "not_accessible"
    IDENTIFIABLE = "identifiable"
    NOT_IDENTIFIABLE = "not_identifiable"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class AnalysisResult:
    """A symbolic rank-analysis result.

    ``rank`` and ``required_rank`` describe the generic local rank witnessed
    by ``matrix``.  ``generic_conditions`` records nonzero expressions under
    which that rank witness applies.
    """

    status: AnalysisStatus
    rank: int
    required_rank: int
    matrix: sp.MatrixBase
    generators: tuple[sp.Expr | sp.MatrixBase, ...]
    generic_conditions: tuple[sp.Expr, ...]
    reason: str | None = None


def result(
    status: AnalysisStatus,
    matrix: sp.MatrixBase,
    generators: tuple[sp.Expr | sp.MatrixBase, ...],
    required_rank: int,
    *,
    reason: str | None = None,
    matrix_rank: int | None = None,
) -> AnalysisResult:
    """Construct a rank-analysis result with a generic witness."""
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
