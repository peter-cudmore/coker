"""Shared result types for symbolic system analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

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
