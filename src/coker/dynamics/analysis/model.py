"""Typed results and shared rank-closure operations for system analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, TypeVar

import sympy as sp

from coker.backends.sympy.analysis import empty_matrix, generic_rank_witnesses

__all__ = (
    "AnalysisStatus",
    "AnalysisResult",
    "ControllabilityResult",
    "IdentifiabilityResult",
    "ObservabilityResult",
)


class AnalysisStatus(Enum):
    """Whether a rank analysis established, disproved, or cannot decide.

    The result specialization names the analysis property.
    """

    TRUE = True
    FALSE = False
    INCONCLUSIVE = None


@dataclass(frozen=True)
class AnalysisResult:
    """Base result for a symbolic rank analysis."""

    status: AnalysisStatus
    rank: int
    required_rank: int
    matrix: sp.MatrixBase
    generators: tuple[sp.Expr | sp.MatrixBase, ...]
    rank_conditions: tuple[sp.Expr, ...]
    reason: str | None = None


@dataclass(frozen=True)
class ControllabilityResult(AnalysisResult):
    """Generic local accessibility rank result."""


@dataclass(frozen=True)
class ObservabilityResult(AnalysisResult):
    """Generic local observability rank result."""


@dataclass(frozen=True)
class IdentifiabilityResult(AnalysisResult):
    """Generic local structural-identifiability rank result."""


Result = TypeVar("Result", bound=AnalysisResult)
Generator = tuple[sp.Expr, ...]


def create_result(
    result_type: type[Result],
    status: AnalysisStatus,
    matrix: sp.MatrixBase,
    generators: tuple[sp.Expr | sp.MatrixBase, ...],
    required_rank: int,
    *,
    reason: str | None = None,
    matrix_rank: int | None = None,
) -> Result:
    """Create a typed rank-analysis result with a generic witness."""
    rank = matrix.rank() if matrix_rank is None else matrix_rank
    return result_type(
        status=status,
        rank=rank,
        required_rank=required_rank,
        matrix=matrix,
        generators=generators,
        rank_conditions=generic_rank_witnesses(matrix, rank),
        reason=reason,
    )


def create_inconclusive(
    result_type: type[Result],
    reason: str,
    *,
    required_rank: int = 0,
    matrix: sp.MatrixBase | None = None,
    generators: tuple[sp.Expr | sp.MatrixBase, ...] = (),
) -> Result:
    """Create a typed inconclusive result with available rank data."""
    return create_result(
        result_type,
        AnalysisStatus.INCONCLUSIVE,
        empty_matrix(0, required_rank) if matrix is None else matrix,
        generators,
        required_rank,
        reason=reason,
    )


def compute_rank_closure(
    result_type: type[Result],
    generators: Generator,
    required_rank: int,
    compute_gradient: Callable[[sp.Expr], sp.MatrixBase],
    compute_next: Callable[[Generator], Generator],
    max_order: int | None,
    limit_reason: str,
) -> Result:
    """Compute the rank closure of scalar generator expressions."""
    current = generators
    rows = [compute_gradient(expression) for expression in current]
    matrix = (
        sp.Matrix.vstack(*rows) if rows else empty_matrix(0, required_rank)
    )
    previous_rank: int | None = None
    order = 0

    while True:
        rank = matrix.rank()
        if rank == required_rank:
            return create_result(
                result_type,
                AnalysisStatus.TRUE,
                matrix,
                generators,
                required_rank,
                matrix_rank=rank,
            )
        if previous_rank == rank:
            return create_result(
                result_type,
                AnalysisStatus.FALSE,
                matrix,
                generators,
                required_rank,
                matrix_rank=rank,
            )
        if max_order is not None and order >= max_order:
            return create_inconclusive(
                result_type,
                limit_reason,
                required_rank=required_rank,
                matrix=matrix,
                generators=generators,
            )
        previous_rank = rank
        current = compute_next(current)
        generators += current
        rows = [compute_gradient(expression) for expression in current]
        matrix = sp.Matrix.vstack(matrix, *rows) if rows else matrix
        order += 1


def add_condition(result: Result, condition: sp.Expr) -> Result:
    """Add a DAE regularity condition to a rank result."""
    return replace(
        result, rank_conditions=(condition, *result.rank_conditions)
    )
