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
    """Outcome of a generic local symbolic rank test.

    Attributes:
        TRUE: The required rank was reached, subject to ``rank_conditions``.
        FALSE: The rank closure stabilized below the required rank.
        INCONCLUSIVE: Unsupported input, an invalid analysis option, or an
            expansion limit prevented a decision; inspect the result's
            ``reason``. This is not evidence that the property is false.

    Examples:
        >>> from coker.dynamics import AnalysisStatus
        >>> AnalysisStatus.INCONCLUSIVE.value is None
        True
    """

    TRUE = True
    FALSE = False
    INCONCLUSIVE = None


@dataclass(frozen=True)
class AnalysisResult:
    """Immutable evidence returned by a generic local symbolic rank test.

    These results describe generic local rank, not global behavior or a
    guarantee at every state or parameter value. Inspect ``status`` explicitly:
    a result object itself is not a boolean verdict.

    Attributes:
        status: Whether the test reached full rank, closed at deficient rank,
            or could not decide.
        rank: Symbolic rank of the available matrix.
        required_rank: Differential-state dimension for accessibility and
            observability; differential-state plus fitted-parameter dimension
            for identifiability. May be zero if lowering failed before the
            dimension was known.
        matrix: Rank-test matrix. Accessibility vector fields are columns;
            observability and identifiability gradients are rows.
        generators: Retained vector fields for accessibility, or output and
            generated Lie-derivative expressions for the gradient tests.
        rank_conditions: Expressions whose nonvanishing witnesses the reported
            generic rank. DAE results may additionally include the determinant
            of the algebraic constraint Jacobian as a regularity condition.
            These witnesses are not an exhaustive classification of singular
            points; an empty tuple does not certify global validity.
        reason: Explanation for an inconclusive outcome, otherwise usually
            ``None``.

    Examples:
        >>> import sympy as sp
        >>> from coker.dynamics import AnalysisResult, AnalysisStatus
        >>> result = AnalysisResult(
        ...     status=AnalysisStatus.TRUE, rank=1, required_rank=1,
        ...     matrix=sp.eye(1), generators=(sp.Integer(1),),
        ...     rank_conditions=(),
        ... )
        >>> result.rank == result.required_rank
        True
    """

    status: AnalysisStatus
    rank: int
    required_rank: int
    matrix: sp.MatrixBase
    generators: tuple[sp.Expr | sp.MatrixBase, ...]
    rank_conditions: tuple[sp.Expr, ...]
    reason: str | None = None


@dataclass(frozen=True)
class ControllabilityResult(AnalysisResult):
    """Generic local accessibility evidence, with fields from AnalysisResult.

    A full-rank result establishes the implemented accessibility rank
    criterion, not global controllability or small-time local controllability.
    ``matrix`` contains independent vector-field columns and ``generators``
    contains those fields. See ``analyse_controllability`` for runnable usage.

    Examples:
        >>> import sympy as sp
        >>> from coker.dynamics import (
        ...     ControllabilityResult, SymbolicSystem, analyse_controllability,
        ... )
        >>> x, u = sp.symbols("x u")
        >>> result = analyse_controllability(
        ...     SymbolicSystem((x,), (), (u,), (u,), (x,))
        ... )
        >>> isinstance(result, ControllabilityResult)
        True
    """


@dataclass(frozen=True)
class ObservabilityResult(AnalysisResult):
    """Generic local observability evidence, with fields from AnalysisResult.

    ``matrix`` contains output/Lie-derivative gradient rows, restricted to the
    constraint tangent space for DAEs. This is not a global reconstruction
    guarantee. See ``analyse_observability`` for runnable usage.

    Examples:
        >>> import sympy as sp
        >>> from coker.dynamics import (
        ...     ObservabilityResult, SymbolicSystem, analyse_observability,
        ... )
        >>> x = sp.Symbol("x")
        >>> result = analyse_observability(
        ...     SymbolicSystem((x,), (), (), (-x,), (x,))
        ... )
        >>> isinstance(result, ObservabilityResult)
        True
    """


@dataclass(frozen=True)
class IdentifiabilityResult(AnalysisResult):
    """Augmented observability evidence, with fields from AnalysisResult.

    Columns correspond to differential states followed by fitted parameters.
    Full rank certifies the implemented joint state/parameter local criterion,
    not global parameter uniqueness. See ``analyse_identifiability`` for usage.

    Examples:
        >>> import sympy as sp
        >>> from coker.dynamics import (
        ...     IdentifiabilityResult, SymbolicSystem, analyse_identifiability,
        ... )
        >>> x, p = sp.symbols("x p")
        >>> result = analyse_identifiability(
        ...     SymbolicSystem((x,), (p,), (), (p * x,), (x,))
        ... )
        >>> isinstance(result, IdentifiabilityResult)
        True
    """


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
    matrix = sp.Matrix.vstack(*rows) if rows else empty_matrix(0, required_rank)
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
    return replace(result, rank_conditions=(condition, *result.rank_conditions))
