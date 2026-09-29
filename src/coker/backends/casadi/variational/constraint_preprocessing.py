"""Structural preprocessing for CasADi NLP constraint rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import casadi as ca


_STRUCTURAL_COMPARISON_DEPTH = 1000


class ConstraintPreprocessingError(ValueError):
    """Raised when structurally identical rows have disjoint bounds."""


@dataclass(frozen=True)
class ConstraintRowProvenance:
    """Describe the transcription source that emitted one residual row."""

    category: str
    interval: int | None
    node: int | None
    component: int

    def describe(self) -> str:
        location = []
        if self.interval is not None:
            location.append(f"interval {self.interval}")
        if self.node is not None:
            location.append(f"node {self.node}")
        location.append(f"component {self.component}")
        return f"{self.category} ({', '.join(location)})"


@dataclass(frozen=True)
class ConstraintRow:
    """One scalar residual, its physical bounds, and all source records."""

    residual: ca.MX
    lower: float
    upper: float
    provenance: tuple[ConstraintRowProvenance, ...]


def _same_residual(left: ca.MX, right: ca.MX) -> bool:
    return bool(ca.is_equal(left, right, _STRUCTURAL_COMPARISON_DEPTH))


def _same_bounds(left: ConstraintRow, right: ConstraintRow) -> bool:
    return left.lower == right.lower and left.upper == right.upper


def _validate_group(rows: tuple[ConstraintRow, ...], tolerance: float) -> None:
    lower = max(row.lower for row in rows)
    upper = min(row.upper for row in rows)
    if lower <= upper + tolerance:
        return

    intervals = ", ".join(
        f"[{row.lower}, {row.upper}] from "
        f"{'; '.join(source.describe() for source in row.provenance)}"
        for row in rows
    )
    raise ConstraintPreprocessingError(
        "Conflicting constraint rows for residual "
        f"{rows[0].residual}: {intervals}"
    )


def preprocess_constraint_rows(
    rows: Iterable[ConstraintRow], *, tolerance: float
) -> tuple[ConstraintRow, ...]:
    """Remove exact rows and reject disjoint bound intervals structurally."""
    groups: list[list[ConstraintRow]] = []
    for row in rows:
        for group in groups:
            if _same_residual(group[0].residual, row.residual):
                group.append(row)
                break
        else:
            groups.append([row])

    retained = []
    for group in groups:
        _validate_group(tuple(group), tolerance)
        unique = []
        for row in group:
            for index, existing in enumerate(unique):
                if _same_bounds(existing, row):
                    unique[index] = ConstraintRow(
                        existing.residual,
                        existing.lower,
                        existing.upper,
                        existing.provenance + row.provenance,
                    )
                    break
            else:
                unique.append(row)
        retained.extend(unique)
    return tuple(retained)
