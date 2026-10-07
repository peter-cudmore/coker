"""Structural preprocessing for CasADi NLP constraint rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import casadi as ca
import numpy as np


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
        "Conflicting constraint rows for residual " f"{rows[0].residual}: {intervals}"
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


def reduce_affine_equality_rows(
    rows: Iterable[ConstraintRow],
    variables: ca.MX,
    *,
    tolerance: float,
) -> tuple[ConstraintRow, ...]:
    """Remove equality rows implied by an independent affine row basis."""
    rows = tuple(rows)
    equalities = [
        row
        for row in rows
        if row.lower == row.upper and ca.is_linear(row.residual, variables)
    ]
    if len(equalities) < 2:
        return rows

    residuals = ca.vertcat(*(row.residual for row in equalities))
    evaluate = ca.Function(
        "affine_rows",
        [variables],
        [residuals, ca.jacobian(residuals, variables)],
    )
    constants, coefficients = evaluate(ca.DM.zeros(variables.numel(), 1))
    coefficients = np.asarray(coefficients, dtype=float)
    constants = np.asarray(constants, dtype=float).reshape(-1)
    retained = []
    coefficient_basis = np.empty((0, coefficients.shape[1]))
    augmented_basis = np.empty((0, coefficients.shape[1] + 1))
    for row, coefficient, constant in zip(equalities, coefficients, constants):
        augmented = np.append(coefficient, constant - row.lower)
        coefficient_candidate = np.vstack((coefficient_basis, coefficient))
        augmented_candidate = np.vstack((augmented_basis, augmented))
        if np.linalg.matrix_rank(
            coefficient_candidate, tolerance
        ) > np.linalg.matrix_rank(coefficient_basis, tolerance):
            retained.append(row)
            coefficient_basis = coefficient_candidate
            augmented_basis = augmented_candidate
        elif np.linalg.matrix_rank(
            augmented_candidate, tolerance
        ) > np.linalg.matrix_rank(augmented_basis, tolerance):
            raise ConstraintPreprocessingError(
                "Inconsistent affine equality row from "
                f"{'; '.join(source.describe() for source in row.provenance)}"
            )
    retained_ids = {id(row) for row in retained}
    equality_ids = {id(row) for row in equalities}
    return tuple(
        row for row in rows if id(row) not in equality_ids or id(row) in retained_ids
    )
