"""Generic local structural identifiability analysis.

The augmented observability rank condition treats constant parameters as states
with zero dynamics.  It is a local, generic result: a nonzero maximal minor
witnesses the rank away from the zero set of that minor.
"""

from __future__ import annotations

import sympy as sp

from . import _rank
from .model import AnalysisResult, AnalysisStatus
from .symbolic import SymbolicSystem, UnsupportedSystemError, lower_system


def _gradient(
    expression: sp.Expr, coordinates: tuple[sp.Symbol, ...]
) -> sp.Matrix:
    return sp.Matrix(
        1,
        len(coordinates),
        [sp.diff(expression, coordinate) for coordinate in coordinates],
    )


def _lie_derivative(
    expression: sp.Expr,
    coordinates: tuple[sp.Symbol, ...],
    vector_field: tuple[sp.Expr, ...],
) -> sp.Expr:
    return sum(
        (
            sp.diff(expression, coordinate) * component
            for coordinate, component in zip(coordinates, vector_field)
        ),
        sp.S.Zero,
    )


def analyse_identifiability(
    system: object, *, max_order: int | None = None
) -> AnalysisResult:
    """Analyse generic local identifiability by augmented observability.

    Parameters are appended to the state and assigned zero dynamics.  The
    gradients of every output Lie derivative are stacked until they attain
    full augmented rank or their rank ceases to grow.  This initial method is
    only defined for autonomous systems without controls.
    """
    try:
        symbolic = (
            system
            if isinstance(system, SymbolicSystem)
            else lower_system(system)
        )
    except UnsupportedSystemError as error:
        return _rank.inconclusive(
            str(error) or "system is unsupported for analysis"
        )

    coordinates = symbolic.state + symbolic.parameters
    required_rank = len(coordinates)
    if symbolic.controls:
        return _rank.inconclusive(
            "identifiability analysis does not support controlled systems",
            required_rank=required_rank,
        )
    if max_order is not None and (
        not isinstance(max_order, int)
        or isinstance(max_order, bool)
        or max_order < 0
    ):
        return _rank.inconclusive(
            "max_order must be a non-negative integer or None",
            required_rank=required_rank,
        )

    vector_field = symbolic.dynamics + (sp.S.Zero,) * len(symbolic.parameters)
    generators = symbolic.outputs
    current_generators = symbolic.outputs
    rows = [
        _gradient(expression, coordinates) for expression in current_generators
    ]
    matrix = (
        sp.Matrix.vstack(*rows)
        if rows
        else _rank.empty_matrix(0, required_rank)
    )
    previous_rank: int | None = None
    order = 0

    while True:
        rank = matrix.rank()
        if rank == required_rank:
            return _rank.result(
                AnalysisStatus.IDENTIFIABLE,
                matrix,
                generators,
                required_rank,
            )
        if previous_rank == rank:
            return _rank.result(
                AnalysisStatus.NOT_IDENTIFIABLE,
                matrix,
                generators,
                required_rank,
            )
        if max_order is not None and order >= max_order:
            return _rank.inconclusive(
                f"maximum Lie-derivative order {max_order} reached before "
                "the augmented observability rank stabilized",
                required_rank=required_rank,
                matrix=matrix,
                generators=generators,
            )

        previous_rank = rank
        current_generators = tuple(
            _lie_derivative(expression, coordinates, vector_field)
            for expression in current_generators
        )
        generators += current_generators
        rows = [
            _gradient(expression, coordinates)
            for expression in current_generators
        ]
        matrix = sp.Matrix.vstack(matrix, *rows) if rows else matrix
        order += 1
