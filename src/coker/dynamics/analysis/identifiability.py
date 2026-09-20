"""Generic local structural identifiability analysis."""

from __future__ import annotations

from dataclasses import replace
from numbers import Real
from typing import Sequence

import sympy as sp

from coker.algebra.ops import Noop
from coker.dynamics.model import DynamicalSystem
from coker.dynamics.variables import BoundedVariable, UnboundedVariable

from . import _rank
from .dae import SymbolicDAESystem, geometry, lower_dae_system
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


def _add_dae_condition(
    result: AnalysisResult, condition: sp.Expr
) -> AnalysisResult:
    return replace(
        result, generic_conditions=(condition, *result.generic_conditions)
    )


def _apply_parameter_roles(
    symbolic: SymbolicSystem | SymbolicDAESystem,
    declarations: Sequence[BoundedVariable | UnboundedVariable | Real] | None,
) -> SymbolicSystem | SymbolicDAESystem:
    """Substitute known parameters and retain fitted parameters."""
    if declarations is None:
        return symbolic
    if len(declarations) != len(symbolic.parameters):
        raise UnsupportedSystemError(
            "parameter declarations must match the flattened parameter count"
        )

    replacements: dict[sp.Symbol, sp.Expr] = {}
    fitted: list[sp.Symbol] = []
    for symbol, declaration in zip(symbolic.parameters, declarations):
        if isinstance(declaration, (BoundedVariable, UnboundedVariable)):
            fitted.append(symbol)
        elif isinstance(declaration, Real) and not isinstance(
            declaration, bool
        ):
            replacements[symbol] = sp.sympify(declaration)
        else:
            raise UnsupportedSystemError(
                "each scalar parameter must be a BoundedVariable, "
                "UnboundedVariable, or numeric constant"
            )
    if not replacements:
        return symbolic

    fields = {
        "parameters": tuple(fitted),
        "dynamics": tuple(
            expression.xreplace(replacements)
            for expression in symbolic.dynamics
        ),
        "outputs": tuple(
            expression.xreplace(replacements)
            for expression in symbolic.outputs
        ),
    }
    if isinstance(symbolic, SymbolicDAESystem):
        fields["constraints"] = tuple(
            expression.xreplace(replacements)
            for expression in symbolic.constraints
        )
    return replace(symbolic, **fields)


def analyse_identifiability(
    system: object,
    *,
    parameters: (
        Sequence[BoundedVariable | UnboundedVariable | Real] | None
    ) = None,
    max_order: int | None = None,
) -> AnalysisResult:
    """Analyse generic local identifiability by augmented observability."""
    try:
        symbolic = (
            lower_dae_system(system)
            if isinstance(system, DynamicalSystem)
            and not isinstance(system.g, Noop)
            else (
                system
                if isinstance(system, SymbolicSystem)
                else lower_system(system)
            )
        )
        symbolic = _apply_parameter_roles(symbolic, parameters)
        tangent = (
            geometry(symbolic)
            if isinstance(symbolic, SymbolicDAESystem)
            else None
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

    if tangent:
        algebraic_velocity = tangent.lift(sp.Matrix(symbolic.dynamics))
        full_coordinates = (
            symbolic.state + symbolic.algebraic + symbolic.parameters
        )
        vector_field = (
            symbolic.dynamics
            + tuple(algebraic_velocity)
            + (sp.S.Zero,) * len(symbolic.parameters)
        )
        gradients = tangent.restrict_gradient
    else:
        full_coordinates = coordinates
        vector_field = symbolic.dynamics + (sp.S.Zero,) * len(
            symbolic.parameters
        )

        def gradients(expression: sp.Expr) -> sp.Matrix:
            return _gradient(expression, coordinates)

    generators = symbolic.outputs
    current_generators = symbolic.outputs
    rows = [gradients(expression) for expression in current_generators]
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
            outcome = _rank.result(
                AnalysisStatus.IDENTIFIABLE,
                matrix,
                generators,
                required_rank,
                matrix_rank=rank,
            )
            return (
                _add_dae_condition(outcome, tangent.condition)
                if tangent
                else outcome
            )
        if previous_rank == rank:
            outcome = _rank.result(
                AnalysisStatus.NOT_IDENTIFIABLE,
                matrix,
                generators,
                required_rank,
                matrix_rank=rank,
            )
            return (
                _add_dae_condition(outcome, tangent.condition)
                if tangent
                else outcome
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
            _lie_derivative(expression, full_coordinates, vector_field)
            for expression in current_generators
        )
        generators += current_generators
        rows = [gradients(expression) for expression in current_generators]
        matrix = sp.Matrix.vstack(matrix, *rows) if rows else matrix
        order += 1
