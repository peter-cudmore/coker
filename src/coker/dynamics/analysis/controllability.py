"""Generic local accessibility analysis for control-affine systems."""

from __future__ import annotations

from dataclasses import replace

import sympy as sp

from coker.algebra.ops import Noop
from coker.dynamics.model import DynamicalSystem

from . import _rank
from .dae import SymbolicDAESystem, geometry, lower_dae_system
from .model import AnalysisResult, AnalysisStatus
from .symbolic import UnsupportedSystemError, lower_system


def _as_field(values) -> sp.ImmutableMatrix:
    return sp.ImmutableMatrix(len(values), 1, values)


def _simplify_field(field: sp.MatrixBase) -> sp.ImmutableMatrix:
    return sp.ImmutableMatrix(
        field.rows, field.cols, [sp.simplify(entry) for entry in field]
    )


def _is_zero(expression: sp.Expr) -> bool:
    return sp.simplify(expression) == 0


def _matrix_from_fields(fields, state_dimension: int) -> sp.ImmutableMatrix:
    if not fields:
        return _rank.empty_matrix(state_dimension, 0, immutable=True)
    return sp.ImmutableMatrix.hstack(*fields)


def _append_if_independent(fields, rank: int, candidate, state_dimension: int):
    candidate_rank = _matrix_from_fields(
        (*fields, candidate), state_dimension
    ).rank()
    if candidate_rank <= rank:
        return fields, rank, False
    return (*fields, candidate), candidate_rank, True


def _extract_control_affine_fields(system):
    controls = system.controls
    dynamics = system.dynamics
    zero_controls = {control: sp.S.Zero for control in controls}
    try:
        control_fields = tuple(
            _as_field(
                tuple(sp.diff(component, control) for component in dynamics)
            )
            for control in controls
        )
        for field in control_fields:
            if any(
                not _is_zero(sp.diff(component, control))
                for component in field
                for control in controls
            ):
                return None
        drift = _as_field(
            tuple(component.subs(zero_controls) for component in dynamics)
        )
        for index, component in enumerate(dynamics):
            reconstructed = drift[index] + sum(
                field[index] * control
                for field, control in zip(control_fields, controls)
            )
            if not _is_zero(component - reconstructed):
                return None
    except (NotImplementedError, TypeError, ValueError):
        return None
    return _simplify_field(drift), tuple(
        _simplify_field(field) for field in control_fields
    )


def _lie_bracket(left, right, state) -> sp.ImmutableMatrix:
    state_vector = sp.ImmutableMatrix(state)
    return _simplify_field(
        right.jacobian(state_vector) * left
        - left.jacobian(state_vector) * right
    )


def _dae_bracket(
    left,
    left_algebraic,
    right,
    right_algebraic,
    symbolic: SymbolicDAESystem,
) -> sp.ImmutableMatrix:
    """Bracket fields using intrinsic constraint-manifold derivatives."""
    coordinates = sp.ImmutableMatrix((*symbolic.state, *symbolic.algebraic))
    left_full = sp.ImmutableMatrix.vstack(left, left_algebraic)
    right_full = sp.ImmutableMatrix.vstack(right, right_algebraic)
    return _simplify_field(
        right.jacobian(coordinates) * left_full
        - left.jacobian(coordinates) * right_full
    )


def _add_dae_condition(
    result: AnalysisResult, condition: sp.Expr
) -> AnalysisResult:
    return replace(
        result, generic_conditions=(condition, *result.generic_conditions)
    )


def analyse_controllability(
    system, *, max_order: int | None = None
) -> AnalysisResult:
    """Establish generic local accessibility through Lie-algebra rank."""
    try:
        symbolic = (
            lower_dae_system(system)
            if isinstance(system, DynamicalSystem)
            and not isinstance(system.g, Noop)
            else lower_system(system)
        )
        tangent = (
            geometry(symbolic)
            if isinstance(symbolic, SymbolicDAESystem)
            else None
        )
    except UnsupportedSystemError as error:
        return _rank.inconclusive(
            str(error) or "The system cannot be represented symbolically.",
            matrix=_rank.empty_matrix(0, 0, immutable=True),
        )

    state_dimension = len(symbolic.state)
    if not isinstance(max_order, int) or isinstance(max_order, bool):
        if max_order is not None:
            return _rank.inconclusive(
                "max_order must be a non-negative integer or None.",
                required_rank=state_dimension,
                matrix=_rank.empty_matrix(state_dimension, 0, immutable=True),
            )
    elif max_order < 0:
        return _rank.inconclusive(
            "max_order must be a non-negative integer or None.",
            required_rank=state_dimension,
            matrix=_rank.empty_matrix(state_dimension, 0, immutable=True),
        )

    affine_fields = _extract_control_affine_fields(symbolic)
    if affine_fields is None:
        return _rank.inconclusive(
            "Dynamics are not affine in the controls.",
            required_rank=state_dimension,
            matrix=_rank.empty_matrix(state_dimension, 0, immutable=True),
        )
    drift, controls = affine_fields
    base_fields = (drift, *controls)
    algebraic_fields = (
        {field: tangent.lift(field) for field in base_fields}
        if tangent
        else {}
    )
    fields = ()
    rank = 0
    frontier = ()
    for field in controls:
        fields, rank, added = _append_if_independent(
            fields, rank, field, state_dimension
        )
        if added:
            frontier = (*frontier, field)

    order = 0
    while True:
        matrix = _matrix_from_fields(fields, state_dimension)
        if rank == state_dimension:
            outcome = _rank.result(
                AnalysisStatus.ACCESSIBLE,
                matrix,
                fields,
                state_dimension,
                matrix_rank=rank,
            )
            return (
                _add_dae_condition(outcome, tangent.condition)
                if tangent
                else outcome
            )
        if not frontier:
            outcome = _rank.result(
                AnalysisStatus.NOT_ACCESSIBLE,
                matrix,
                fields,
                state_dimension,
                matrix_rank=rank,
            )
            return (
                _add_dae_condition(outcome, tangent.condition)
                if tangent
                else outcome
            )
        if max_order is not None and order >= max_order:
            return _rank.inconclusive(
                "Lie-bracket expansion reached max_order before closure.",
                required_rank=state_dimension,
                matrix=matrix,
                generators=fields,
            )
        next_frontier = ()
        for field in frontier:
            for base_field in base_fields:
                bracket = (
                    _dae_bracket(
                        field,
                        algebraic_fields[field],
                        base_field,
                        algebraic_fields[base_field],
                        symbolic,
                    )
                    if tangent
                    else _lie_bracket(field, base_field, symbolic.state)
                )
                fields, rank, added = _append_if_independent(
                    fields, rank, bracket, state_dimension
                )
                if added:
                    next_frontier = (*next_frontier, bracket)
                    if tangent:
                        algebraic_fields[bracket] = tangent.lift(bracket)
        if not next_frontier:
            outcome = _rank.result(
                AnalysisStatus.NOT_ACCESSIBLE,
                _matrix_from_fields(fields, state_dimension),
                fields,
                state_dimension,
                matrix_rank=rank,
            )
            return (
                _add_dae_condition(outcome, tangent.condition)
                if tangent
                else outcome
            )
        frontier = next_frontier
        order += 1
