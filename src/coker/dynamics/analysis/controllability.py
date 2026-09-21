"""Generic local accessibility analysis for control-affine systems."""

from __future__ import annotations

import sympy as sp

from coker.backends.sympy.analysis import (
    compute_lie_bracket,
    extract_control_affine_fields,
)

from coker.algebra.ops import Noop
from coker.dynamics.model import DynamicalSystem

from . import model as _rank
from .dae import SymbolicDAESystem, geometry, lower_dae_system
from .model import AnalysisStatus, ControllabilityResult
from .symbolic import UnsupportedSystemError, lower_system


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


def _dae_bracket(
    left,
    left_algebraic,
    right,
    right_algebraic,
    symbolic: SymbolicDAESystem,
) -> sp.ImmutableMatrix:
    """Bracket fields using intrinsic constraint-manifold derivatives."""
    bracket = compute_lie_bracket(
        sp.ImmutableMatrix.vstack(left, left_algebraic),
        sp.ImmutableMatrix.vstack(right, right_algebraic),
        (*symbolic.state, *symbolic.algebraic),
    )
    return sp.ImmutableMatrix(bracket[: left.rows, :])


def analyse_controllability(
    system, *, max_order: int | None = None
) -> ControllabilityResult:
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
        return _rank.create_inconclusive(
            ControllabilityResult,
            str(error) or "The system cannot be represented symbolically.",
            matrix=_rank.empty_matrix(0, 0, immutable=True),
        )

    state_dimension = len(symbolic.state)
    if not isinstance(max_order, int) or isinstance(max_order, bool):
        if max_order is not None:
            return _rank.create_inconclusive(
                ControllabilityResult,
                "max_order must be a non-negative integer or None.",
                required_rank=state_dimension,
                matrix=_rank.empty_matrix(state_dimension, 0, immutable=True),
            )
    elif max_order < 0:
        return _rank.create_inconclusive(
            ControllabilityResult,
            "max_order must be a non-negative integer or None.",
            required_rank=state_dimension,
            matrix=_rank.empty_matrix(state_dimension, 0, immutable=True),
        )

    affine_fields = extract_control_affine_fields(
        symbolic.dynamics, symbolic.controls
    )
    if affine_fields is None:
        return _rank.create_inconclusive(
            ControllabilityResult,
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
            outcome = _rank.create_result(
                ControllabilityResult,
                AnalysisStatus.TRUE,
                matrix,
                fields,
                state_dimension,
                matrix_rank=rank,
            )
            return (
                _rank.add_condition(outcome, tangent.condition)
                if tangent
                else outcome
            )
        if not frontier:
            outcome = _rank.create_result(
                ControllabilityResult,
                AnalysisStatus.FALSE,
                matrix,
                fields,
                state_dimension,
                matrix_rank=rank,
            )
            return (
                _rank.add_condition(outcome, tangent.condition)
                if tangent
                else outcome
            )
        if max_order is not None and order >= max_order:
            return _rank.create_inconclusive(
                ControllabilityResult,
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
                    else compute_lie_bracket(field, base_field, symbolic.state)
                )
                fields, rank, added = _append_if_independent(
                    fields, rank, bracket, state_dimension
                )
                if added:
                    next_frontier = (*next_frontier, bracket)
                    if tangent:
                        algebraic_fields[bracket] = tangent.lift(bracket)
        if not next_frontier:
            outcome = _rank.create_result(
                ControllabilityResult,
                AnalysisStatus.FALSE,
                _matrix_from_fields(fields, state_dimension),
                fields,
                state_dimension,
                matrix_rank=rank,
            )
            return (
                _rank.add_condition(outcome, tangent.condition)
                if tangent
                else outcome
            )
        frontier = next_frontier
        order += 1
