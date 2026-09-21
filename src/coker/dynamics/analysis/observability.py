"""Generic local observability analysis for input-affine systems."""

from __future__ import annotations


import sympy as sp
from coker.backends.sympy.analysis import (
    compute_gradient,
    compute_lie_derivative,
    extract_control_affine_fields,
    extract_field_components,
)

from coker.algebra.ops import Noop
from coker.dynamics.model import DynamicalSystem

from . import model as _rank
from .dae import SymbolicDAESystem, geometry, lower_dae_system
from .model import ObservabilityResult
from .symbolic import SymbolicSystem, UnsupportedSystemError, lower_system


def analyse_observability(
    system: object, *, max_order: int | None = None
) -> ObservabilityResult:
    """Establish generic local observability using the Lie-rank test.

    Inputs are treated as known, arbitrary signals.  Consequently, controlled
    systems must be affine in their controls; the analysis closes the output
    codistribution under both the drift and each control vector field.
    """
    try:
        if isinstance(system, SymbolicDAESystem | SymbolicSystem):
            symbolic = system
        elif isinstance(system, DynamicalSystem):
            symbolic = (
                lower_dae_system(system)
                if not isinstance(system.g, Noop)
                else lower_system(system)
            )
        else:
            return _rank.create_inconclusive(
                ObservabilityResult, "system is unsupported for analysis"
            )
        tangent = (
            geometry(symbolic)
            if isinstance(symbolic, SymbolicDAESystem)
            else None
        )
    except UnsupportedSystemError as error:
        return _rank.create_inconclusive(
            ObservabilityResult,
            str(error) or "system is unsupported for analysis",
        )

    state_dimension = len(symbolic.state)
    if max_order is not None and (
        not isinstance(max_order, int)
        or isinstance(max_order, bool)
        or max_order < 0
    ):
        return _rank.create_inconclusive(
            ObservabilityResult,
            "max_order must be a non-negative integer or None",
            required_rank=state_dimension,
        )

    affine_fields = extract_control_affine_fields(
        symbolic.dynamics, symbolic.controls
    )
    if affine_fields is None:
        return _rank.create_inconclusive(
            ObservabilityResult,
            "Dynamics are not affine in the controls.",
            required_rank=state_dimension,
        )
    drift, controls = affine_fields

    if isinstance(symbolic, SymbolicDAESystem):
        assert tangent is not None
        coordinates = symbolic.state
        full_coordinates = symbolic.state + symbolic.algebraic
        vector_fields = tuple(
            extract_field_components(field)
            + extract_field_components(tangent.lift(field))
            for field in (drift, *controls)
        )
        gradients = tangent.restrict_gradient
    else:
        coordinates = symbolic.state
        full_coordinates = coordinates
        vector_fields = tuple(
            extract_field_components(field) for field in (drift, *controls)
        )

        def gradients(expression: sp.Expr) -> sp.Matrix:
            return compute_gradient(expression, coordinates)

    outcome = _rank.compute_rank_closure(
        ObservabilityResult,
        symbolic.outputs,
        state_dimension,
        gradients,
        lambda expressions: tuple(
            compute_lie_derivative(vector_field, expression, full_coordinates)
            for expression in expressions
            for vector_field in vector_fields
        ),
        max_order,
        f"maximum Lie-derivative order {max_order} reached before "
        "the observability rank stabilized",
    )
    return (
        _rank.add_condition(outcome, tangent.condition) if tangent else outcome
    )
