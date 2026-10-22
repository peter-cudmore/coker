"""Generic local structural identifiability analysis."""

from __future__ import annotations

from dataclasses import replace
from numbers import Real
from typing import Sequence

import sympy as sp
from coker.backends.sympy.analysis import (
    compute_gradient,
    compute_lie_derivative,
)

from coker.parameters import BoundedVariable, UnboundedVariable

from . import model as _rank
from .dae import (
    SymbolicDAESystem,
    geometry,
    normalize_symbolic_system,
)
from .model import IdentifiabilityResult
from .symbolic import SymbolicSystem, UnsupportedSystemError


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
        elif isinstance(declaration, Real) and not isinstance(declaration, bool):
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
            expression.xreplace(replacements) for expression in symbolic.dynamics
        ),
        "outputs": tuple(
            expression.xreplace(replacements) for expression in symbolic.outputs
        ),
    }
    if isinstance(symbolic, SymbolicDAESystem):
        fields["constraints"] = tuple(
            expression.xreplace(replacements) for expression in symbolic.constraints
        )
    return replace(symbolic, **fields)


def analyse_identifiability(
    system: object,
    *,
    parameters: Sequence[BoundedVariable | UnboundedVariable | Real] | None = None,
    max_order: int | None = None,
) -> IdentifiabilityResult:
    """Analyse generic local identifiability by augmented observability.

    Fitted parameters are appended to differential states with zero dynamics.
    Full rank tests joint local state/parameter observability, not global
    parameter uniqueness or a parameter-only test with known initial states.

    Args:
        system: A supported autonomous ``DynamicalSystem``, ``SymbolicSystem``
            or semi-explicit index-one ``SymbolicDAESystem`` without controls.
            General residual models and function-valued parameters are not
            supported.
        parameters: Optional roles in flattened scalar parameter order.
            ``BoundedVariable`` and ``UnboundedVariable`` mark fitted unknowns;
            real numeric constants are substituted as known values. Bounds do
            not alter the symbolic rank test. ``None`` fits every parameter.
        max_order: Non-negative Lie-derivative expansion limit, or ``None`` to
            continue until full rank or closure. Zero tests output gradients.

    Returns:
        IdentifiabilityResult: Rank evidence with differential-state then
        fitted-parameter columns. See :class:`AnalysisStatus` for verdicts and
        :class:`AnalysisResult` for generic-rank and DAE regularity conditions.
        Unsupported models, incompatible parameter roles, invalid options or
        premature limits give ``INCONCLUSIVE`` with ``reason``.

    Examples:
        >>> import sympy as sp
        >>> from coker.dynamics import (
        ...     SymbolicSystem, AnalysisStatus, analyse_identifiability,
        ... )
        >>> x, p = sp.symbols("x p")
        >>> system = SymbolicSystem((x,), (p,), (), (p * x,), (x,))
        >>> result = analyse_identifiability(system)
        >>> result.status is AnalysisStatus.TRUE
        True
        >>> result.rank, result.required_rank
        (2, 2)
        >>> result.reason is None
        True
    """
    try:
        symbolic = normalize_symbolic_system(system)
        symbolic = _apply_parameter_roles(symbolic, parameters)
        tangent = (
            geometry(symbolic) if isinstance(symbolic, SymbolicDAESystem) else None
        )
    except UnsupportedSystemError as error:
        return _rank.create_inconclusive(
            IdentifiabilityResult,
            str(error) or "system is unsupported for analysis",
        )

    coordinates = symbolic.state + symbolic.parameters
    required_rank = len(coordinates)
    if symbolic.controls:
        return _rank.create_inconclusive(
            IdentifiabilityResult,
            "identifiability analysis does not support controlled systems",
            required_rank=required_rank,
        )
    if max_order is not None and (
        not isinstance(max_order, int) or isinstance(max_order, bool) or max_order < 0
    ):
        return _rank.create_inconclusive(
            IdentifiabilityResult,
            "max_order must be a non-negative integer or None",
            required_rank=required_rank,
        )

    if tangent:
        algebraic_velocity = tangent.lift(sp.Matrix(symbolic.dynamics))
        full_coordinates = symbolic.state + symbolic.algebraic + symbolic.parameters
        vector_field = (
            symbolic.dynamics
            + tuple(algebraic_velocity)
            + (sp.S.Zero,) * len(symbolic.parameters)
        )
        gradients = tangent.restrict_gradient
    else:
        full_coordinates = coordinates
        vector_field = symbolic.dynamics + (sp.S.Zero,) * len(symbolic.parameters)

        def gradients(expression: sp.Expr) -> sp.Matrix:
            return compute_gradient(expression, coordinates)

    outcome = _rank.compute_rank_closure(
        IdentifiabilityResult,
        symbolic.outputs,
        required_rank,
        gradients,
        lambda expressions: tuple(
            compute_lie_derivative(vector_field, expression, full_coordinates)
            for expression in expressions
        ),
        max_order,
        f"maximum Lie-derivative order {max_order} reached before "
        "the augmented observability rank stabilized",
    )
    return _rank.add_condition(outcome, tangent.condition) if tangent else outcome
