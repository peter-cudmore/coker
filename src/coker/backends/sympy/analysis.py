"""Reusable symbolic rank and constraint-manifold operations."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ConstraintGeometry:
    """Constraint-manifold tangent operations."""

    state: tuple[sp.Symbol, ...]
    algebraic: tuple[sp.Symbol, ...]
    parameters: tuple[sp.Symbol, ...]
    constraint_jacobian: sp.Matrix
    state_jacobian: sp.Matrix
    parameter_jacobian: sp.Matrix
    condition: sp.Expr

    def lift(self, field: sp.MatrixBase) -> sp.Matrix:
        """Solve for the algebraic component tangent to the constraint."""
        return -self.constraint_jacobian.LUsolve(self.state_jacobian * field)

    def restrict_gradient(self, expression: sp.Expr) -> sp.Matrix:
        """Differentiate in independent state and parameter coordinates."""
        gradient_x = sp.Matrix(
            1,
            len(self.state),
            [sp.diff(expression, value) for value in self.state],
        )
        gradient_z = sp.Matrix(
            1,
            len(self.algebraic),
            [sp.diff(expression, value) for value in self.algebraic],
        )
        gradient_p = sp.Matrix(
            1,
            len(self.parameters),
            [sp.diff(expression, value) for value in self.parameters],
        )
        adjoint = self.constraint_jacobian.T.LUsolve(gradient_z.T).T
        return (gradient_x - adjoint * self.state_jacobian).row_join(
            gradient_p - adjoint * self.parameter_jacobian
        )


def constraint_geometry(
    state: tuple[sp.Symbol, ...],
    algebraic: tuple[sp.Symbol, ...],
    parameters: tuple[sp.Symbol, ...],
    constraints: tuple[sp.Expr, ...],
) -> ConstraintGeometry:
    """Construct index-one tangent operations without forming an inverse."""
    constraint_vector = sp.Matrix(constraints)
    jacobian = constraint_vector.jacobian(sp.Matrix(algebraic))
    determinant = sp.factor(jacobian.det())
    if sp.simplify(determinant) == 0:
        raise ValueError("DAE algebraic Jacobian is generically singular")
    return ConstraintGeometry(
        state,
        algebraic,
        parameters,
        jacobian,
        constraint_vector.jacobian(sp.Matrix(state)),
        (
            constraint_vector.jacobian(sp.Matrix(parameters))
            if parameters
            else sp.zeros(len(algebraic), 0)
        ),
        determinant,
    )


def empty_matrix(
    rows: int, columns: int, *, immutable: bool = False
) -> sp.MatrixBase:
    """Create an empty matrix while preserving the requested matrix kind."""
    return (
        sp.ImmutableMatrix.zeros(rows, columns)
        if immutable
        else sp.zeros(rows, columns)
    )


def generic_rank_witnesses(
    matrix: sp.MatrixBase, rank: int
) -> tuple[sp.Expr, ...]:
    """Return a nonzero maximal minor that witnesses ``matrix``'s rank."""
    if rank == 0:
        return ()
    _reduced, columns = matrix.rref()
    selected_columns = matrix[:, list(columns)]
    _reduced, rows = selected_columns.T.rref()
    witness = sp.factor(matrix.extract(list(rows), list(columns)).det())
    if sp.simplify(witness) == 0:
        raise RuntimeError("A rank-positive matrix must have a nonzero minor")
    return (witness,)
