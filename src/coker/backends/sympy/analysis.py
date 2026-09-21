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
        state_gradient = sp.Matrix(
            1,
            len(self.state),
            [sp.diff(expression, coordinate) for coordinate in self.state],
        )
        algebraic_gradient = sp.Matrix(
            1,
            len(self.algebraic),
            [sp.diff(expression, coordinate) for coordinate in self.algebraic],
        )
        parameter_gradient = sp.Matrix(
            1,
            len(self.parameters),
            [
                sp.diff(expression, coordinate)
                for coordinate in self.parameters
            ],
        )
        constrained_gradient = self.constraint_jacobian.T.LUsolve(
            algebraic_gradient.T
        ).T
        return sp.Matrix.hstack(
            state_gradient - constrained_gradient * self.state_jacobian,
            parameter_gradient
            - constrained_gradient * self.parameter_jacobian,
        )


def constraint_geometry(
    state: tuple[sp.Symbol, ...],
    algebraic: tuple[sp.Symbol, ...],
    parameters: tuple[sp.Symbol, ...],
    constraints: tuple[sp.Expr, ...],
) -> ConstraintGeometry:
    """Construct index-one tangent operations without forming an inverse."""
    constraint_vector = sp.Matrix(constraints)
    algebraic_vector = sp.Matrix(algebraic)
    constraint_jacobian = constraint_vector.jacobian(algebraic_vector)
    determinant = sp.factor(constraint_jacobian.det())
    if determinant == 0:
        raise ValueError("DAE constraint Jacobian is generically singular")
    parameter_jacobian = (
        constraint_vector.jacobian(sp.Matrix(parameters))
        if parameters
        else sp.zeros(len(constraints), 0)
    )
    return ConstraintGeometry(
        state=state,
        algebraic=algebraic,
        parameters=parameters,
        constraint_jacobian=constraint_jacobian,
        state_jacobian=constraint_vector.jacobian(sp.Matrix(state)),
        parameter_jacobian=parameter_jacobian,
        condition=determinant,
    )


def empty_matrix(
    rows: int, columns: int, *, immutable: bool = False
) -> sp.MatrixBase:
    """Create an empty matrix while preserving the requested matrix kind."""
    matrix_type = sp.ImmutableMatrix if immutable else sp.Matrix
    return matrix_type(rows, columns, [])


def generic_rank_witnesses(
    matrix: sp.MatrixBase, rank: int
) -> tuple[sp.Expr, ...]:
    """Return a nonzero maximal minor that witnesses ``matrix``'s rank."""
    if rank == 0:
        return ()
    rows, columns = matrix.shape
    for row_indices in sp.utilities.iterables.combinations(range(rows), rank):
        for column_indices in sp.utilities.iterables.combinations(
            range(columns), rank
        ):
            witness = sp.factor(
                matrix.extract(row_indices, column_indices).det()
            )
            if witness != 0:
                return (witness,)
    return ()


def compute_gradient(
    expression: sp.Expr, coordinates: tuple[sp.Symbol, ...]
) -> sp.Matrix:
    """Differentiate a scalar expression in the supplied coordinates."""
    return sp.Matrix(
        1,
        len(coordinates),
        [sp.diff(expression, coordinate) for coordinate in coordinates],
    )


def compute_lie_derivative(
    vector_field: tuple[sp.Expr, ...],
    expression: sp.Expr,
    coordinates: tuple[sp.Symbol, ...],
) -> sp.Expr:
    """Differentiate an expression along a symbolic vector field."""
    return sum(
        (
            sp.diff(expression, coordinate) * component
            for coordinate, component in zip(coordinates, vector_field)
        ),
        sp.S.Zero,
    )


def extract_field_components(field: sp.MatrixBase) -> tuple[sp.Expr, ...]:
    """Extract a column vector as immutable SymPy expressions."""
    return tuple(sp.sympify(field[index, 0]) for index in range(field.rows))


def extract_control_affine_fields(
    dynamics: tuple[sp.Expr, ...], controls: tuple[sp.Symbol, ...]
) -> tuple[sp.ImmutableMatrix, tuple[sp.ImmutableMatrix, ...]] | None:
    """Split dynamics into a drift and control fields when input-affine."""
    zero_controls = {control: sp.S.Zero for control in controls}
    try:
        control_fields = tuple(
            create_field(
                tuple(sp.diff(component, control) for component in dynamics)
            )
            for control in controls
        )
        if any(
            sp.simplify(sp.diff(component, control)) != 0
            for field in control_fields
            for component in field
            for control in controls
        ):
            return None
        drift = create_field(
            tuple(component.xreplace(zero_controls) for component in dynamics)
        )
        if any(
            sp.simplify(
                component
                - drift[index]
                - sum(
                    field[index] * control
                    for field, control in zip(control_fields, controls)
                )
            )
            != 0
            for index, component in enumerate(dynamics)
        ):
            return None
    except (NotImplementedError, TypeError, ValueError):
        return None
    return simplify_field(drift), tuple(
        simplify_field(field) for field in control_fields
    )


def compute_lie_bracket(
    left: sp.MatrixBase,
    right: sp.MatrixBase,
    coordinates: tuple[sp.Symbol, ...],
) -> sp.ImmutableMatrix:
    """Calculate the Lie bracket ``[left, right]``."""
    coordinate_vector = sp.ImmutableMatrix(coordinates)
    return simplify_field(
        right.jacobian(coordinate_vector) * left
        - left.jacobian(coordinate_vector) * right
    )


def create_field(values: tuple[sp.Expr, ...]) -> sp.ImmutableMatrix:
    """Represent a vector field as an immutable column matrix."""
    return sp.ImmutableMatrix(len(values), 1, values)


def simplify_field(field: sp.MatrixBase) -> sp.ImmutableMatrix:
    """Simplify every component of a vector field."""
    return sp.ImmutableMatrix(
        field.rows, field.cols, [sp.simplify(entry) for entry in field]
    )
