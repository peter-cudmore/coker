"""Specialize heterogeneous system parameters for variational solvers."""

from __future__ import annotations

from collections.abc import Sequence

from coker.algebra.dimensions import FunctionSpace, VectorSpace
from coker.algebra.function import function
from coker.algebra.ops import Noop
from coker.dynamics.controls import BoundedVariable
from coker.dynamics.model import DynamicalSystem
from coker.dynamics.parameters import ParameterSpace


def _theta_declarations(index: int, declaration) -> list[BoundedVariable]:
    _, initial, lower, upper = declaration.decision_declarations()
    return [
        BoundedVariable(
            f"p_{index}_theta_{offset}",
            float(lower[offset]),
            float(upper[offset]),
            guess=float(initial[offset]),
        )
        for offset in range(declaration.size)
    ]


def specialize_system_parameters(
    system: DynamicalSystem, declarations: Sequence[object]
) -> tuple[DynamicalSystem, list[object]]:
    """Bind function-valued parameters and return a numeric solver system."""
    space = system.parameter_space
    if space is None:
        return system, list(declarations)
    if len(space) != len(declarations):
        raise ValueError("parameter specialization has the wrong arity")

    solver_declarations: list[object] = []
    offsets: list[tuple[int, int] | None] = []
    width = 0
    for index, (target, declaration) in enumerate(zip(space, declarations)):
        if isinstance(target, FunctionSpace):
            declaration.validate_target(target)
            size = declaration.size
            offsets.append((width, width + size))
            solver_declarations.extend(_theta_declarations(index, declaration))
            width += size
        else:
            offsets.append((width, width + 1))
            solver_declarations.append(declaration)
            width += 1

    numeric_parameters = VectorSpace("p", width)

    def bound_function(declaration, theta):
        def evaluate(x):
            return declaration._evaluate(theta, x)

        return evaluate

    def bound_arguments(parameters):
        values = []
        for target, declaration, offset in zip(space, declarations, offsets):
            assert offset is not None
            start, end = offset
            if isinstance(target, FunctionSpace):
                theta = parameters[start:end]
                values.append(bound_function(declaration, theta))
            else:
                values.append(parameters[start])
        return values

    def component(original, prefix):
        if original is Noop():
            return Noop()
        spaces = original.input_spaces()
        if prefix == "x0":
            arguments = [spaces[0], spaces[1], numeric_parameters]
            return function(
                arguments,
                lambda z, u, p: original(z, u, *bound_arguments(p)),
                backend=system.backend(),
            )
        if prefix == "output":
            arguments = [*spaces[:4], numeric_parameters, spaces[-1]]
            return function(
                arguments,
                lambda t, x, z, u, p, q: original(
                    t, x, z, u, *bound_arguments(p), q
                ),
                backend=system.backend(),
            )
        arguments = [*spaces[:4], numeric_parameters]
        return function(
            arguments,
            lambda t, x, z, u, p: original(t, x, z, u, *bound_arguments(p)),
            backend=system.backend(),
        )

    return (
        DynamicalSystem(
            system.inputs,
            numeric_parameters,
            component(system.x0, "x0"),
            component(system.dxdt, "dynamics"),
            component(system.g, "constraints"),
            component(system.dqdt, "quadratures"),
            component(system.y, "output"),
            solver_parameters=system.solver_parameters,
        ),
        solver_declarations,
    )
