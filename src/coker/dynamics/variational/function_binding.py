"""Specialize heterogeneous system parameters for variational solvers."""

from __future__ import annotations

import numpy as np

from collections.abc import Sequence

from coker.algebra.dimensions import FunctionSpace, VectorSpace
from coker.algebra.function import function
from coker.algebra.ops import Noop
from coker.dynamics.controls import BoundedVariable, DenseTensorVariable
from coker.dynamics.model import DynamicalSystem


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


def _tensor_declarations(
    declaration: DenseTensorVariable,
) -> list[BoundedVariable]:
    return [
        BoundedVariable(
            f"{declaration.name}_{index}",
            float(declaration.lower_bound.reshape(-1)[index]),
            float(declaration.upper_bound.reshape(-1)[index]),
            guess=float(declaration.guess.reshape(-1)[index]),
        )
        for index in range(declaration.size)
    ]


def specialize_system_parameters(
    system: DynamicalSystem, declarations: Sequence[object]
) -> tuple[DynamicalSystem, list[object]]:
    """Bind function-valued parameters and return a numeric solver system."""
    space = system.parameters
    if not isinstance(space, tuple):
        return system, list(declarations)
    if len(space) != len(declarations):
        raise ValueError("parameter specialization has the wrong arity")

    solver_declarations: list[object] = []
    offsets: list[tuple[int, int]] = []
    width = 0
    for index, (target, declaration) in enumerate(zip(space, declarations)):
        if isinstance(target, FunctionSpace):
            size = declaration.size
            offsets.append((width, width + size))
            solver_declarations.extend(_theta_declarations(index, declaration))
            width += size
        elif isinstance(target, VectorSpace):
            if not isinstance(declaration, DenseTensorVariable):
                raise TypeError(
                    "VectorSpace parameters require DenseTensorVariable"
                )
            if declaration.shape != (
                (target.dimension,)
                if isinstance(target.dimension, int)
                else target.dimension
            ):
                raise ValueError(
                    "DenseTensorVariable shape does not match VectorSpace"
                )
            offsets.append((width, width + target.size))
            solver_declarations.extend(_tensor_declarations(declaration))
            width += target.size
        else:
            offsets.append((width, width + 1))
            solver_declarations.append(declaration)
            width += 1
    parameter_blocks = {
        declaration.name: (start, end, declaration.shape)
        for target, declaration, (start, end) in zip(
            space, declarations, offsets
        )
        if isinstance(target, VectorSpace)
    }

    numeric_parameters = VectorSpace("p", width)

    def bound_function(declaration, theta):
        def evaluate(x):
            return declaration._evaluate(theta, x)

        return evaluate

    def bound_arguments(parameters):
        values = []
        for target, declaration, offset in zip(space, declarations, offsets):
            start, end = offset
            if isinstance(target, FunctionSpace):
                theta = parameters[start:end]
                values.append(bound_function(declaration, theta))
            elif isinstance(target, VectorSpace):
                values.append(
                    np.reshape(parameters[start:end], declaration.shape)
                )
            else:
                values.append(parameters[start])
        return values

    def bind_initial_conditions(original):
        if original is Noop():
            return Noop()
        spaces = original.input_spaces()
        return function(
            [spaces[0], spaces[1], numeric_parameters],
            lambda z, u, p: original(z, u, *bound_arguments(p)),
            backend=system.backend(),
        )

    def bind_dynamics(original):
        if original is Noop():
            return Noop()
        spaces = original.input_spaces()
        return function(
            [*spaces[:4], numeric_parameters],
            lambda t, x, z, u, p: original(t, x, z, u, *bound_arguments(p)),
            backend=system.backend(),
        )

    def bind_outputs(original):
        spaces = original.input_spaces()
        return function(
            [*spaces[:4], numeric_parameters, spaces[-1]],
            lambda t, x, z, u, p, q: original(
                t, x, z, u, *bound_arguments(p), q
            ),
            backend=system.backend(),
        )

    return (
        DynamicalSystem(
            system.inputs,
            numeric_parameters,
            bind_initial_conditions(system.x0),
            bind_dynamics(system.dxdt),
            bind_dynamics(system.g),
            bind_dynamics(system.dqdt),
            bind_outputs(system.y),
            solver_parameters=system.solver_parameters,
            parameter_blocks=parameter_blocks,
        ),
        solver_declarations,
    )
