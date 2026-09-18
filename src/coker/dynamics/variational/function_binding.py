"""Specialize heterogeneous system parameters for variational solvers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from coker.algebra.dimensions import FunctionSpace, VectorSpace
from coker.algebra.function import function
from coker.algebra.ops import Noop
from coker.dynamics.function_parameters import (
    FunctionParameter,
    trace_function_parameter,
)
from coker.dynamics.variables import (
    BoundVector,
    BoundedVariable,
    DenseTensorVariable,
    UnboundedVariable,
)
from coker.dynamics.model import DynamicalSystem


@dataclass(frozen=True)
class ParameterValueLayout:
    targets: tuple[object, ...]
    declarations: tuple[object, ...]
    offsets: tuple[tuple[int, int], ...]

    def reconstruct(self, values: np.ndarray) -> Mapping[str, object]:
        result = {}
        for target, declaration, (start, end) in zip(
            self.targets, self.declarations, self.offsets
        ):
            basis = np.asarray(values[start:end], dtype=float)
            if isinstance(target, FunctionSpace):
                result[declaration.name] = self._function(declaration, basis)
            elif isinstance(target, VectorSpace):
                result[declaration.name] = basis.reshape(declaration.shape)
            else:
                result[declaration.name] = float(basis[0])
        return result

    @staticmethod
    def _function(
        declaration: FunctionParameter, basis: np.ndarray
    ) -> Callable:
        return lambda argument: declaration.evaluate(basis, argument)


def _basis_declarations(
    index: int, target: FunctionSpace, declaration: FunctionParameter
) -> list[BoundedVariable | UnboundedVariable]:
    if not isinstance(declaration.name, str) or not declaration.name:
        raise ValueError("function parameter name must be a non-empty string")
    declaration.validate_target(target)
    values = declaration.decision_declarations()
    if len(values) == 2:
        basis, initial = values
        lower = upper = None
    elif len(values) == 4:
        basis, initial, lower, upper = values
    else:
        raise TypeError(
            "function parameter declarations must include zero or two bounds"
        )
    if not isinstance(basis, VectorSpace):
        raise TypeError("function parameter basis must be a VectorSpace")
    size = basis.size
    initial = np.asarray(initial, dtype=float)
    if (
        initial.ndim != 1
        or initial.size != size
        or not np.all(np.isfinite(initial))
    ):
        raise ValueError("function parameter initial values are invalid")
    if lower is None:
        return [
            UnboundedVariable(
                f"p_{index}_theta_{offset}", guess=float(initial[offset])
            )
            for offset in range(size)
        ]
    lower, upper = (np.asarray(value, dtype=float) for value in (lower, upper))
    if (
        lower.ndim != 1
        or upper.ndim != 1
        or lower.size != size
        or upper.size != size
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
        or np.any(initial < lower)
        or np.any(initial > upper)
    ):
        raise ValueError("function parameter bounds are invalid")
    return [
        BoundedVariable(
            f"p_{index}_theta_{offset}",
            float(lower[offset]),
            float(upper[offset]),
            guess=float(initial[offset]),
        )
        for offset in range(size)
    ]


def _tensor_declarations(
    declaration: BoundVector | DenseTensorVariable,
) -> list[BoundedVariable | UnboundedVariable]:
    if isinstance(declaration, DenseTensorVariable):
        return [
            UnboundedVariable(
                f"{declaration.name}_{index}",
                guess=float(declaration.guess.reshape(-1)[index]),
            )
            for index in range(declaration.size)
        ]
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
) -> tuple[DynamicalSystem, list[object], ParameterValueLayout | None]:
    """Bind function-valued parameters and return a numeric solver system."""
    space = system.parameters
    if not isinstance(space, tuple):
        return system, list(declarations), None
    if len(space) != len(declarations):
        raise ValueError("parameter specialization has the wrong arity")

    solver_declarations: list[object] = []
    offsets: list[tuple[int, int]] = []
    width = 0
    for index, (target, declaration) in enumerate(zip(space, declarations)):
        if isinstance(target, FunctionSpace):
            if not isinstance(declaration, FunctionParameter):
                raise TypeError(
                    f"Parameter {index} must implement FunctionParameter, got "
                    f"{type(declaration).__name__}"
                )
            basis_declarations = _basis_declarations(
                index, target, declaration
            )
            size = len(basis_declarations)
            offsets.append((width, width + size))
            solver_declarations.extend(basis_declarations)
            width += size
        elif isinstance(target, VectorSpace):
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

    def reconstruct_function_parameter(declaration, basis):
        def evaluate(x):
            return trace_function_parameter(declaration, basis, x)

        return evaluate

    def reconstruct_parameters(parameters):
        values = []
        for target, declaration, offset in zip(space, declarations, offsets):
            start, end = offset
            if isinstance(target, FunctionSpace):
                basis = parameters[start:end]
                values.append(
                    reconstruct_function_parameter(declaration, basis)
                )
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
            lambda z, u, p: original(z, u, *reconstruct_parameters(p)),
            backend=system.backend(),
        )

    def bind_dynamics(original):
        if original is Noop():
            return Noop()
        spaces = original.input_spaces()
        return function(
            [*spaces[:4], numeric_parameters],
            lambda t, x, z, u, p: original(
                t, x, z, u, *reconstruct_parameters(p)
            ),
            backend=system.backend(),
        )

    def bind_outputs(original):
        spaces = original.input_spaces()
        return function(
            [*spaces[:4], numeric_parameters, spaces[-1]],
            lambda t, x, z, u, p, q: original(
                t, x, z, u, *reconstruct_parameters(p), q
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
        ParameterValueLayout(
            tuple(space), tuple(declarations), tuple(offsets)
        ),
    )
