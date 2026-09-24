"""Specialize heterogeneous system parameters for variational solvers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace
from coker.algebra.function import BoundCallable, function
from coker.algebra.ops import Noop
from coker.backends.backend import get_backend_by_name
from coker.parameters.function_parameters import FunctionParameter
from coker.dynamics.model import DynamicalSystem
from coker.parameters import (
    BoundVector,
    BoundedVariable,
    DenseTensorVariable,
    UnboundedVariable,
)


@dataclass(frozen=True)
class ParameterValueLayout:
    targets: tuple[object, ...]
    declarations: tuple[object, ...]
    offsets: tuple[tuple[int, int], ...]

    def reconstruct(
        self, values: object, backend: object | None = None
    ) -> Mapping[str, object]:
        result = {}
        for target, declaration, (start, end) in zip(
            self.targets, self.declarations, self.offsets
        ):
            basis = values[start:end]
            name = self._name(target, declaration)
            if isinstance(target, FunctionSpace):
                reconstruction_backend = (
                    get_backend_by_name("numpy", set_current=False)
                    if backend is None
                    else backend
                )
                result[name] = reconstruction_backend.fit_function_parameter(
                    declaration, target, basis
                )
            elif isinstance(target, VectorSpace):
                result[name] = self._numpy(basis, backend).reshape(
                    declaration.shape
                )
            else:
                result[name] = float(
                    self._numpy(basis, backend).reshape((-1,))[0]
                )
        return result

    @staticmethod
    def _name(target: object, declaration: object) -> str:
        if isinstance(
            declaration,
            (
                BoundedVariable,
                BoundVector,
                DenseTensorVariable,
                FunctionParameter,
                UnboundedVariable,
            ),
        ):
            return declaration.name
        assert isinstance(target, (Scalar, VectorSpace))
        return target.name

    @staticmethod
    def _numpy(values: object, backend: object | None) -> np.ndarray:
        if backend is not None:
            values = backend.to_numpy_array(values)
        return np.asarray(values, dtype=float)


def _function_concrete_declarations(
    declaration: FunctionParameter,
) -> tuple[
    BoundedVariable | UnboundedVariable | BoundVector | DenseTensorVariable,
    ...,
]:
    if not isinstance(declaration.name, str) or not declaration.name:
        raise ValueError("function parameter name must be a non-empty string")
    values = declaration.list_concrete_parameters()
    if not isinstance(values, tuple):
        raise TypeError(
            "function parameter concrete declarations must be a tuple"
        )
    return values


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


def _flatten_declaration(
    declaration: (
        BoundedVariable | UnboundedVariable | BoundVector | DenseTensorVariable
    ),
) -> list[BoundedVariable | UnboundedVariable]:
    if not isinstance(declaration.name, str) or not declaration.name:
        raise ValueError(
            "concrete parameter declaration name must be non-empty"
        )
    if isinstance(declaration, (BoundedVariable, UnboundedVariable)):
        return [declaration]
    if isinstance(declaration, (BoundVector, DenseTensorVariable)):
        return _tensor_declarations(declaration)
    raise TypeError(
        "concrete parameter declarations must be scalar, vector, or dense "
        "tensor variables"
    )


def _flatten_declarations(
    declarations: Sequence[
        BoundedVariable | UnboundedVariable | BoundVector | DenseTensorVariable
    ],
) -> list[BoundedVariable | UnboundedVariable]:
    return [
        scalar
        for declaration in declarations
        for scalar in _flatten_declaration(declaration)
    ]


def _reconstruct_concrete_values(
    values: object,
    declarations: Sequence[
        BoundedVariable | UnboundedVariable | BoundVector | DenseTensorVariable
    ],
) -> tuple[object, ...]:
    offset = 0
    result = []
    for declaration in declarations:
        size = (
            1
            if isinstance(declaration, (BoundedVariable, UnboundedVariable))
            else declaration.size
        )
        block = values[offset : offset + size]
        result.append(
            block[0]
            if isinstance(declaration, (BoundedVariable, UnboundedVariable))
            else np.reshape(block, declaration.shape)
        )
        offset += size
    return tuple(result)


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
    function_declarations: list[tuple[object, ...] | None] = []
    width = 0
    for index, (target, declaration) in enumerate(zip(space, declarations)):
        if isinstance(target, FunctionSpace):
            if not isinstance(declaration, FunctionParameter):
                raise TypeError(
                    f"Parameter {index} must implement FunctionParameter, got "
                    f"{type(declaration).__name__}"
                )
            declaration.validate_target(target)
            concrete_declarations = _function_concrete_declarations(
                declaration
            )
            flattened = _flatten_declarations(concrete_declarations)
            function_declarations.append(concrete_declarations)
        else:
            flattened = _flatten_declarations((declaration,))
            function_declarations.append(None)
            expected_size = (
                target.size if isinstance(target, VectorSpace) else 1
            )
            if len(flattened) != expected_size:
                raise ValueError(
                    f"Parameter {index} declares {len(flattened)} decisions, "
                    f"expected {expected_size}"
                )
        size = len(flattened)
        offsets.append((width, width + size))
        solver_declarations.extend(flattened)
        width += size

    numeric_parameters = VectorSpace("p", width)

    parameterizations = tuple(
        (
            declaration.build_function(target, system.backend())
            if isinstance(target, FunctionSpace)
            else None
        )
        for target, declaration in zip(space, declarations)
    )

    def reconstruct_parameters(parameters):
        values = []
        for (
            target,
            declaration,
            offset,
            parameterization,
            concrete_declarations,
        ) in zip(
            space,
            declarations,
            offsets,
            parameterizations,
            function_declarations,
        ):
            start, end = offset
            if parameterization is not None:
                assert concrete_declarations is not None
                values.append(
                    BoundCallable(
                        parameterization,
                        target,
                        _reconstruct_concrete_values(
                            parameters[start:end], concrete_declarations
                        ),
                    )
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
        ),
        solver_declarations,
        ParameterValueLayout(
            tuple(space), tuple(declarations), tuple(offsets)
        ),
    )
