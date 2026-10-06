"""Implicit residual dynamical-system model and legacy conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias, cast

import numpy as np

from coker.algebra.dimensions import Dimension, FunctionSpace, VectorSpace
from coker.algebra.function import Function, function
from coker.algebra.graph import TraceContext
from coker.algebra.ops import Noop
from coker.dynamics.model import (
    DynamicalSystem,
    DynamicsParameters,
    _TrajectorySystem,
)


@dataclass(frozen=True)
class LegacyFormCertificate:
    """Certified public semi-explicit lowering; never an implicit solve."""

    rate: Function
    quadrature: Function | Noop
    algebraic: Function | Noop


@dataclass(frozen=True)
class ResidualDynamicalSystem(_TrajectorySystem):
    """Dynamical system defined by one implicit residual function.

    ``F`` receives time, differential variables, their rates, algebraic
    variables, inputs, and parameters.  The dimension declarations describe
    how the public initial-condition and output functions partition those
    variables; they do not impose a semi-explicit form on ``F``.  A declared
    nonempty quadrature starts at zero for trajectory evaluation.
    """

    inputs: FunctionSpace | Noop
    parameters: DynamicsParameters
    x0: Function
    F: Function
    y: Function
    differential: Dimension
    algebraic: Dimension | None
    quadrature: Dimension | None
    solver_parameters: object | None = None
    legacy: LegacyFormCertificate | None = None

    def __post_init__(self) -> None:
        _validate_residual_system(self)

    def get_state_dimensions(
        self,
    ) -> tuple[Dimension, Dimension | None, Dimension | None]:
        return self.differential, self.algebraic, self.quadrature

    def backend(self) -> str:
        return self.F.backend

    def _initial_quadrature(self) -> np.ndarray | None:
        if self.quadrature is None or self.quadrature.flat() == 0:
            return None
        return np.zeros(self.quadrature.flat())

    def __call__(self, *args):
        return self._evaluate_integrals(self, *args)


_DirectIntegrationCallback: TypeAlias = Callable[..., Any]
_DirectIntegrationCallbacks: TypeAlias = tuple[
    _DirectIntegrationCallback,
    _DirectIntegrationCallback | Noop | None,
    _DirectIntegrationCallback | Noop | None,
]
_DirectIntegrationFunctionSource: TypeAlias = (
    DynamicalSystem
    | ResidualDynamicalSystem
    | _DirectIntegrationCallbacks
    | list[_DirectIntegrationCallback | Noop | None]
)


def _normalise_direct_integration_functions(
    functions: _DirectIntegrationFunctionSource,
) -> _DirectIntegrationCallbacks:
    """Return the semi-explicit callbacks supported by direct integrators."""
    if isinstance(functions, ResidualDynamicalSystem):
        certificate = functions.legacy
        if certificate is None:
            raise NotImplementedError(
                "Direct integration does not support independent residual "
                "systems without a semi-explicit certificate"
            )
        return (
            certificate.rate,
            certificate.algebraic,
            certificate.quadrature,
        )
    if isinstance(functions, DynamicalSystem):
        return functions.dxdt, functions.g, functions.dqdt

    rate, algebraic, quadrature = functions
    return (
        cast(_DirectIntegrationCallback, rate),
        cast(_DirectIntegrationCallback | Noop | None, algebraic),
        cast(_DirectIntegrationCallback | Noop | None, quadrature),
    )


def _validate_residual_system(system: ResidualDynamicalSystem) -> None:
    """Validate the invariants required to evaluate a residual system."""
    _dimension_size(system.differential, "differential")
    _optional_dimension_size(system.algebraic, "algebraic")
    _optional_dimension_size(system.quadrature, "quadrature")

    _validate_callback(system.F, "residual")
    _validate_callback(system.x0, "initial-condition", system.F.backend)
    _validate_callback(system.y, "output", system.F.backend)
    _validate_residual_input_arity(system.F, system.parameters)


def _dimension_size(dimension: object, name: str) -> int:
    if not isinstance(dimension, Dimension):
        raise TypeError(f"{name} dimension must be a Dimension")
    try:
        size = dimension.flat()
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} dimension must be finite") from ex
    if size < 0:
        raise ValueError(f"{name} dimension must not have a negative size")
    return size


def _optional_dimension_size(dimension: object, name: str) -> int:
    if dimension is None:
        return 0
    return _dimension_size(dimension, name)


def _validate_residual_input_arity(
    callback: Function, parameters: DynamicsParameters
) -> None:
    parameter_count = len(parameters) if isinstance(parameters, (tuple, list)) else 1
    expected_count = 5 + parameter_count
    actual_count = len(callback.input_spaces())
    if actual_count != expected_count:
        raise TypeError(
            f"residual callback has {actual_count} inputs; "
            f"expected {expected_count}"
        )


def _validate_callback(callback: object, name: str, backend: str | None = None) -> None:
    if not isinstance(callback, Function):
        raise TypeError(f"{name} callback must be a Function")
    if backend is not None and callback.backend != backend:
        raise ValueError(
            f"{name} callback uses backend {callback.backend!r}, "
            f"expected {backend!r}"
        )


def _compose_residual_callback(callback: Function, *arguments: Any) -> Any:
    """Record callable inputs without switching the callback's backend."""
    if not any(isinstance(space, FunctionSpace) for space in callback.input_spaces()):
        return callback(*arguments)

    tape = TraceContext.get_local_tape()
    assert tape is not None
    outputs = callback._append_symbolic_call(tape, arguments)
    return outputs[0] if callback.is_single else tuple(outputs)


def to_residual_dynamical_system(
    system: DynamicalSystem,
) -> ResidualDynamicalSystem:
    """Convert a semi-explicit system into an equivalent residual system."""
    x_dim, z_dim, q_dim = system.get_state_dimensions()
    state_size = x_dim.flat()
    quadrature_size = 0 if q_dim is None else q_dim.flat()
    algebraic_size = 0 if z_dim is None else z_dim.flat()
    rate_size = state_size + quadrature_size
    state_rows = slice(0, state_size)
    quadrature_rows = slice(state_size, rate_size)
    w_space = VectorSpace("_residual_w", rate_size)
    wdot_space = VectorSpace("_residual_wdot", rate_size)
    z_space = VectorSpace("_residual_z", algebraic_size)
    time_space, _, _, input_space, *parameter_spaces = system.dxdt.input_spaces()
    quadrature = system.dqdt
    algebraic = system.g

    if quadrature_size and not isinstance(quadrature, Function):
        raise ValueError("system quadrature layout requires a quadrature callback")
    if algebraic_size and not isinstance(algebraic, Function):
        raise ValueError("system algebraic layout requires a constraint callback")

    legacy = LegacyFormCertificate(
        rate=system.dxdt,
        quadrature=quadrature if isinstance(quadrature, Function) else Noop(),
        algebraic=algebraic if isinstance(algebraic, Function) else Noop(),
    )

    def residual(t: Any, w: Any, wdot: Any, z: Any, u: Any, *parameters: Any) -> Any:
        x = w[state_rows]
        z_public = z if algebraic_size else None
        state_rate = _compose_residual_callback(
            system.dxdt, t, x, z_public, u, *parameters
        )
        state_residual = wdot[state_rows] - np.reshape(state_rate, (state_size,))
        rows = [state_residual]
        if quadrature_size:
            assert isinstance(quadrature, Function)
            quadrature_rate = _compose_residual_callback(
                quadrature, t, x, z_public, u, *parameters
            )
            rows.append(
                wdot[quadrature_rows] - np.reshape(quadrature_rate, (quadrature_size,))
            )
        if algebraic_size:
            assert isinstance(algebraic, Function)
            algebraic_residual = _compose_residual_callback(
                algebraic, t, x, z, u, *parameters
            )
            rows.append(np.reshape(algebraic_residual, (algebraic_size,)))
        return rows[0] if len(rows) == 1 else np.concatenate(rows)

    residual_function = function(
        [
            time_space,
            w_space,
            wdot_space,
            z_space,
            input_space,
            *parameter_spaces,
        ],
        cast(Callable[..., Any], residual),
        backend=system.backend(),
        name="_residual",
    )
    return ResidualDynamicalSystem(
        inputs=system.inputs,
        parameters=system.parameters,
        x0=system.x0,
        F=residual_function,
        y=system.y,
        differential=x_dim,
        algebraic=z_dim,
        quadrature=q_dim,
        solver_parameters=system.solver_parameters,
        legacy=legacy,
    )
