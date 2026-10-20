"""Shared trajectory preparation and output assembly without graph conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from coker.algebra.dimensions import FunctionSpace
from coker.algebra.function import Function, function
from coker.algebra.ops import Noop
from coker.parameters.function_parameters import FittedFunction

if TYPE_CHECKING:
    from coker.dynamics.model import DynamicalSystem
    from coker.dynamics.residual import ResidualDynamicalSystem


def evaluate(system: DynamicalSystem | ResidualDynamicalSystem, *args):
    """Evaluate a trajectory through its backend using the original model."""
    time, inputs, parameters = _map_arguments(system, args)
    from coker.backends import get_backend_by_name

    backend = get_backend_by_name(system.backend)
    inputs = _prepare_value(backend, inputs)
    parameters = tuple(_prepare_value(backend, parameter) for parameter in parameters)
    x0, z0 = system.x0(0, inputs, *parameters)
    q0 = _initial_quadrature(system)
    x, z, q = backend.evaluate_integrals(
        system,
        [x0, z0, q0],
        time,
        [inputs, *parameters],
        solver_parameters=system.solver_parameters,
    )
    if isinstance(time, (float, int)):
        return system.y(time, x, z, inputs, *parameters, q)

    def map_arguments(index):
        x_i = x[:, index]
        z_i = z[:, index] if z is not None else None
        q_i = q[:, index] if q is not None else None
        return x_i, z_i, inputs, *parameters, q_i

    if system.y.output_shape()[0].is_scalar() or system.y.output_shape()[0].dim == (1,):
        return np.concatenate(
            [
                system.y(time_i, *map_arguments(index))
                for index, time_i in enumerate(time)
            ]
        )
    return np.vstack(
        [system.y(time_i, *map_arguments(index)) for index, time_i in enumerate(time)]
    )


def _map_arguments(system, args):
    declarations = (
        ()
        if system.parameters is None
        else (
            system.parameters
            if isinstance(system.parameters, (tuple, list))
            else (system.parameters,)
        )
    )
    has_inputs = not isinstance(system.inputs, Noop)
    expected_count = 1 + int(has_inputs) + len(declarations)
    if len(args) != expected_count:
        raise ValueError(
            "Trajectory argument count does not match the system declaration: "
            f"expected {expected_count}, got {len(args)}"
        )
    input_value = args[1] if has_inputs else None
    parameter_start = 1 + int(has_inputs)
    parameters = tuple(
        _prepare_parameter(system, declaration, value, index)
        for index, (declaration, value) in enumerate(
            zip(declarations, args[parameter_start:])
        )
    )
    return args[0], input_value, parameters if declarations else (None,)


def _prepare_parameter(system, declaration, value, index):
    if not isinstance(declaration, FunctionSpace):
        return value
    if isinstance(value, Function):
        if value.backend != system.backend:
            raise ValueError(
                f"Function-valued parameter {index} uses backend "
                f"{value.backend!r}, expected {system.backend!r}"
            )
        if value not in declaration:
            raise ValueError(
                f"Function-valued parameter {index} does not match declared "
                f"FunctionSpace {declaration.name!r}"
            )
        return value
    if isinstance(value, FittedFunction) and value not in declaration:
        raise ValueError(
            f"Function-valued parameter {index} does not match declared "
            f"FunctionSpace {declaration.name!r}"
        )
    if not callable(value):
        raise TypeError(f"Function-valued parameter {index} must be callable")
    prepared = function(declaration.arguments, value, backend=system.backend)
    if prepared not in declaration:
        raise ValueError(
            f"Function-valued parameter {index} does not match declared "
            f"FunctionSpace {declaration.name!r}"
        )
    return prepared


def _prepare_value(backend, value):
    return (
        value if value is None or callable(value) else backend.to_backend_array(value)
    )


def _initial_quadrature(system) -> np.ndarray | None:
    quadrature = system.get_state_dimensions()[2]
    if quadrature is None or quadrature.flat() == 0:
        return None
    return np.zeros(quadrature.flat())
