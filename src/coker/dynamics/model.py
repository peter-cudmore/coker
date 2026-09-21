from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, TypeAlias

import numpy as np
from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.function import Function
from coker.algebra.ops import Noop

ParameterDeclaration: TypeAlias = Scalar | VectorSpace | FunctionSpace
DynamicsParameters: TypeAlias = (
    ParameterDeclaration
    | tuple[ParameterDeclaration, ...]
    | list[ParameterDeclaration]
    | None
)


@dataclass
class DynamicsSpec:
    inputs: FunctionSpace | Noop
    parameters: DynamicsParameters
    algebraic: Optional[VectorSpace]

    initial_conditions: Callable
    """ [x, z] = initial_conditions(t_0, p) """

    dynamics: Callable
    """dx = dynamics(t, x, z, u, p)"""

    constraints: Callable
    """g(t, x, z, u, p) = 0."""

    outputs: Callable
    """y(t) = outputs(t, x, z, u, p, q)"""

    quadratures: Callable
    """dq/dt = quadratures(t, x, z, u, p)"""

    def __post_init__(self):
        if isinstance(self.parameters, list):
            self.parameters = tuple(self.parameters)
        if isinstance(self.parameters, tuple) and not all(
            isinstance(parameter, (Scalar, VectorSpace, FunctionSpace))
            for parameter in self.parameters
        ):
            raise TypeError(
                "parameter tuples must contain Scalar, VectorSpace, or "
                "FunctionSpace elements"
            )


@dataclass
class DynamicalSystem:
    inputs: FunctionSpace
    parameters: ParameterDeclaration | tuple[ParameterDeclaration, ...] | None
    x0: Function
    dxdt: Function
    g: Optional[Function]
    dqdt: Optional[Function]
    y: Function
    solver_parameters: Optional[object] = field(default=None)

    def get_state_dimensions(self) -> Tuple[Dimension, Dimension, Dimension]:
        shapes = self.y.input_shape()
        return shapes[1], shapes[2], shapes[-1]

    def backend(self):
        return self.dxdt.backend

    def _map_arguments(self, *args):
        values = list(args)
        if not values:
            raise ValueError(
                "A trajectory evaluation requires a time argument"
            )
        t = values.pop(0)

        if self.inputs is not Noop():
            if not values:
                raise ValueError(
                    "A trajectory evaluation requires an input value"
                )
            u = values.pop(0)
        elif not isinstance(self.parameters, tuple) and len(values) == 2:
            u = values.pop(0)
        else:
            u = None

        if isinstance(self.parameters, tuple):
            p = tuple(values)
            if len(p) != len(self.parameters):
                raise ValueError(
                    "Trajectory parameter count does not match the system "
                    f"declaration: expected {len(self.parameters)}, got {len(p)}"
                )
        elif self.parameters is None:
            if values:
                raise ValueError(
                    "Trajectory evaluation received unexpected parameters"
                )
            p = None
        else:
            if len(values) != 1:
                raise ValueError(
                    "Trajectory evaluation requires one packed parameter value"
                )
            p = values[0]
        return t, u, p

    def __call__(self, *args):
        from coker.backends import get_backend_by_name

        t, u, p = self._map_arguments(*args)
        parameter_arguments = p if isinstance(self.parameters, tuple) else (p,)
        x0, z0 = self.x0(0, u, *parameter_arguments)

        if self.dqdt is not Noop():
            raise NotImplementedError
        q0 = None

        backend = get_backend_by_name(self.dxdt.backend)
        x, z, q = backend.evaluate_integrals(
            [self.dxdt, self.g, self.dqdt],
            [x0, z0, q0],
            t,
            [u, *parameter_arguments],
            solver_parameters=self.solver_parameters,
        )

        if isinstance(t, (float, int)):
            return self.y(t, x, z, u, *parameter_arguments, q)

        def map_args(index):
            x_i = x[:, index]
            z_i = z[:, index] if z is not None else None
            q_i = q[:, index] if q is not None else None
            return x_i, z_i, u, *parameter_arguments, q_i

        if self.y.output_shape()[0].is_scalar() or self.y.output_shape()[
            0
        ].dim == (1,):
            return np.concatenate(
                [self.y(t_i, *map_args(index)) for index, t_i in enumerate(t)]
            )
        return np.vstack(
            [self.y(t_i, *map_args(index)) for index, t_i in enumerate(t)]
        )

    def output_as_function_space(self) -> FunctionSpace:
        t, _x, _z, u, *parameter_shapes, _q = self.y.input_shape()
        (out,) = self.y.output_shape()
        args = [t.to_space("t")]
        if self.inputs is not Noop():
            args.append(u if isinstance(u, FunctionSpace) else u.to_space("u"))
        if isinstance(self.parameters, tuple):
            for index, (element, shape) in enumerate(
                zip(self.parameters, parameter_shapes)
            ):
                args.append(
                    element
                    if isinstance(shape, FunctionSpace)
                    else shape.to_space(f"p{index}")
                )
        elif self.parameters is not None:
            (parameter_shape,) = parameter_shapes
            args.append(parameter_shape.to_space("p"))
        return FunctionSpace("y", args, [out.to_space("y")])
