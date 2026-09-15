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
    parameter_blocks: dict[str, tuple[int, int, tuple[int, ...]]] = field(
        default_factory=dict
    )

    def get_state_dimensions(self) -> Tuple[Dimension, Dimension, Dimension]:
        shapes = self.y.input_shape()
        return shapes[1], shapes[2], shapes[-1]

    def backend(self):
        return self.dxdt.backend

    def _map_arguments(self, *args):
        arg_stack = list(reversed(args))
        t = arg_stack.pop()
        try:
            u = (
                arg_stack.pop()
                if self.inputs is not Noop() or len(args) == 3
                else None
            )
            p = (
                arg_stack.pop()
                if self.parameters is not None or len(args) == 3
                else None
            )
        except IndexError as ex:
            raise ValueError(
                "Invalid number of arguments: expected 2 - 3, "
                f"received: {len(args)}"
            ) from ex

        return t, u, p

    def __call__(self, *args):
        from coker.backends import get_backend_by_name

        t, u, p = self._map_arguments(*args)

        x0, z0 = self.x0(0, u, p)

        # solve ODE
        # x' = dxdt(...)
        # 0  = g(...)
        # to get x,z over the interval

        if self.dqdt is not Noop():
            # zeros, the same size a q
            raise NotImplementedError
        else:
            q0 = None

        backend = get_backend_by_name(self.dxdt.backend)
        x, z, q = backend.evaluate_integrals(
            [self.dxdt, self.g, self.dqdt],
            [x0, z0, q0],
            t,
            [u, p],
            solver_parameters=self.solver_parameters,
        )

        if isinstance(t, (float, int)):
            return self.y(t, x, z, u, p, q)

        def map_args(i):
            x_i = x[:, i]
            z_i = z[:, i] if z is not None else None
            q_i = q[:, i] if q is not None else None
            return x_i, z_i, u, p, q_i

        if self.y.output_shape()[0].is_scalar() or self.y.output_shape()[
            0
        ].dim == (1,):
            y = np.concatenate(
                [self.y(t_i, *map_args(i)) for i, t_i in enumerate(t)]
            )
        else:
            y = np.vstack(
                [self.y(t_i, *map_args(i)) for i, t_i in enumerate(t)]
            )

        return y

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
