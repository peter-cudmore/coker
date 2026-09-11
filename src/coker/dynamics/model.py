from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from coker.algebra.kernel import (
    Dimension,
    Function,
    FunctionSpace,
    Noop,
    Scalar,
    VectorSpace,
)


@dataclass
class DynamicsSpec:
    inputs: FunctionSpace
    parameters: Scalar | VectorSpace
    algebraic: Optional[VectorSpace]

    initial_conditions: Callable[
        [VectorSpace, VectorSpace], Tuple[np.ndarray, np.ndarray]
    ]
    """ [x, z] = initial_conditions(t_0, p) """

    dynamics: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    """dx = dynamics(t, x, z, u, p)"""

    constraints: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    """ g(t, x, z, u, p) = constraints(t, x, z, u, p) = 0."""

    outputs: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ]
    """ y(t) = outputs(t, x, z, u, p, q) """

    quadratures: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    """ dq/dt = quadratures(t, x, u, p) """


@dataclass
class DynamicalSystem:
    inputs: FunctionSpace
    parameters: VectorSpace | Scalar
    x0: Function
    dxdt: Function
    g: Optional[Function]
    dqdt: Optional[Function]
    y: Function
    solver_parameters: Optional[object] = field(default=None)

    def get_state_dimensions(self) -> Tuple[Dimension, Dimension, Dimension]:
        _t, x_dim, z_dim, _u, _p, q_dim = self.y.input_shape()
        return x_dim, z_dim, q_dim

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
        t, _x_dim, _z_dim, u, p, _q_dim = self.y.input_shape()
        (out,) = self.y.output_shape()
        args = [t.to_space("t")]
        if self.inputs is not Noop():
            args.append(u if isinstance(u, FunctionSpace) else u.to_space("u"))
        if p is not None:
            args.append(p.to_space("p"))
        return FunctionSpace("y", args, [out.to_space("y")])
