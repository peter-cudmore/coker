# Dynamical System
#
# Input Spaces:
# - Domain:     t in [t_0, t_1]
# - Control:    u(t)
# - Parameters: p
from dataclasses import dataclass
from typing import List, Callable

import numpy as np

from coker import (
    VectorSpace,
    FunctionSpace,
    Optional,
    function,
    Tuple,
    Scalar,
    Function,
    Dimension
)
from coker.backends import get_backend_by_name


# Internal Spaces
# - State:      x(t)
# - Algebraic:  z(t)
#
# Output Space:
# - y(t)
# - q(t)

# Kernels:
# Explicit f(x,u,z, p) = dx, g(x,u, z, p) = 0, dq = h(x, z, u, p), x0(t_0, p)
# Implicit F(dx, x, u, z, p) = 0, dq = h(x, z, u, p), x0(t_0, p)


# Interpret this as a function from ([0, T], u(t), p) -> (y(T), q(T))


@dataclass
class DynamicsSpec:
    inputs: FunctionSpace
    parameters: Scalar | VectorSpace

    algebraic: Optional[VectorSpace]

    initial_conditions: Callable[
        [float, VectorSpace, VectorSpace], Tuple[np.ndarray, np.ndarray]
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

    quadratures: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    ] = None
    """ dq/dt = quadratures(t, x, u, p) """


class DynamicalSystem:
    def __init__(
        self,
        inputs: FunctionSpace,
        parameters: VectorSpace | Scalar,
        x0: Function,
        dxdt: Function,
        g: Optional[Function],
        dqdt: Optional[Function],
        y: Function,
    ):
        self.inputs = inputs
        self.parameters = parameters
        self.x0 = x0
        self.dxdt = dxdt
        self.g = g
        self.dqdt = dqdt
        self.y = y

    def __call__(self, *args):
        if len(args) == 2:
            if self.inputs is not None:
                raise ValueError(
                    f"Invalid number of arguments: Expected 3, received: {len(args)}"
                )
            t, p = args
            u = lambda _: None  # No input so treat it as a noop
        elif len(args) == 3:
            t, u, p = args
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")

        if self.g is not None:
            # solve z s.t. 0 = g(0, x0, z, u(0), p)
            raise NotImplementedError
        else:
            z0 = None

        x0 = self.x0(0, z0, u(0), p)

        # solve ODE
        # x' = dxdt(...)
        # 0  = g(...)
        # to get x,z over the interval

        if self.dqdt is not None:
            # zeros, the same size a q
            raise NotImplementedError
        else:
            q0 = None

        backend = get_backend_by_name(self.dxdt.backend)
        x, z, q = backend.evaluate_integrals(
            [self.dxdt, self.g, self.dqdt], [x0, z0, q0], t, [u, p]
        )

        out = self.y(t, x, z, u(t), p, q)

        return out


def create_dynamics_from_spec(spec: DynamicsSpec, backend=None):

    if backend is None:
        backend = "casadi"

    x0 = function(
        arguments=[
            Scalar("t"),
            spec.constraints,
            spec.inputs,
            spec.parameters,
        ],
        implementation=spec.initial_conditions,
        backend=backend,
    )
    (state_space,) = [
        (
            VectorSpace("x", tracer.shape[0])
            if tracer.dim.is_vector()
            else Scalar("x")
        )
        for tracer in x0.output
    ]
    # Order: t, x, z, u, p
    arguments = [
        Scalar("t"),
        state_space,
        spec.algebraic,
        spec.inputs,
        spec.parameters,
    ]
    xdot = function(arguments, spec.dynamics, backend)

    assert all(
        x0.dim == dx.dim for x0, dx in zip(x0.output, xdot.output)
    ), f"Initial conditions and dynamics have different output dimensions"

    constraint = (
        function(arguments, spec.constraints, backend)
        if spec.algebraic
        else None
    )
    quadrature = (
        function(arguments, spec.quadratures, backend)
        if spec.quadratures
        else None
    )
    if quadrature is not None:
        assert (
            len(quadrature.output) == 1
        ), "Quadratures must be a scalar or vector space"
        q = quadrature.output[0]
        arguments.append(
            VectorSpace("q", q.dim.flat())
            if not q.dim.is_scalar()
            else Scalar("q")
        )
    else:
        arguments.append(None)

    output = function(arguments, spec.outputs, backend)

    return DynamicalSystem(
        spec.inputs, spec.parameters, x0, xdot, constraint, quadrature, output
    )


def create_homogenous_ode(
    inputs: FunctionSpace,
    parameters=Scalar | VectorSpace,
    x0=Callable[[np.ndarray, List[np.ndarray]], np.ndarray],
    xdot=Callable[[np.ndarray, np.ndarray, List[np.ndarray]], np.ndarray],
    output=Callable[[np.ndarray, np.ndarray, List[np.ndarray]], np.ndarray],
    backend="coker",
) -> "DynamicalSystem":

    spec = DynamicsSpec(
        inputs,
        parameters,
        algebraic=None,
        initial_conditions=lambda t, z, u, p: x0(u(0), p),
        dynamics=lambda t, x, z, u, p: xdot(x, u(t), p),
        constraints=None,
        outputs=lambda t, x, z, u, p, q: output(x, u(t), p),
        quadratures=None,
    )

    return create_dynamics_from_spec(spec, backend=backend)
