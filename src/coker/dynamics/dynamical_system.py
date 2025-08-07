from typing import Callable, List, Optional
import numpy as np
from coker import (
    VectorSpace,
    FunctionSpace,
    function,
    Scalar,
)

from .types import DynamicsSpec, DynamicalSystem


def create_dynamics_from_spec(spec: DynamicsSpec, backend=None):

    if backend is None:
        backend = "casadi"

    # just put a dummy value in here so that
    # the shape calculation doesn't spew.
    inputs = spec.inputs or FunctionSpace("u",[Scalar("t")], None)

    x0 = function(
        arguments=[
            Scalar("t"),
            spec.algebraic,
            inputs,
            spec.parameters,
        ],
        implementation=spec.initial_conditions,
        backend=backend,
    )

    assert len(x0.output) == 1, ""

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
        inputs,
        spec.parameters,
    ]

    xdot = function(arguments, spec.dynamics, backend)

    assert len(xdot.output) == len(x0.output) == 1

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
    inputs: Optional[FunctionSpace],
    parameters=Optional[Scalar | VectorSpace],
    x0=Callable[[np.ndarray, List[np.ndarray]], np.ndarray],
    xdot=Callable[[np.ndarray, np.ndarray, List[np.ndarray]], np.ndarray],
    output=Callable[[np.ndarray, np.ndarray, List[np.ndarray]], np.ndarray],
    backend="coker",
) -> "DynamicalSystem":


    spec = DynamicsSpec(
        inputs=inputs,
        parameters=parameters,
        algebraic=None,
        initial_conditions=lambda t, z, u, p: x0(u(t), p),
        dynamics=lambda t, x, z, u, p: xdot(x, u(t), p),
        constraints=None,
        outputs=lambda t, x, z, u, p, q: output(x, u(t), p),
        quadratures=None,
    )

    return create_dynamics_from_spec(spec, backend=backend)
