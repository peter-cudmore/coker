from typing import Callable, List, Optional
import numpy as np
from coker import VectorSpace, FunctionSpace, function, Scalar, Noop

from .types import DynamicsSpec, DynamicalSystem
from ..algebra import is_scalar


def create_dynamics_from_spec(
    spec: DynamicsSpec, backend="numpy"
) -> DynamicalSystem:

    # just put a dummy value in here so that
    # the shape calculation doesn't spew.

    x0 = function(
        arguments=[
            spec.algebraic,
            spec.inputs,
            spec.parameters,
        ],
        implementation=spec.initial_conditions,
        backend=backend,
    )

    assert (
        len(x0.output) == 2
    ), "Initial conditions must a pair, one for the state and one for the algebraic variables"

    state = x0.output[0]
    algebraic = x0.output[1]

    state_space = VectorSpace("x", state.dim.flat())

    if algebraic is not None:
        assert (
            algebraic.dim == spec.algebraic.dim
        ), "Initial algebraic conditions must have the same dimension as the algebraic variables"

    # Order: t, x, z, u, p
    arguments = [
        Scalar("t"),
        state_space,
        spec.algebraic,
        spec.inputs,
        spec.parameters,
    ]

    xdot = function(arguments, spec.dynamics, backend)

    assert len(xdot.output) == 1, "Dynamics must return a single vector"

    assert (
        xdot.output[0].dim.shape == state.dim.shape
    ), f"Dynamics must return a vector of the same dimension as the state: x0 gave {state.dim} and dynamics gave {xdot.output[0].dim}"

    if spec.algebraic is not None:
        assert (
            spec.constraints is not Noop()
        ), "If algebraic constraints are specified, constraints must be specified"
    elif spec.constraints is not Noop():
        raise ValueError("Constraints specified, but no algebraic variables")

    constraint = (
        function(arguments, spec.constraints, backend)
        if spec.algebraic is not None
        else Noop()
    )

    quadrature = (
        function(arguments, spec.quadratures, backend)
        if spec.quadratures is not Noop()
        else Noop()
    )
    if quadrature is not Noop():
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


def create_control_system(
    x0: Callable[[np.ndarray], np.ndarray],
    xdot: Callable[[float, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    control: FunctionSpace,
    parameters: Optional[Scalar | VectorSpace] = None,
    output: Callable[
        [Scalar, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ] = None,
    backend: str = "numpy",
    p_init: np.ndarray = None,
    u_init: Callable[[float], np.ndarray] = None,
) -> DynamicalSystem:

    if isinstance(x0, (list, tuple, int, float)):
        x0 = np.array(x0)
    if isinstance(x0, np.ndarray):
        x0_func = lambda z, u, p: (x0, None)
    else:
        x0_func = lambda z, u, p: (x0(p), None)

    if p_init is not None:
        assert (
            parameters is not None
        ), "Parameters must be specified if p_init is specified"
        if isinstance(parameters, Scalar):
            assert isinstance(p_init, (float, int)) or p_init.shape == (
                1,
            ), "p_init must be a scalar or a 1D array"
            p_init = np.array([p_init])
        else:
            assert p_init.shape == (
                parameters.dimension,
            ), "p_init must be a 1D array of the same size as the parameters"
    else:
        p_init = None if parameters is None else np.zeros(parameters.dimension)

    assert len(control.output) == 1, "Control must return a single vector"
    (u_dim,) = control.output_dimensions()
    if u_init is not None:
        assert callable(u_init), "u_init must be a callable"
        u0 = u_init(0)
        assert (
            u0.shape == u_dim.shape
        ), f"u0 must have the same shape as the control output; u0 is {u0} and control output is {u_dim}"
    else:
        u0 = np.zeros(u_dim.shape)

    x0_eval, _z0 = x0_func(None, u0, p_init)
    dot_x_eval = xdot(0, x0_eval, u0, p_init)

    assert (
        is_scalar(x0_eval) and is_scalar(dot_x_eval)
    ) or x0_eval.shape == dot_x_eval.shape, f"x0 and xdot must have the same shape; x0 is {x0_eval} and xdot is {dot_x_eval}"

    if output is None:
        output_func = lambda t, x, z, u, p, q: x
    else:
        y_eval = output(0, x0_eval, u0, p_init)
        assert y_eval is not None, "Output function must return a value"
        output_func = lambda t, x, z, u, p, q: output(t, x, u(t), p)

    dynamics = lambda t, x, z, u, p: xdot(t, x, u(t), p)

    spec = DynamicsSpec(
        inputs=control,
        parameters=parameters,
        algebraic=None,
        initial_conditions=x0_func,
        dynamics=dynamics,
        constraints=Noop(),
        outputs=output_func,
        quadratures=Noop(),
    )

    return create_dynamics_from_spec(spec, backend=backend)


def create_autonomous_ode(
    x0: Callable[[List[np.ndarray]], np.ndarray],
    xdot: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    parameters: Optional[Scalar | VectorSpace] = None,
    output: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    backend: str = "numpy",
    p_init: np.ndarray = None,
) -> DynamicalSystem:

    # case 1,
    # - x0 is an array

    if isinstance(x0, (list, tuple, int, float)):
        x0 = np.array(x0)
    if isinstance(x0, np.ndarray):
        x0_func = lambda z, u, p: (x0, None)
    else:
        x0_func = lambda z, u, p: (x0(p), None)

    if p_init is not None:
        assert (
            parameters is not None
        ), "Parameters must be specified if p_init is specified"
        if isinstance(parameters, Scalar):
            assert isinstance(p_init, (float, int)) or p_init.shape == (
                1,
            ), "p_init must be a scalar or a 1D array"
            p_init = np.array([p_init])
        else:
            assert p_init.shape == (
                parameters.dimension,
            ), "p_init must be a 1D array of the same size as the parameters"
    else:
        p_init = None if parameters is None else np.zeros(parameters.dimension)

    x0_eval, _z0 = x0_func(None, None, p_init)
    dot_x_eval = xdot(x0_eval, p_init)

    assert (
        is_scalar(x0_eval) and is_scalar(dot_x_eval)
    ) or x0_eval.shape == dot_x_eval.shape, f"x0 and xdot must have the same shape; x0 is {x0_eval} and xdot is {dot_x_eval}"

    if output is None:
        output_func = lambda t, x, z, u, p, q: x
    else:
        y_eval = output(x0_eval, p_init)
        assert y_eval is not None, "Output function must return a value"
        output_func = (lambda t, x, z, u, p, q: output(x, p),)

    dynamics = lambda t, x, z, u, p: xdot(x, p)

    spec = DynamicsSpec(
        inputs=Noop(),
        parameters=parameters,
        algebraic=None,
        initial_conditions=x0_func,
        dynamics=dynamics,
        constraints=Noop(),
        outputs=output_func,
        quadratures=Noop(),
    )

    return create_dynamics_from_spec(spec, backend=backend)
