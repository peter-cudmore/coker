import dataclasses
from typing import Callable, List, Optional
import numpy as np
from coker import VectorSpace, FunctionSpace, function, Scalar, Noop, Function

from .types import DynamicsSpec, DynamicalSystem
from ..algebra import is_scalar
from typing import Tuple

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



class CompositionOperator:

    def __init__(self, a: Scalar | VectorSpace | None, b: Scalar | VectorSpace | None):
        self.a = a
        self.b = b

    def dim(self) -> Scalar | VectorSpace | None:
        if self.a is None:
            return self.b
        if self.b is None:
            return self.a

        dim = self.a.size + self.b.size
        return VectorSpace(f"{self.a.name} + {self.b.name}", (dim,))

    def inverse(self, ab) -> Tuple:
        if self.a is None:
            return None, ab
        if self.b is None:
            return ab, None
        dim_a = self.a.size
        a = ab[:dim_a]
        dim_b = self.b.size
        b = ab[-dim_b:]
        return a, b

    def __call__(self, a, b):
        if self.a is None:
            return b
        if self.b is None:
            return a
        return  np.concatenate([a, b])

    @staticmethod
    def from_dimensions(name: str, dim_a, dim_b) -> 'CompositionOperator':
        return CompositionOperator(
            VectorSpace(f"{name}_a", dim_a) if dim_a else None,
            VectorSpace(f"{name}_b", dim_b) if dim_b else None
        )


@dataclasses.dataclass
class ProjectionSet:
    parameters: CompositionOperator
    state: CompositionOperator
    algebraic: CompositionOperator
    quadratures: CompositionOperator
    outputs: CompositionOperator
    controls: CompositionOperator



def direct_sum(a: DynamicalSystem, b: DynamicalSystem) -> Tuple[DynamicalSystem, ProjectionSet]:

    proj_p = CompositionOperator(a.parameters, b.parameters)
    a_x, a_z, a_q = a.get_state_dimensions()
    b_x, b_z, b_q = b.get_state_dimensions()

    proj_x = CompositionOperator.from_dimensions('x', a_x, b_x)
    proj_z = CompositionOperator.from_dimensions('z', a_z, b_z)
    proj_q = CompositionOperator.from_dimensions('q', a_q, b_q)
    y_a_range, = a.y.output_shape()
    y_b_range, = b.y.output_shape()
    proj_y = CompositionOperator.from_dimensions('y', y_a_range, y_b_range)


    u_a_range, = a.inputs.output_dimensions()[0] if a.inputs is not Noop() else None,
    u_b_range, = b.inputs.output_dimensions()[0] if b.inputs is not Noop() else None,

    proj_u = CompositionOperator.from_dimensions(f"u", u_a_range, u_b_range)
    if proj_u.dim() is None:
        u_space = Noop()
    else:
        u_space = FunctionSpace(proj_u.dim().name, [Scalar('t')], [proj_u.dim()])


    def x0_impl(t, u_outer, p_outer):
        p_a, p_b = proj_p.inverse(p_outer)

        u_a = lambda t: proj_u.inverse(u_outer(t))[0]
        u_b = lambda t: proj_u.inverse(u_outer(t))[1]

        x0_a, z0_a = a.x0.call_inline(t, u_a, p_a)
        x0_b, z0_b = b.x0.call_inline(t, u_b, p_b)
        x0_ab = proj_x(x0_a, x0_b)
        z0_ab = proj_z(z0_a, z0_b)
        return x0_ab, z0_ab

    x0=function([Scalar('t'), u_space, proj_p.dim()], x0_impl, a.backend())

    def dxdt_impl(t, x_outer, z_outer, u_outer, p_outer):
        p_a, p_b = proj_p.inverse(p_outer)
        x_a, x_b = proj_x.inverse(x_outer)
        z_a, z_b = proj_z.inverse(z_outer)
        u_a = lambda t: proj_u.inverse(u_outer(t))[0]
        u_b = lambda t: proj_u.inverse(u_outer(t))[1]
        dx_a = a.dxdt.call_inline(t, x_a, z_a, u_a, p_a)
        dx_b = b.dxdt.call_inline(t, x_b, z_b, u_b, p_b)
        return proj_x(dx_a, dx_b)


    args = [Scalar('t'), proj_x.dim(), proj_z.dim(), u_space, proj_p.dim()]
    dx=function(args, dxdt_impl, a.backend())
    if proj_z.dim() is not None:
        def g_impl(t, x_outer, z_outer, u_outer, p_outer):
            p_a, p_b = proj_p.inverse(p_outer)
            x_a, x_b = proj_x.inverse(x_outer)
            z_a, z_b = proj_z.inverse(z_outer)
            u_a = lambda t: proj_u.inverse(u_outer(t))[0]
            u_b = lambda t: proj_u.inverse(u_outer(t))[1]
            g_a = a.g.call_inline(t, x_a, z_a, u_a, p_a)
            g_b = b.g.call_inline(t, x_b, z_b, u_b, p_b)
            return proj_z(g_a, g_b)
        g=function(args, g_impl, a.backend())
    else:
        g = Noop()

    if proj_q.dim() is not None:


        def dqdt_impl(t, x_outer, z_outer, u_outer, p_outer):
            p_a, p_b = proj_p.inverse(p_outer)
            x_a, x_b = proj_x.inverse(x_outer)
            z_a, z_b = proj_z.inverse(z_outer)
            u_a = lambda t: proj_u.inverse(u_outer(t))[0]
            u_b = lambda t: proj_u.inverse(u_outer(t))[1]
            dq_a = a.dqdt.call_inline(t, x_a, z_a, u_a, p_a) if a.dqdt is not None else None
            dq_b = b.dqdt.call_inline(t, x_b, z_b, u_b, p_b) if b.dqdt is not None else None
            return proj_q(dq_a, dq_b)

        dqdt=function(args, dqdt_impl, a.backend())
    else:
        dqdt = Noop()

    def y_impl(t, x_outer, z_outer, u_outer, p_outer, q_outer):
        p_a, p_b = proj_p.inverse(p_outer)
        x_a, x_b = proj_x.inverse(x_outer)
        z_a, z_b = proj_z.inverse(z_outer)
        q_a, q_b = proj_q.inverse(q_outer)
        u_a = lambda t: proj_u.inverse(u_outer(t))[0]
        u_b = lambda t: proj_u.inverse(u_outer(t))[1]

        y_a = a.y.call_inline(t, x_a, z_a, u_a, p_a, q_a)
        y_b = b.y.call_inline(t, x_b, z_b, u_b, p_b, q_b)
        return proj_y(y_a, y_b)


    y=function(args + [proj_q.dim()], y_impl, a.backend())

    system =  DynamicalSystem(
        inputs=u_space,
        parameters=proj_p.dim(),
        x0=x0,
        dxdt=dx,
        g=g,
        dqdt=dqdt,
        y=y
    )
    projections = ProjectionSet(
        state=proj_x,
        algebraic=proj_z,
        quadratures=proj_q,
        outputs=proj_y,
        controls=proj_u,
        parameters=proj_p
    )

    return system, projections


#
#    return DynamicalSystem(
#        inputs=a.inputs + b.inputs,
#        parameters=a.parameters + b.parameters,
#        algebraic=a.algebraic + b.algebraic,
#        initial_conditions=lambda z, u, p: (
#            np.concatenate([a.initial_conditions(z, u, p)[0], b.initial_conditions(z, u, p)[0]]),
#            None,
#        ),
#        dynamics=lambda t, x, z, u, p: np.concatenate([a.dynamics(t, x, z, u, p), b.dynamics(t, x, z, u, p)]),
#        constraints=lambda z, u, p: np.concatenate([a.constraints(z, u, p), b.constraints(z, u, p)]),
#        outputs=lambda t, x, z, u, p: np.concatenate([a.outputs(t, x, z, u, p), b.outputs(t, x, z, u, p)]),
#    )



    raise NotImplementedError