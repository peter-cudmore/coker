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

    def __init__(self, *spaces: Scalar | VectorSpace | None):
        self.spaces = spaces

    def offsets(self):
        total = 0
        for space in self.spaces:
            yield total
            if space is not None:
                total += space.size

    def sizes(self):
        for space in self.spaces:
            if space is None:
                yield 0
            else:
                yield space.size

    def dim(self) -> Scalar | VectorSpace | None:

        dim = sum(self.sizes())
        return VectorSpace(f"composition", (dim,)) if dim else None

    def inverse(self, ab) -> List:
        next_slice = ab
        output = []
        for space in self.spaces:
            if space is None:
                output.append(None)
            else:
                dim = space.size
                output.append(next_slice[:dim])
                next_slice = (
                    next_slice[dim:] if dim < next_slice.shape[0] else []
                )
        return output

    def __call__(self, *args):
        output = [
            a for a, space in zip(args, self.spaces) if space is not None
        ]
        if not output:
            return None

        return np.concatenate(output)

    @staticmethod
    def from_dimensions(name: str, *dims) -> "CompositionOperator":
        spaces = [
            VectorSpace(f"{name}_{i}", dim) if dim else None
            for i, dim in enumerate(dims)
        ]
        return CompositionOperator(*spaces)

    def as_matrices(self):
        offsets = list(self.offsets())
        sizes = list(self.sizes())

        matrices = []
        for i, (offset, size) in enumerate(zip(offsets, sizes)):
            matrix = np.zeros((size, sum(self.sizes())), dtype=float)

            matrix[:, offset : offset + size] = np.eye(size)
            matrices.append(matrix)
        return matrices


@dataclasses.dataclass
class ProjectionSet:
    parameters: CompositionOperator
    state: CompositionOperator
    algebraic: CompositionOperator
    quadratures: CompositionOperator
    outputs: CompositionOperator
    controls: CompositionOperator


def direct_sum(
    *systems: DynamicalSystem, backend=None
) -> Tuple[DynamicalSystem, ProjectionSet]:

    backend = backend or systems[0].backend()

    proj_p = CompositionOperator(*[system.parameters for system in systems])
    x_dim, z_dim, q_dim = zip(
        *[system.get_state_dimensions() for system in systems]
    )
    y_dim = [system.y.output_shape()[0] for system in systems]

    proj_x = CompositionOperator.from_dimensions("x", *x_dim)
    proj_z = CompositionOperator.from_dimensions("z", *z_dim)
    proj_q = CompositionOperator.from_dimensions("q", *q_dim)

    proj_y = CompositionOperator.from_dimensions("y", *y_dim)

    u_range = [
        (
            system.inputs.output_dimensions()[0]
            if system.inputs is not Noop()
            else None
        )
        for system in systems
    ]

    proj_u = CompositionOperator.from_dimensions(f"u", *u_range)
    if proj_u.dim() is None:
        u_space = Noop()
    else:
        u_space = FunctionSpace(
            proj_u.dim().name, [Scalar("t")], [proj_u.dim()]
        )

    def x0_impl(t, u_outer, p_outer):
        p_inner = proj_p.inverse(p_outer)
        u_inner = [
            lambda t: proj_u.inverse(u_outer(t))[i]
            for i, _ in enumerate(proj_u.spaces)
        ]
        x0_inner, z0_inner = zip(
            *[
                system.x0.call_inline(t, u_i, p_i)
                for system, u_i, p_i in zip(systems, u_inner, p_inner)
            ]
        )
        x0_ab = proj_x(*x0_inner)
        z0_ab = proj_z(*z0_inner)
        return x0_ab, z0_ab

    x0 = function(
        [Scalar("t"), u_space, proj_p.dim()], x0_impl, backend=backend
    )

    def dxdt_impl(t, x_outer, z_outer, u_outer, p_outer):
        p = proj_p.inverse(p_outer)
        x = proj_x.inverse(x_outer)
        z = proj_z.inverse(z_outer)

        dx_inner = [
            system.dxdt.call_inline(
                t,
                x[i],
                z[i],
                lambda t_i: proj_u.inverse(u_outer(t_i))[i],
                p[i],
            )
            for i, system in enumerate(systems)
        ]

        return proj_x(*dx_inner)

    args = [Scalar("t"), proj_x.dim(), proj_z.dim(), u_space, proj_p.dim()]
    dx = function(args, dxdt_impl, backend=backend)
    if proj_z.dim() is not None:

        def g_impl(t, x_outer, z_outer, u_outer, p_outer):
            p = proj_p.inverse(p_outer)
            x = proj_x.inverse(x_outer)
            z = proj_z.inverse(z_outer)
            g_inner = [
                system.g.call_inline(
                    t,
                    x[i],
                    z[i],
                    lambda t_i: proj_u.inverse(u_outer(t_i))[i],
                    p[i],
                )
                for i, system in enumerate(systems)
            ]
            return proj_z(*g_inner)

        g = function(args, g_impl, backend=backend)
    else:
        g = Noop()

    if proj_q.dim() is not None:

        def dqdt_impl(t, x_outer, z_outer, u_outer, p_outer):
            p = proj_p.inverse(p_outer)
            x = proj_x.inverse(x_outer)
            z = proj_z.inverse(z_outer)
            dq_inner = [
                system.dqdt.call_inline(
                    t,
                    x[i],
                    z[i],
                    lambda t_i: proj_u.inverse(u_outer(t_i))[i],
                    p[i],
                )
                for i, system in enumerate(systems)
            ]
            return proj_q(*dq_inner)

        dqdt = function(args, dqdt_impl, backend=backend)
    else:
        dqdt = Noop()

    def y_impl(t, x_outer, z_outer, u_outer, p_outer, q_outer):
        p = proj_p.inverse(p_outer)
        x = proj_x.inverse(x_outer)
        z = proj_z.inverse(z_outer)
        q = proj_q.inverse(q_outer)
        y_inner = [
            system.y.call_inline(
                t,
                x[i],
                z[i],
                lambda t_i: proj_u.inverse(u_outer(t_i))[i],
                p[i],
                q[i],
            )
            for i, system in enumerate(systems)
        ]

        return proj_y(*y_inner)

    y = function(args + [proj_q.dim()], y_impl, backend=backend)

    system = DynamicalSystem(
        inputs=u_space,
        parameters=proj_p.dim(),
        x0=x0,
        dxdt=dx,
        g=g,
        dqdt=dqdt,
        y=y,
    )
    projections = ProjectionSet(
        state=proj_x,
        algebraic=proj_z,
        quadratures=proj_q,
        outputs=proj_y,
        controls=proj_u,
        parameters=proj_p,
    )

    return system, projections
