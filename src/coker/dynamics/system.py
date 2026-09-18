import dataclasses
from typing import Callable, List, Optional
import numpy as np
from coker import VectorSpace, FunctionSpace, function, Scalar
from coker.algebra.dimensions import Dimension
from coker.algebra.ops import Noop

from .model import DynamicsSpec, DynamicalSystem
from ..algebra import is_scalar
from typing import Tuple


def _initial_parameters(callback):
    return lambda z, u, *p: callback(z, u, p)


def _dynamics_parameters(callback):
    return lambda t, x, z, u, *p: callback(t, x, z, u, p)


def _output_parameters(callback):
    def wrapped(t, x, z, u, *parameters_and_quadrature):
        *parameters, quadrature = parameters_and_quadrature
        return callback(t, x, z, u, tuple(parameters), quadrature)

    return wrapped


def _parameter_callback(callback, heterogeneous, adapter):
    return adapter(callback) if heterogeneous else callback


def create_dynamics_from_spec(
    spec: DynamicsSpec, backend="numpy"
) -> DynamicalSystem:
    heterogeneous = isinstance(spec.parameters, tuple)
    parameter_arguments = (
        list(spec.parameters) if heterogeneous else [spec.parameters]
    )
    x0 = function(
        arguments=[spec.algebraic, spec.inputs, *parameter_arguments],
        implementation=_parameter_callback(
            spec.initial_conditions, heterogeneous, _initial_parameters
        ),
        backend=backend,
    )
    assert len(x0.output) == 2, (
        "Initial conditions must be a pair, one for the state and one "
        "for the algebraic variables"
    )
    state, algebraic = x0.output
    state_space = VectorSpace("x", state.dim.flat())
    if algebraic is not None:
        assert algebraic.dim == Dimension(spec.algebraic.dimension), (
            "Initial algebraic conditions must have the same dimension "
            "as the algebraic variables"
        )
    arguments = [
        Scalar("t"),
        state_space,
        spec.algebraic,
        spec.inputs,
        *parameter_arguments,
    ]
    xdot = function(
        arguments,
        _parameter_callback(
            spec.dynamics, heterogeneous, _dynamics_parameters
        ),
        backend,
    )
    assert len(xdot.output) == 1, "Dynamics must return a single vector"
    assert xdot.output[0].dim.shape == state.dim.shape, (
        "Dynamics must return a vector of the same dimension as the "
        f"state: x0 gave {state.dim} and dynamics gave {xdot.output[0].dim}"
    )
    if spec.algebraic is not None:
        assert spec.constraints is not Noop(), (
            "If algebraic constraints are specified, constraints must "
            "also be specified"
        )
    elif spec.constraints is not Noop():
        raise ValueError("Constraints specified, but no algebraic variables")
    constraint = (
        function(
            arguments,
            _parameter_callback(
                spec.constraints, heterogeneous, _dynamics_parameters
            ),
            backend,
        )
        if spec.algebraic is not None
        else Noop()
    )
    quadrature = (
        function(
            arguments,
            _parameter_callback(
                spec.quadratures, heterogeneous, _dynamics_parameters
            ),
            backend,
        )
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
    output = function(
        arguments,
        _parameter_callback(spec.outputs, heterogeneous, _output_parameters),
        backend,
    )
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

        def x0_func(z, u, p):
            _ = z, u
            return x0, None

    else:

        def x0_func(z, u, p):
            _ = z, u
            return x0(p), None

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
        assert u0.shape == u_dim.shape, (
            "u0 must have the same shape as the control output; "
            f"u0 is {u0} and control output is {u_dim}"
        )
    else:
        u0 = np.zeros(u_dim.shape)

    x0_eval, _z0 = x0_func(None, u0, p_init)
    dot_x_eval = xdot(0, x0_eval, u0, p_init)

    assert (
        is_scalar(x0_eval) and is_scalar(dot_x_eval)
    ) or x0_eval.shape == dot_x_eval.shape, (
        "x0 and xdot must have the same shape; "
        f"x0 is {x0_eval} and xdot is {dot_x_eval}"
    )

    if output is None:

        def output_func(t, x, z, u, p, q):
            _ = t, z, u, p, q
            return x

    else:
        y_eval = output(0, x0_eval, u0, p_init)
        assert y_eval is not None, "Output function must return a value"

        def output_func(t, x, z, u, p, q):
            _ = z, q
            return output(t, x, u(t), p)

    def dynamics(t, x, z, u, p):
        _ = z
        return xdot(t, x, u(t), p)

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

    if isinstance(x0, (list, tuple)):
        x0 = np.array(x0)

    if isinstance(x0, (np.ndarray, int, float)):

        def x0_func(z, u, p):
            _ = z, u
            return x0, None

    else:

        def x0_func(z, u, p):
            _ = z, u
            return x0(p), None

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
    ) or x0_eval.shape == dot_x_eval.shape, (
        "x0 and xdot must have the same shape; "
        f"x0 is {x0_eval} and xdot is {dot_x_eval}"
    )

    if output is None:
        if is_scalar(x0_eval):

            def output_func(t, x, z, u, p, q):
                _ = t, z, u, p, q
                return x[0]

        else:

            def output_func(t, x, z, u, p, q):
                _ = t, z, u, p, q
                return x

    else:
        y_eval = output(x0_eval, p_init)
        assert y_eval is not None, "Output function must return a value"

        def output_func(t, x, z, u, p, q):
            _ = t, z, u, q
            return output(x, p)

    def dynamics(t, x, z, u, p):
        _ = t, z, u
        return xdot(x, p)

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
        return VectorSpace("composition", (dim,)) if dim else None

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
            (
                dim.to_space(f"{name}_{index}")
                if isinstance(dim, Dimension)
                else VectorSpace(f"{name}_{index}", dim) if dim else None
            )
            for index, dim in enumerate(dims)
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


class StructuredParameterProjection:
    """Partition a heterogeneous parameter tuple by subsystem."""

    def __init__(self, widths):
        self.widths = tuple(widths)

    def split(self, parameters):
        start = 0
        groups = []
        for width in self.widths:
            groups.append(tuple(parameters[start : start + width]))
            start += width
        return groups


class _NumericParameterPartition:
    def __init__(self, systems):
        self.projection = CompositionOperator(
            *[system.parameters for system in systems]
        )

    def arguments(self):
        return [self.projection.dim()]

    def parameter_space(self):
        return self.projection.dim()

    def split(self, parameters):
        return self.projection.inverse(parameters[0])

    @staticmethod
    def call_arguments(_system, values):
        return (values,)


class _StructuredParameterPartition:
    def __init__(self, systems):
        self.groups = tuple(_parameter_spaces(system) for system in systems)
        self.projection = StructuredParameterProjection(
            len(group) for group in self.groups
        )

    def arguments(self):
        return [space for group in self.groups for space in group]

    def parameter_space(self):
        return tuple(self.arguments())

    def split(self, parameters):
        return self.projection.split(parameters)

    @staticmethod
    def call_arguments(system, values):
        return values if system.parameters is not None else (None,)


def _parameter_spaces(system):
    if system.parameters is None:
        return ()
    return (
        system.parameters
        if isinstance(system.parameters, tuple)
        else (system.parameters,)
    )


def _compose_direct_sum(systems, backend, partition):
    x_dim, z_dim, q_dim = zip(
        *[system.get_state_dimensions() for system in systems]
    )
    proj_x = CompositionOperator.from_dimensions("x", *x_dim)
    proj_z = CompositionOperator.from_dimensions("z", *z_dim)
    proj_q = CompositionOperator.from_dimensions("q", *q_dim)
    proj_y = CompositionOperator.from_dimensions(
        "y", *[system.y.output_shape()[0] for system in systems]
    )
    u_range = [
        (
            system.inputs.output_dimensions()[0]
            if system.inputs is not Noop()
            else None
        )
        for system in systems
    ]
    proj_u = CompositionOperator.from_dimensions("u", *u_range)
    u_space = (
        Noop()
        if proj_u.dim() is None
        else FunctionSpace(proj_u.dim().name, [Scalar("t")], [proj_u.dim()])
    )

    def inputs(u_outer, index):
        return lambda time: proj_u.inverse(u_outer(time))[index]

    def initial(z_outer, u_outer, *p_outer):
        groups = partition.split(p_outer)
        z = proj_z.inverse(z_outer)
        values = [
            system.x0.call_inline(
                z[index],
                inputs(u_outer, index),
                *partition.call_arguments(system, groups[index]),
            )
            for index, system in enumerate(systems)
        ]
        x, z = zip(*values)
        return proj_x(*x), proj_z(*z)

    def component_call(
        component, t, x_outer, z_outer, u_outer, p_outer, projection
    ):
        groups = partition.split(p_outer)
        x, z = proj_x.inverse(x_outer), proj_z.inverse(z_outer)

        def evaluate(index, system):
            function = component(system)
            if function is Noop():
                return 0
            return function.call_inline(
                t,
                x[index],
                z[index],
                inputs(u_outer, index),
                *partition.call_arguments(system, groups[index]),
            )

        return projection(
            *[evaluate(index, system) for index, system in enumerate(systems)]
        )

    parameter_arguments = partition.arguments()
    state_arguments = [
        Scalar("t"),
        proj_x.dim(),
        proj_z.dim(),
        u_space,
        *parameter_arguments,
    ]
    x0 = function(
        [proj_z.dim(), u_space, *parameter_arguments], initial, backend=backend
    )
    dx = function(
        state_arguments,
        lambda t, x, z, u, *p: component_call(
            lambda system: system.dxdt, t, x, z, u, p, proj_x
        ),
        backend=backend,
    )
    g = (
        function(
            state_arguments,
            lambda t, x, z, u, *p: component_call(
                lambda system: system.g, t, x, z, u, p, proj_z
            ),
            backend=backend,
        )
        if proj_z.dim() is not None
        else Noop()
    )
    dqdt = (
        function(
            state_arguments,
            lambda t, x, z, u, *p: component_call(
                lambda system: system.dqdt, t, x, z, u, p, proj_q
            ),
            backend=backend,
        )
        if proj_q.dim() is not None
        else Noop()
    )

    def outputs(t, x_outer, z_outer, u_outer, *p_and_q):
        *p_outer, q_outer = p_and_q
        groups = partition.split(p_outer)
        x, z, q = (
            proj_x.inverse(x_outer),
            proj_z.inverse(z_outer),
            proj_q.inverse(q_outer),
        )
        return proj_y(
            *[
                system.y.call_inline(
                    t,
                    x[index],
                    z[index],
                    inputs(u_outer, index),
                    *partition.call_arguments(system, groups[index]),
                    q[index],
                )
                for index, system in enumerate(systems)
            ]
        )

    y = function([*state_arguments, proj_q.dim()], outputs, backend=backend)
    parameters = partition.parameter_space()
    system = DynamicalSystem(u_space, parameters, x0, dx, g, dqdt, y)
    projections = ProjectionSet(
        partition.projection, proj_x, proj_z, proj_q, proj_y, proj_u
    )
    return system, projections


def direct_sum(
    *systems: DynamicalSystem, backend=None
) -> Tuple[DynamicalSystem, ProjectionSet]:
    backend = backend or systems[0].backend()
    partition = (
        _StructuredParameterPartition(systems)
        if any(
            isinstance(space, FunctionSpace)
            for system in systems
            for space in _parameter_spaces(system)
        )
        else _NumericParameterPartition(systems)
    )
    return _compose_direct_sum(systems, backend, partition)
