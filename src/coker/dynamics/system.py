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


def create_dynamics_from_spec(spec: DynamicsSpec, backend="numpy") -> DynamicalSystem:
    """Trace a semi-explicit system specification.

    The callbacks use ``(t, x, z, u, p)`` for dynamics, constraints and
    quadrature rates, ``(z, u, p)`` for initial conditions and
    ``(t, x, z, u, p, q)`` for outputs. Initial conditions return
    ``(x0, z0)``. Absent algebraic variables and quadratures use ``None``.
    With a tuple of parameter declarations, callbacks receive the parameter
    values as a tuple in ``p``; a single declaration receives its bare value.

    Args:
        spec: Initial conditions, declarations and traceable callbacks.
            Inputs are trajectories, so callbacks evaluate them as ``u(t)``.
        backend: Registered backend used to trace and evaluate all callbacks.

    Returns:
        A dynamical system holding the compiled callbacks, not a simulation.

    Raises:
        TypeError: A supplied callback is not callable.
        ValueError: Initial conditions do not return a pair, algebraic initial
            dimensions differ from the declaration, dynamics do not return one
            value with the state shape, or quadratures return multiple values.
            Also raised when constraints and algebraic variables are not
            configured together.

    Examples:
        >>> import numpy as np
        >>> from coker.dynamics import DynamicsSpec, create_dynamics_from_spec
        >>> spec = DynamicsSpec(
        ...     inputs=None, parameters=None, algebraic=None,
        ...     initial_conditions=lambda z, u, p: (np.ones(2), None),
        ...     dynamics=lambda t, x, z, u, p: -x,
        ...     constraints=None, quadratures=None,
        ...     outputs=lambda t, x, z, u, p, q: x,
        ... )
        >>> system = create_dynamics_from_spec(spec)
        >>> system.dxdt(0.0, np.ones(2), None, None, None).tolist()
        [-1.0, -1.0]
    """
    for name, callback in (
        ("initial_conditions", spec.initial_conditions),
        ("dynamics", spec.dynamics),
        ("constraints", spec.constraints),
        ("quadratures", spec.quadratures),
        ("outputs", spec.outputs),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    heterogeneous = isinstance(spec.parameters, tuple)
    parameter_arguments = list(spec.parameters) if heterogeneous else [spec.parameters]
    x0 = function(
        arguments=[spec.algebraic, spec.inputs, *parameter_arguments],
        implementation=_parameter_callback(
            spec.initial_conditions, heterogeneous, _initial_parameters
        ),
        backend=backend,
    )
    if len(x0.output) != 2:
        raise ValueError(
            "Initial conditions must be a pair, one for the state and one "
            "for the algebraic variables"
        )
    state, algebraic = x0.output
    state_space = VectorSpace("x", state.dim.flat())
    if algebraic is not None:
        if spec.algebraic is None or algebraic.dim != Dimension(
            spec.algebraic.dimension
        ):
            raise ValueError(
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
        _parameter_callback(spec.dynamics, heterogeneous, _dynamics_parameters),
        backend,
    )
    if len(xdot.output) != 1:
        raise ValueError("Dynamics must return a single vector")
    if xdot.output[0] is None or xdot.output[0].dim.shape != state.dim.shape:
        raise ValueError(
            "Dynamics must return a vector of the same dimension as the "
            f"state: x0 gave {state.dim} and dynamics gave {xdot.output_shape()}"
        )
    if spec.algebraic is not None:
        if spec.constraints is Noop():
            raise ValueError(
                "If algebraic variables are specified, constraints must "
                "also be specified"
            )
    elif spec.constraints is not Noop():
        raise ValueError("Constraints specified, but no algebraic variables")
    constraint = (
        function(
            arguments,
            _parameter_callback(spec.constraints, heterogeneous, _dynamics_parameters),
            backend,
        )
        if spec.algebraic is not None
        else Noop()
    )
    quadrature = (
        function(
            arguments,
            _parameter_callback(spec.quadratures, heterogeneous, _dynamics_parameters),
            backend,
        )
        if spec.quadratures is not Noop()
        else Noop()
    )
    if quadrature is not Noop():
        if len(quadrature.output) != 1:
            raise ValueError("Quadratures must be a scalar or vector space")
        q = quadrature.output[0]
        arguments.append(
            VectorSpace("q", q.dim.flat()) if not q.dim.is_scalar() else Scalar("q")
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
    output: Callable[[Scalar, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = None,
    backend: str = "numpy",
    p_init: np.ndarray = None,
    u_init: Callable[[float], np.ndarray] = None,
) -> DynamicalSystem:
    """Construct a controlled ODE without algebraic states or quadratures.

    Callbacks receive the control's value, rather than its trajectory.
    Construction evaluates callbacks at time zero before tracing them.

    Args:
        x0: Fixed initial array, list, tuple or number, or callable ``x0(p)``.
        xdot: Traceable callback ``xdot(t, x, u, p)`` returning the state rate.
        control: Time-dependent function space with exactly one output.
        parameters: Optional scalar or vector parameter declaration.
        output: Optional callback ``output(t, x, u, p)``. Defaults to the state.
        backend: Registered evaluation backend; defaults to NumPy.
        p_init: Parameter values for construction-time checks, not stored
            simulation parameters. Vector parameters default to zeros.
            Supply this explicitly for scalar parameters, as a number or
            shape ``(1,)`` array. Vector parameters require an array matching
            their dimension.
        u_init: Optional callable evaluated as ``u_init(0)`` for the
            construction-time control value. Defaults to a zero array.

    Returns:
        A dynamical system containing compiled initial, rate and output maps.

    Raises:
        TypeError: A supplied callback, including ``u_init``, is not callable,
            or ``x0`` is neither a supported initial value nor callable.
        ValueError: ``p_init`` lacks a parameter declaration or has the wrong
            shape, control does not have exactly one output, ``u_init(0)`` has
            the wrong shape, the initial state and rate shapes differ, or
            output is absent.

    Examples:
        >>> import numpy as np
        >>> from coker import FunctionSpace, Scalar, VectorSpace
        >>> from coker.dynamics import create_control_system
        >>> control = FunctionSpace("u", [Scalar("t")], [VectorSpace("u", 1)])
        >>> system = create_control_system(
        ...     [0.0], lambda t, x, u, p: u - x, control,
        ... )
        >>> system.dxdt(
        ...     0.0, np.array([2.0]), None, lambda t: np.array([3.0]), None
        ... ).tolist()
        [1.0]
    """

    if not isinstance(x0, (np.ndarray, list, tuple, int, float)) and not callable(x0):
        raise TypeError("x0 must be an initial value or callable")
    if not callable(xdot):
        raise TypeError("xdot must be callable")
    if output is not None and not callable(output):
        raise TypeError("output must be callable")

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
        if parameters is None:
            raise ValueError("Parameters must be specified if p_init is specified")
        if isinstance(parameters, Scalar):
            if not (isinstance(p_init, (float, int)) or np.shape(p_init) == (1,)):
                raise ValueError("p_init must be a scalar or a 1D array")
            p_init = np.array([p_init])
        else:
            if np.shape(p_init) != (parameters.dimension,):
                raise ValueError(
                    "p_init must be a 1D array of the same size as the parameters"
                )
    else:
        p_init = None if parameters is None else np.zeros(parameters.dimension)

    if len(control.output) != 1:
        raise ValueError("Control must return a single vector")
    (u_dim,) = control.output_dimensions()
    if u_init is not None:
        if not callable(u_init):
            raise TypeError("u_init must be a callable")
        u0 = u_init(0)
        if np.shape(u0) != u_dim.shape:
            raise ValueError(
                "u0 must have the same shape as the control output; "
                f"u0 is {u0} and control output is {u_dim}"
            )
    else:
        u0 = np.zeros(u_dim.shape)

    x0_eval, _z0 = x0_func(None, u0, p_init)
    dot_x_eval = xdot(0, x0_eval, u0, p_init)

    if not (
        (is_scalar(x0_eval) and is_scalar(dot_x_eval))
        or np.shape(x0_eval) == np.shape(dot_x_eval)
    ):
        raise ValueError(
            "x0 and xdot must have the same shape; "
            f"x0 is {x0_eval} and xdot is {dot_x_eval}"
        )

    if output is None:

        def output_func(t, x, z, u, p, q):
            _ = t, z, u, p, q
            return x

    else:
        y_eval = output(0, x0_eval, u0, p_init)
        if y_eval is None:
            raise ValueError("Output function must return a value")

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
    """Construct an autonomous ODE without controls or algebraic variables.

    Construction checks callbacks at the initial state before tracing them.
    The returned system contains functions; it does not integrate the ODE.

    Args:
        x0: Fixed initial array, list, tuple or number, or callable ``x0(p)``.
        xdot: Traceable callback ``xdot(x, p)`` returning the state rate.
        parameters: Optional scalar or vector parameter declaration.
        output: Optional callback ``output(x, p)``; defaults to the state.
        backend: Registered evaluation backend; defaults to NumPy.
        p_init: Parameter values used only for construction-time checks.
            Vector parameters default to zeros. Supply this explicitly for
            scalar parameters, as a number or shape ``(1,)`` array.
            Vector parameters require an array matching their dimension.

    Returns:
        A dynamical system with no controls, constraints or quadratures.
        Its compiled state is a flat vector, even for scalar initial values;
        the default output preserves a scalar initial value as a scalar.

    Raises:
        TypeError: A supplied callback is not callable, or ``x0`` is neither
            a supported initial value nor callable.
        ValueError: ``p_init`` lacks a parameter declaration or has the wrong
            shape, initial state and rate shapes differ, or output is absent.

    Examples:
        >>> import numpy as np
        >>> from coker.dynamics import create_autonomous_ode
        >>> system = create_autonomous_ode([2.0], lambda x, p: -x)
        >>> system.dxdt(0.0, np.array([2.0]), None, None, None).tolist()
        [-2.0]
    """

    if not isinstance(x0, (np.ndarray, list, tuple, int, float)) and not callable(x0):
        raise TypeError("x0 must be an initial value or callable")
    if not callable(xdot):
        raise TypeError("xdot must be callable")
    if output is not None and not callable(output):
        raise TypeError("output must be callable")

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
        if parameters is None:
            raise ValueError("Parameters must be specified if p_init is specified")
        if isinstance(parameters, Scalar):
            if not (isinstance(p_init, (float, int)) or np.shape(p_init) == (1,)):
                raise ValueError("p_init must be a scalar or a 1D array")
            p_init = np.array([p_init])
        else:
            if np.shape(p_init) != (parameters.dimension,):
                raise ValueError(
                    "p_init must be a 1D array of the same size as the parameters"
                )
    else:
        p_init = None if parameters is None else np.zeros(parameters.dimension)

    x0_eval, _z0 = x0_func(None, None, p_init)
    dot_x_eval = xdot(x0_eval, p_init)

    if not (
        (is_scalar(x0_eval) and is_scalar(dot_x_eval))
        or np.shape(x0_eval) == np.shape(dot_x_eval)
    ):
        raise ValueError(
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
        if y_eval is None:
            raise ValueError("Output function must return a value")

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
    """Flatten numeric subsystem values and recover their component slices.

    Args:
        *spaces: Subsystem scalar/vector spaces, with ``None`` for absent slots.

    Examples:
        >>> from coker import VectorSpace
        >>> from coker.dynamics import CompositionOperator
        >>> projection = CompositionOperator(VectorSpace("x", 2), None)
        >>> projection([1.0, 2.0], None).tolist()
        [1.0, 2.0]
        >>> [None if value is None else value.tolist()
        ...  for value in projection.inverse(projection([1.0, 2.0], None))]
        [[1.0, 2.0], None]
    """

    def __init__(self, *spaces: Scalar | VectorSpace | None):
        self.spaces = spaces

    def offsets(self):
        """Yield each subsystem's starting offset in the flat vector.

        Yields:
            Integer offsets; absent slots consume no entries.
        """
        total = 0
        for space in self.spaces:
            yield total
            if space is not None:
                total += space.size

    def sizes(self):
        """Yield the flattened size of each subsystem.

        Yields:
            Integer sizes, with zero for absent slots.
        """
        for space in self.spaces:
            if space is None:
                yield 0
            else:
                yield space.size

    def dim(self) -> Scalar | VectorSpace | None:
        """Return the combined vector space, or ``None`` when empty.

        Returns:
            A vector space named ``composition`` with the total flat size.
        """

        dim = sum(self.sizes())
        return VectorSpace("composition", (dim,)) if dim else None

    def inverse(self, ab) -> List:
        """Split a flat vector into subsystem slices.

        Args:
            ab: Combined numeric or symbolic vector.

        Returns:
            List of flat slices in subsystem order, preserving absent slots
            as ``None``. Original multidimensional shapes are not restored.
        """
        next_slice = ab
        output = []
        for space in self.spaces:
            if space is None:
                output.append(None)
            else:
                dim = space.size
                output.append(next_slice[:dim])
                next_slice = next_slice[dim:] if dim < next_slice.shape[0] else []
        return output

    def __call__(self, *args):
        """Concatenate the flattened values of present subsystem slots.

        Args:
            *args: One value per declared space, including absent slots.

        Returns:
            Flat NumPy array, or ``None`` if every space is absent.
        """
        output = [
            np.reshape(a, (space.size,))
            for a, space in zip(args, self.spaces)
            if space is not None
        ]
        if not output:
            return None

        return np.concatenate(output)

    @staticmethod
    def from_dimensions(name: str, *dims) -> "CompositionOperator":
        """Build subsystem spaces from dimensions or shape declarations.

        Args:
            name: Prefix for generated subsystem names.
            *dims: Dimensions, vector shapes, or absent ``None`` slots.

        Returns:
            A composition operator over the corresponding spaces.
        """
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
        """Return matrices that select each subsystem from the combined vector.

        Returns:
            List of arrays with shape ``(subsystem_size, total_size)``.
            Absent slots have zero rows.
        """
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
    """Group the projection operators returned by :func:`direct_sum`.

    Args:
        parameters: Numeric concatenation or structured parameter partition.
        state: Differential-state projection.
        algebraic: Algebraic-state projection.
        quadratures: Quadrature-state projection.
        outputs: Output projection.
        controls: Control-value projection.

    Examples:
        >>> from coker.dynamics import create_autonomous_ode, direct_sum
        >>> left = create_autonomous_ode([1.0], lambda x, p: -x)
        >>> right = create_autonomous_ode([2.0], lambda x, p: -x)
        >>> combined, projections = direct_sum(left, right)
        >>> list(projections.state.sizes())
        [1, 1]
    """

    parameters: "CompositionOperator | StructuredParameterProjection"
    state: CompositionOperator
    algebraic: CompositionOperator
    quadratures: CompositionOperator
    outputs: CompositionOperator
    controls: CompositionOperator


class StructuredParameterProjection:
    """Partition heterogeneous parameter values by subsystem.

    Args:
        widths: Number of parameter declarations in each subsystem.

    Examples:
        >>> from coker.dynamics import StructuredParameterProjection
        >>> projection = StructuredParameterProjection([1, 2])
        >>> projection.split((1.0, 2.0, 3.0))
        [(1.0,), (2.0, 3.0)]
    """

    def __init__(self, widths):
        self.widths = tuple(widths)

    def split(self, parameters):
        """Split ordered parameter values into subsystem tuples.

        Args:
            parameters: Flat tuple of numeric or function-valued parameters.

        Returns:
            List of parameter tuples in subsystem order.
        """
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
    x_dim, z_dim, q_dim = zip(*[system.get_state_dimensions() for system in systems])
    proj_x = CompositionOperator.from_dimensions("x", *x_dim)
    proj_z = CompositionOperator.from_dimensions("z", *z_dim)
    proj_q = CompositionOperator.from_dimensions("q", *q_dim)
    proj_y = CompositionOperator.from_dimensions(
        "y", *[system.y.output_shape()[0] for system in systems]
    )
    u_range = [
        (system.inputs.output_dimensions()[0] if system.inputs is not Noop() else None)
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

    def component_call(component, t, x_outer, z_outer, u_outer, p_outer, projection):
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
    """Compose independent systems and return their projection operators.

    Numeric parameters use :class:`CompositionOperator`; heterogeneous
    function-valued parameters use :class:`StructuredParameterProjection`.
    The remaining projections are numeric composition operators.

    Args:
        *systems: One or more independent dynamical systems.
        backend: Evaluation backend, defaulting to the first system's backend.

    Returns:
        Pair ``(system, projections)`` containing the combined system and
        operators for concatenating and splitting subsystem values.

    Examples:
        >>> from coker.dynamics import create_autonomous_ode, direct_sum
        >>> left = create_autonomous_ode([1.0], lambda x, p: -x)
        >>> right = create_autonomous_ode([2.0], lambda x, p: -2 * x)
        >>> combined, projections = direct_sum(left, right)
        >>> list(projections.state.sizes())
        [1, 1]
    """
    backend = backend or systems[0].backend
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
