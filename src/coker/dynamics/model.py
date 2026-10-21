from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, TypeAlias

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
    """Describe callbacks for an explicit or semi-explicit dynamic system.

    Trace with :func:`~coker.dynamics.create_dynamics_from_spec`. Missing
    inputs, constraints and quadratures accept ``None`` or ``Noop()``;
    absent algebraic and parameter spaces use ``None``.

    Args:
        inputs: A time-to-control ``FunctionSpace``, or no input.
        parameters: One scalar/vector/function declaration, a positional
            tuple or list of declarations, or ``None``. Tuple declarations
            reach callbacks as one final ``p`` tuple; a single declaration
            is passed directly.
        algebraic: Algebraic variable space, or ``None``.
        initial_conditions: Callback ``(z, u, p) -> (x0, z0)``. Its first
            argument is the algebraic slot, not time. Return ``None`` for
            absent algebraic initial values.
        dynamics: Callback ``(t, x, z, u, p) -> dxdt``.
        constraints: Callback ``(t, x, z, u, p) -> g`` for ``g = 0``.
        outputs: Callback ``(t, x, z, u, p, q) -> y``.
        quadratures: Callback ``(t, x, z, u, p) -> dqdt``. Convert to a
            residual system for direct evaluation of this channel.

    Input callbacks receive the control function, not its value; evaluate
    ``u(t)`` where needed. State and derivative shapes must match.

    Examples:
        Trace a specification after defining its callbacks::

            system = create_dynamics_from_spec(spec, backend="numpy")
    """

    inputs: FunctionSpace | Noop | None
    parameters: DynamicsParameters
    algebraic: Optional[VectorSpace]
    initial_conditions: Callable
    """``(x0, z0) = initial_conditions(z, u, p)``."""

    dynamics: Callable
    """dx = dynamics(t, x, z, u, p)"""

    constraints: Callable | Noop | None
    """g(t, x, z, u, p) = 0."""

    outputs: Callable
    """y(t) = outputs(t, x, z, u, p, q)"""

    quadratures: Callable | Noop | None
    """dq/dt = quadratures(t, x, z, u, p)"""

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = Noop()
        if self.constraints is None:
            self.constraints = Noop()
        if self.quadratures is None:
            self.quadratures = Noop()
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


def _output_function_space(
    output: Function,
    inputs: FunctionSpace | Noop,
    parameters: DynamicsParameters,
) -> FunctionSpace:
    """Build the public output space from one system's output callback."""
    t, _x, _z, u, *parameter_shapes, _q = output.input_shape()
    (out,) = output.output_shape()
    args = [t.to_space("t")]
    if not isinstance(inputs, Noop):
        args.append(u if isinstance(u, FunctionSpace) else u.to_space("u"))
    if isinstance(parameters, (tuple, list)):
        for index, (element, shape) in enumerate(zip(parameters, parameter_shapes)):
            args.append(
                element
                if isinstance(shape, FunctionSpace)
                else shape.to_space(f"p{index}")
            )
    elif parameters is not None:
        (parameter_shape,) = parameter_shapes
        args.append(parameter_shape.to_space("p"))
    return FunctionSpace("y", args, [out.to_space("y")])


@dataclass
class DynamicalSystem:
    """Hold traced dynamics, initial conditions and observed outputs.

    Prefer the constructors in :mod:`coker.dynamics` over assembling
    component functions manually. Construction and numerical integration
    are separate: the function backend selects direct evaluation, while a
    variational problem independently selects its solver backend.

    Args:
        inputs: Control function space, or ``None``/``Noop()`` when absent.
        parameters: Ordered parameter declarations, or ``None``.
        x0: Function ``(z, u, *p) -> (x0, z0)``.
        dxdt: Function ``(t, x, z, u, *p) -> dxdt``.
        g: Algebraic residual function, or ``None``/``Noop()``.
        dqdt: Quadrature derivative function, or ``None``/``Noop()``.
        y: Observed output function ``(t, x, z, u, *p, q) -> y``.
        solver_parameters: Backend-specific direct integration settings.

    Examples:
        Create an explicit system with the autonomous constructor::

            system = create_autonomous_ode([1.0], lambda x, p: -x)
    """

    inputs: FunctionSpace | Noop | None
    parameters: DynamicsParameters
    x0: Function
    dxdt: Function
    g: Function | Noop | None
    dqdt: Function | Noop | None
    y: Function
    solver_parameters: object | None = field(default=None)

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = Noop()
        if self.g is None:
            self.g = Noop()
        if self.dqdt is None:
            self.dqdt = Noop()
        if isinstance(self.parameters, list):
            self.parameters = tuple(self.parameters)

    def output_as_function_space(self) -> FunctionSpace:
        """Return the callable space of the public trajectory output."""
        return _output_function_space(self.y, self.inputs, self.parameters)

    def get_state_dimensions(self) -> Tuple[Dimension, Dimension, Dimension]:
        """Return differential, algebraic and quadrature dimensions.

        Returns:
            An ``(x, z, q)`` tuple of dimensions; absent channels are ``None``.
        """
        shapes = self.y.input_shape()
        return shapes[1], shapes[2], shapes[-1]

    @property
    def backend(self) -> str:
        """Return the backend name of the differential-rate function."""
        return self.dxdt.backend

    def __call__(self, *args):
        """Integrate from time zero and return the declared output.

        Args:
            *args: ``(time, *parameters)`` without inputs, or
                ``(time, input_function, *parameters)`` with inputs.
                ``time`` is a numeric endpoint or a one-dimensional NumPy
                sampling array. Parameters follow declaration order;
                function-valued slots accept compatible Coker functions,
                fitted functions or traceable Python callables.

        Returns:
            The output at an endpoint in its declared shape. With NumPy,
            sampled vector outputs have shape ``(n_times,)`` for a declared
            shape of ``(1,)``, or ``(n_times, n_output)`` for larger vectors.
            NumPy grid evaluation of genuinely scalar outputs currently raises
            ValueError; stack scalar-time evaluations with ``numpy.stack``
            instead.

        Raises:
            ValueError: Argument count or function parameter shape/backend
                does not match the declaration, or NumPy grid sampling is
                requested for a scalar output.
            TypeError: A function-valued parameter is not callable.
            NotImplementedError: Quadratures are present, or the selected
                backend does not support the required integration. Convert
                quadrature systems with ``to_residual_dynamical_system``.
        """
        if not isinstance(self.dqdt, Noop):
            raise NotImplementedError
        from coker.dynamics.residual import to_residual_dynamical_system

        return to_residual_dynamical_system(self)(*args)
