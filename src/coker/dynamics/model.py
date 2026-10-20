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
    """Describe a semi-explicit dynamic system before it is traced.

    Args:
        inputs: Optional input trajectory declaration; ``None`` means no input.
        parameters: Optional scalar, vector, or function parameter declarations.
        algebraic: Optional algebraic-state declaration.
        initial_conditions: Callable returning ``(x0, z0)``.
        dynamics: Callable computing ``dxdt(t, x, z, u, p)``.
        constraints: Callable computing ``g(t, x, z, u, p)``.
        outputs: Callable computing ``y(t, x, z, u, p, q)``.
        quadratures: Callable computing ``dqdt(t, x, z, u, p)``.

    Examples:
        Build a system with :func:`create_dynamics_from_spec`::

            spec = DynamicsSpec(...)
    """

    inputs: FunctionSpace | Noop | None
    parameters: DynamicsParameters
    algebraic: Optional[VectorSpace]
    initial_conditions: Callable
    """[x, z] = initial_conditions(z, u, p)."""

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
    """Define an explicit or semi-explicit system and its output.

    Args:
        inputs: Optional input trajectory declaration; ``None`` means no input.
        parameters: Optional model parameter declarations.
        x0: Initial-condition function.
        dxdt: Differential-rate function.
        g: Optional algebraic residual function.
        dqdt: Optional quadrature-rate function.
        y: Output function.
        solver_parameters: Backend-specific integration settings.

    Examples:
        Create a simple explicit system with
        :func:`~coker.dynamics.create_autonomous_ode`::

            system = create_autonomous_ode(x0, xdot)
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
        """Return the differential, algebraic, and quadrature dimensions."""
        shapes = self.y.input_shape()
        return shapes[1], shapes[2], shapes[-1]

    @property
    def backend(self) -> str:
        """Return the backend used by the differential-rate callback."""
        return self.dxdt.backend

    def __call__(self, *args):
        """Evaluate a non-quadrature trajectory at time or a time grid."""
        if not isinstance(self.dqdt, Noop):
            raise NotImplementedError
        from coker.dynamics.residual import to_residual_dynamical_system

        return to_residual_dynamical_system(self)(*args)
