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
    inputs: FunctionSpace | Noop
    parameters: DynamicsParameters
    algebraic: Optional[VectorSpace]

    initial_conditions: Callable
    """ [x, z] = initial_conditions(t_0, p) """

    dynamics: Callable
    """dx = dynamics(t, x, z, u, p)"""

    constraints: Callable
    """g(t, x, z, u, p) = 0."""

    outputs: Callable
    """y(t) = outputs(t, x, z, u, p, q)"""

    quadratures: Callable
    """dq/dt = quadratures(t, x, z, u, p)"""

    def __post_init__(self):
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
    inputs: FunctionSpace | Noop
    parameters: DynamicsParameters
    x0: Function
    dxdt: Function
    g: Optional[Function]
    dqdt: Optional[Function]
    y: Function
    solver_parameters: object | None = field(default=None)

    def output_as_function_space(self) -> FunctionSpace:
        return _output_function_space(self.y, self.inputs, self.parameters)

    def get_state_dimensions(self) -> Tuple[Dimension, Dimension, Dimension]:
        shapes = self.y.input_shape()
        return shapes[1], shapes[2], shapes[-1]

    def backend(self) -> str:
        return self.dxdt.backend

    def __call__(self, *args):
        if not isinstance(self.dqdt, Noop):
            raise NotImplementedError
        from coker.dynamics.residual import to_residual_dynamical_system

        return to_residual_dynamical_system(self)(*args)
