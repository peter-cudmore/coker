from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, TypeAlias

import numpy as np
from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.function import Function, function
from coker.algebra.ops import Noop
from coker.dynamics.function_parameters import FittedFunction


def _same_function_shape(
    actual: Dimension | FunctionSpace | None,
    expected: Dimension | FunctionSpace | None,
) -> bool:
    if isinstance(actual, Dimension) or isinstance(expected, Dimension):
        return (
            isinstance(actual, Dimension)
            and isinstance(expected, Dimension)
            and actual == expected
        )
    if isinstance(actual, FunctionSpace) and isinstance(
        expected, FunctionSpace
    ):
        return _matches_function_signature(
            actual.input_dimensions(),
            actual.output_dimensions(),
            expected,
        )
    return actual is expected


def _matches_function_signature(
    actual_inputs, actual_outputs, declaration: FunctionSpace
) -> bool:
    expected_inputs = tuple(declaration.input_dimensions())
    expected_outputs = tuple(declaration.output_dimensions())
    return (
        len(actual_inputs) == len(expected_inputs)
        and len(actual_outputs) == len(expected_outputs)
        and all(
            _same_function_shape(actual, expected)
            for actual, expected in zip(actual_inputs, expected_inputs)
        )
        and all(
            _same_function_shape(actual, expected)
            for actual, expected in zip(actual_outputs, expected_outputs)
        )
    )


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


@dataclass
class DynamicalSystem:
    inputs: FunctionSpace
    parameters: ParameterDeclaration | tuple[ParameterDeclaration, ...] | None
    x0: Function
    dxdt: Function
    g: Optional[Function]
    dqdt: Optional[Function]
    y: Function
    solver_parameters: Optional[object] = field(default=None)

    def get_state_dimensions(self) -> Tuple[Dimension, Dimension, Dimension]:
        shapes = self.y.input_shape()
        return shapes[1], shapes[2], shapes[-1]

    def backend(self):
        return self.dxdt.backend

    @staticmethod
    def _matches_function_space(
        value: Function, declaration: FunctionSpace
    ) -> bool:
        return _matches_function_signature(
            value.input_shape(), value.output_shape(), declaration
        )

    def _prepare_direct_parameter(
        self, declaration: ParameterDeclaration, value, index: int
    ):
        if not isinstance(declaration, FunctionSpace):
            return value

        if isinstance(value, FittedFunction) and not (
            _matches_function_signature(
                value.space.input_dimensions(),
                value.space.output_dimensions(),
                declaration,
            )
        ):
            raise ValueError(
                f"Function-valued parameter {index} does not match declared "
                f"FunctionSpace {declaration.name!r}"
            )
        if isinstance(value, Function) and not self._matches_function_space(
            value, declaration
        ):
            raise ValueError(
                f"Function-valued parameter {index} does not match declared "
                f"FunctionSpace {declaration.name!r}"
            )
        if not callable(value):
            raise TypeError(
                f"Function-valued parameter {index} must be a callable, "
                "Coker Function, or FittedFunction"
            )
        prepared = function(
            declaration.arguments, value, backend=self.backend()
        )

        if not self._matches_function_space(prepared, declaration):
            raise ValueError(
                f"Function-valued parameter {index} does not match declared "
                f"FunctionSpace {declaration.name!r}"
            )
        return prepared

    def _prepare_direct_parameters(self, value):
        if isinstance(self.parameters, tuple):
            return tuple(
                self._prepare_direct_parameter(declaration, parameter, index)
                for index, (declaration, parameter) in enumerate(
                    zip(self.parameters, value)
                )
            )
        if self.parameters is None:
            return None
        return self._prepare_direct_parameter(self.parameters, value, 0)

    def _map_arguments(self, *args):
        if not args:
            raise ValueError(
                "A trajectory evaluation requires a time argument"
            )

        has_inputs = self.inputs is not Noop()
        parameter_count = (
            len(self.parameters)
            if isinstance(self.parameters, tuple)
            else int(self.parameters is not None)
        )
        expected_count = 1 + int(has_inputs) + parameter_count
        if len(args) != expected_count:
            raise ValueError(
                "Trajectory argument count does not match the system "
                f"declaration: expected {expected_count}, got {len(args)}"
            )

        t = args[0]
        next_argument = 1
        if has_inputs:
            u = args[next_argument]
            next_argument += 1
        else:
            u = None

        values = args[next_argument:]
        if isinstance(self.parameters, tuple):
            p = tuple(values)
        elif self.parameters is None:
            p = None
        else:
            (p,) = values
        return t, u, self._prepare_direct_parameters(p)

    def __call__(self, *args):
        from coker.backends import get_backend_by_name

        t, u, p = self._map_arguments(*args)
        parameter_arguments = p if isinstance(self.parameters, tuple) else (p,)
        x0, z0 = self.x0(0, u, *parameter_arguments)

        if self.dqdt is not Noop():
            raise NotImplementedError
        q0 = None

        backend = get_backend_by_name(self.dxdt.backend)
        x, z, q = backend.evaluate_integrals(
            [self.dxdt, self.g, self.dqdt],
            [x0, z0, q0],
            t,
            [u, *parameter_arguments],
            solver_parameters=self.solver_parameters,
        )

        if isinstance(t, (float, int)):
            return self.y(t, x, z, u, *parameter_arguments, q)

        def map_args(index):
            x_i = x[:, index]
            z_i = z[:, index] if z is not None else None
            q_i = q[:, index] if q is not None else None
            return x_i, z_i, u, *parameter_arguments, q_i

        if self.y.output_shape()[0].is_scalar() or self.y.output_shape()[
            0
        ].dim == (1,):
            return np.concatenate(
                [self.y(t_i, *map_args(index)) for index, t_i in enumerate(t)]
            )
        return np.vstack(
            [self.y(t_i, *map_args(index)) for index, t_i in enumerate(t)]
        )

    def output_as_function_space(self) -> FunctionSpace:
        t, _x, _z, u, *parameter_shapes, _q = self.y.input_shape()
        (out,) = self.y.output_shape()
        args = [t.to_space("t")]
        if self.inputs is not Noop():
            args.append(u if isinstance(u, FunctionSpace) else u.to_space("u"))
        if isinstance(self.parameters, tuple):
            for index, (element, shape) in enumerate(
                zip(self.parameters, parameter_shapes)
            ):
                args.append(
                    element
                    if isinstance(shape, FunctionSpace)
                    else shape.to_space(f"p{index}")
                )
        elif self.parameters is not None:
            (parameter_shape,) = parameter_shapes
            args.append(parameter_shape.to_space("p"))
        return FunctionSpace("y", args, [out.to_space("y")])
