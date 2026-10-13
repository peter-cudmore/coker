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
from coker.parameters.function_parameters import FittedFunction


ParameterDeclaration: TypeAlias = Scalar | VectorSpace | FunctionSpace
DynamicsParameters: TypeAlias = (
    ParameterDeclaration
    | tuple[ParameterDeclaration, ...]
    | list[ParameterDeclaration]
    | None
)


@dataclass
class DynamicsSpec:
    """Describe callbacks for a differential/algebraic system.

    Construct with ``coker.dynamics.system.create_dynamics_from_spec``.
    Missing inputs, constraints, and quadratures use ``Noop()`` from
    ``coker.algebra.ops``; absent algebraic and parameter spaces use ``None``.

    Args:
        inputs: A time-to-control ``FunctionSpace``, or ``Noop()``.
        parameters: One scalar/vector/function declaration, a positional
            tuple or list of declarations, or ``None``. Tuple declarations
            are passed to callbacks as one final ``p`` tuple; a single
            declaration is passed directly.
        algebraic: Algebraic variable space, or ``None``.
        initial_conditions: Callback ``(z, u, p) -> (x0, z0)``. Its first
            argument is the algebraic slot, not time. Return ``None`` for
            absent algebraic initial values.
        dynamics: Callback ``(t, x, z, u, p) -> dxdt``.
        constraints: Callback ``(t, x, z, u, p) -> g`` for ``g = 0``.
        outputs: Callback ``(t, x, z, u, p, q) -> y``.
        quadratures: Callback ``(t, x, z, u, p) -> dqdt``. Direct
            ``DynamicalSystem`` evaluation does not support this channel.

    Input callbacks receive the control function, not its value; evaluate
    ``u(t)`` where needed. State and derivative shapes must match.
    """

    inputs: FunctionSpace | Noop
    parameters: DynamicsParameters
    algebraic: Optional[VectorSpace]

    initial_conditions: Callable
    """``(x0, z0) = initial_conditions(z, u, p)``."""

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
    """Hold traced dynamics, initial conditions, and observed outputs.

    Prefer the constructors in ``coker.dynamics.system`` over assembling
    component functions manually. Construction and numerical integration
    are separate: the function backend selects direct evaluation, while a
    variational problem independently selects its solver backend.

    Attributes:
        inputs: Control function space, or ``Noop()`` when absent.
        parameters: Ordered parameter declarations, or ``None``.
        x0: Function ``(z, u, *p) -> (x0, z0)``.
        dxdt: Function ``(t, x, z, u, *p) -> dxdt``.
        g: Algebraic residual function, or ``Noop()``.
        dqdt: Quadrature derivative function, or ``Noop()``.
        y: Observed output function ``(t, x, z, u, *p, q) -> y``.
        solver_parameters: Backend-specific direct integration settings.
    """

    inputs: FunctionSpace
    parameters: ParameterDeclaration | tuple[ParameterDeclaration, ...] | None
    x0: Function
    dxdt: Function
    g: Optional[Function]
    dqdt: Optional[Function]
    y: Function
    solver_parameters: Optional[object] = field(default=None)

    def get_state_dimensions(self) -> Tuple[Dimension, Dimension, Dimension]:
        """Return differential, algebraic, and quadrature dimensions.

        Returns:
            An ``(x, z, q)`` tuple of dimensions; absent channels are ``None``.
        """
        shapes = self.y.input_shape()
        return shapes[1], shapes[2], shapes[-1]

    def backend(self):
        """Return the backend name of the differential dynamics function."""
        return self.dxdt.backend

    def _prepare_backend_value(self, backend, value):
        return (
            value
            if value is None or callable(value)
            else backend.to_backend_array(value)
        )

    def _prepare_direct_parameter(
        self, declaration: ParameterDeclaration, value, index: int
    ) -> object:
        if not isinstance(declaration, FunctionSpace):
            return value

        if isinstance(value, Function):
            if value.backend != self.backend():
                raise ValueError(
                    f"Function-valued parameter {index} uses backend "
                    f"{value.backend!r}, expected {self.backend()!r}"
                )
            if value not in declaration:
                raise ValueError(
                    f"Function-valued parameter {index} does not match "
                    f"declared FunctionSpace {declaration.name!r}"
                )
            return value

        if isinstance(value, FittedFunction) and value not in declaration:
            raise ValueError(
                f"Function-valued parameter {index} does not match declared "
                f"FunctionSpace {declaration.name!r}"
            )
        if not callable(value):
            raise TypeError(
                f"Function-valued parameter {index} must be callable"
            )

        prepared = function(
            declaration.arguments, value, backend=self.backend()
        )
        if prepared not in declaration:
            raise ValueError(
                f"Function-valued parameter {index} does not match declared "
                f"FunctionSpace {declaration.name!r}"
            )
        return prepared

    def _map_arguments(
        self, *args
    ) -> tuple[object, object, tuple[object, ...]]:
        declarations = (
            ()
            if self.parameters is None
            else (
                self.parameters
                if isinstance(self.parameters, tuple)
                else (self.parameters,)
            )
        )
        has_inputs = self.inputs is not Noop()
        expected_count = 1 + int(has_inputs) + len(declarations)
        if len(args) != expected_count:
            raise ValueError(
                "Trajectory argument count does not match the system "
                f"declaration: expected {expected_count}, got {len(args)}"
            )

        time = args[0]
        input_value = args[1] if has_inputs else None
        parameter_start = 1 + int(has_inputs)
        parameters = tuple(
            self._prepare_direct_parameter(declaration, value, index)
            for index, (declaration, value) in enumerate(
                zip(declarations, args[parameter_start:])
            )
        )
        return time, input_value, parameters if declarations else (None,)

    def __call__(self, *args):
        """Integrate from time zero and return the declared output.

        Args:
            *args: ``(time, *parameters)`` without inputs, or
                ``(time, input_function, *parameters)`` with inputs.
                ``time`` is a numeric endpoint or a one-dimensional NumPy
                sampling array. Parameters follow declaration order;
                function-valued slots accept compatible Coker functions,
                fitted functions, or traceable Python callables.

        Returns:
            The output at an endpoint in its declared shape. Sampled scalar
            and one-component outputs have shape ``(n_times,)``; sampled
            multi-component vector outputs have shape ``(n_times, n_output)``.

        Raises:
            ValueError: Argument count or function parameter shape/backend
                does not match the declaration.
            TypeError: A function-valued parameter is not callable.
            NotImplementedError: Quadratures are present, or the selected
                backend does not support the required integration.
        """
        from coker.backends import get_backend_by_name

        t, u, parameter_arguments = self._map_arguments(*args)
        backend = get_backend_by_name(self.dxdt.backend)
        u = self._prepare_backend_value(backend, u)
        parameter_arguments = tuple(
            self._prepare_backend_value(backend, parameter)
            for parameter in parameter_arguments
        )
        x0, z0 = self.x0(0, u, *parameter_arguments)

        if self.dqdt is not Noop():
            raise NotImplementedError
        q0 = None

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
        """Return a function space matching the public trajectory arguments.

        Returns:
            A time-to-output space with the optional control function and
            parameter declarations following time in evaluation order.
        """
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
