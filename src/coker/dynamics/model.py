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
        inputs: Input trajectory ``FunctionSpace``, or ``None`` without inputs.
        parameters: Scalar, vector or function-space declaration, an ordered
            tuple/list of declarations, or ``None`` without parameters.
        algebraic: Algebraic-state vector declaration, or ``None``.
        initial_conditions: Callback ``initial_conditions(z, u, p)`` returning
            exactly ``(x0, z0)``; use ``None`` for absent algebraic values.
        dynamics: Callback ``dynamics(t, x, z, u, p)`` returning one rate with
            the same shape as ``x0``.
        constraints: Callback ``constraints(t, x, z, u, p)`` returning the
            zero-valued algebraic residual, or ``None`` without algebraic state.
        outputs: Callback ``outputs(t, x, z, u, p, q)`` returning the observable.
        quadratures: Callback ``quadratures(t, x, z, u, p)`` returning one
            quadrature rate, or ``None`` without quadratures.

    Raises:
        TypeError: A tuple/list parameter entry is not a supported declaration.

    Notes:
        The input ``u`` is a trajectory callable: use ``u(t)`` for its value.
        Absent channels are passed to callbacks as ``None``. A tuple/list of
        parameters supplies one ``p`` tuple in declaration order; a single
        declaration supplies its value directly. This object stores callbacks;
        :func:`coker.dynamics.create_dynamics_from_spec` traces and checks them.

    Examples:
        >>> import numpy as np
        >>> from coker.dynamics import DynamicsSpec, create_dynamics_from_spec
        >>> spec = DynamicsSpec(
        ...     inputs=None, parameters=None, algebraic=None,
        ...     initial_conditions=lambda z, u, p: (np.array([1.0]), None),
        ...     dynamics=lambda t, x, z, u, p: -x,
        ...     constraints=None,
        ...     outputs=lambda t, x, z, u, p, q: x,
        ...     quadratures=None,
        ... )
        >>> system = create_dynamics_from_spec(spec, backend="numpy")
        >>> np.testing.assert_allclose(
        ...     system.dxdt(0.0, np.array([2.0]), None, None, None), [-2.0]
        ... )
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
        inputs: Input trajectory declaration, or ``None`` without inputs.
        parameters: Optional scalar, vector or function-space declarations.
        x0: Traced initial-condition function returning ``(x0, z0)`` from
            initial algebraic values, an input trajectory and model parameters.
        dxdt: Traced rate function with arguments ``(t, x, z, u, p)``.
        g: Traced algebraic residual, or ``None`` without algebraic state.
        dqdt: Traced quadrature rate, or ``None`` without quadratures.
        y: Traced output with arguments ``(t, x, z, u, p, q)``.
        solver_parameters: Backend-specific integration settings, or ``None``
            for backend defaults.

    Notes:
        Prefer :func:`coker.dynamics.create_autonomous_ode`,
        :func:`coker.dynamics.create_control_system`, or
        :func:`coker.dynamics.create_dynamics_from_spec` to construct consistent
        traced functions. Calling this object integrates a trajectory; it does
        not perform parameter fitting or return a ``VariationalSolution``.
        Numerical support follows the backend, not just callback traceability.

    Examples:
        NumPy uses SciPy's default relative integration tolerance of ``1e-3``.

        >>> import numpy as np
        >>> from coker.dynamics import create_autonomous_ode
        >>> system = create_autonomous_ode(
        ...     x0=np.array([1.0]), xdot=lambda x, p: -x, backend="numpy"
        ... )
        >>> np.testing.assert_allclose(system(0.5), np.exp(-0.5), rtol=1e-3)
        >>> np.testing.assert_allclose(
        ...     [system(t) for t in (0.0, 0.5, 1.0)],
        ...     np.exp(-np.array([0.0, 0.5, 1.0])), rtol=1e-3,
        ... )
        >>> assert system.backend == "numpy"
        >>> differential, algebraic, quadrature = system.get_state_dimensions()
        >>> assert differential.flat() == 1
        >>> assert system.output_as_function_space().name == "y"
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
        """Describe the public trajectory callable's argument and output spaces.

        Returns:
            FunctionSpace: Space named ``"y"`` accepting time, the input
            trajectory when declared, and one value per parameter declaration.
            Its single output is the declared observable, not internal state.
        """
        return _output_function_space(self.y, self.inputs, self.parameters)

    def get_state_dimensions(self) -> Tuple[Dimension, Dimension, Dimension]:
        """Return dimensions of the channels used by the output evaluator.

        Returns:
            tuple[Dimension, Dimension, Dimension]: Differential, algebraic and
            quadrature dimensions, in that order. Absent channels retain the
            output function's absence metadata.
        """
        shapes = self.y.input_shape()
        return shapes[1], shapes[2], shapes[-1]

    @property
    def backend(self) -> str:
        """Return the backend used by the differential-rate function.

        Returns:
            str: Registered backend name, such as ``"numpy"`` or ``"casadi"``.
        """
        return self.dxdt.backend

    def __call__(self, *args):
        """Integrate and evaluate the declared observable at physical times.

        Args:
            *args: Physical endpoint or one-dimensional time grid, followed by
                the input trajectory callable when declared, then one value
                per parameter declaration. Omit absent inputs and parameters;
                do not add ``None`` placeholders.

        Returns:
            object: Declared output at an endpoint. On a supported time grid,
            larger vectors have one row per requested time. Numeric values
            follow the evaluation backend's conventions.

        Raises:
            NotImplementedError: Quadratures are declared, or the selected
                integration backend does not support the model.
            ValueError: The argument count does not match the declarations,
                or a function parameter has an incompatible space or backend.
            TypeError: A declared function parameter value is not callable.

        Notes:
            Use :func:`coker.dynamics.to_residual_dynamical_system` to evaluate
            trajectories with quadratures; their initial values then start at
            zero. Integration errors propagate from the selected backend.
            NumPy squeezes one-component outputs to zero-dimensional values
            at scalar endpoints. Time-grid evaluation of these outputs
            currently raises ``ValueError`` when concatenating them; evaluate
            separate Python scalar endpoints instead.
        """
        if not isinstance(self.dqdt, Noop):
            raise NotImplementedError
        from coker.dynamics.residual import to_residual_dynamical_system

        return to_residual_dynamical_system(self)(*args)
