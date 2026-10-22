"""Implicit residual dynamical-system model and legacy conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias, cast

import numpy as np

from coker.algebra.dimensions import Dimension, FunctionSpace, VectorSpace
from coker.algebra.function import Function, function
from coker.algebra.graph import TraceContext
from coker.algebra.ops import Noop
from coker.dynamics.model import (
    DynamicalSystem,
    DynamicsParameters,
    _output_function_space,
)
from coker.parameters.function_parameters import FittedFunction


@dataclass(frozen=True)
class LegacyFormCertificate:
    """Certified public semi-explicit lowering; never an implicit solve."""

    rate: Function
    quadrature: Function | Noop
    algebraic: Function | Noop


@dataclass(frozen=True)
class ResidualDynamicalSystem:
    """Define a trajectory by ``F(t, w, wdot, z, u, *parameters) = 0``.

    ``w`` concatenates differential state ``x`` and quadrature state ``q``;
    ``wdot`` has the same layout. ``z`` contains algebraic variables, using an
    empty vector when absent. The residual must return one vector with
    ``differential.flat() + quadrature.flat() + algebraic.flat()`` rows,
    counting absent dimensions as zero. Rows may be coupled; no semi-explicit
    form is inferred or solved symbolically.

    Integration starts at time zero. ``x0(0, u, *parameters)`` supplies
    ``(x0, z0)``; the first argument is the initialization slot, not a
    user-selected initial time. Supply algebraic initial values when present;
    construction does not project them onto the constraint manifold.
    Quadratures start at zero. Outputs use
    ``y(t, x, z, u, *parameters, q)``, with absent ``z``/``q`` represented by
    ``None``. Parameter-free callbacks retain one ``None`` parameter slot.

    Independent residual simulation is supported by NumPy's implicit
    state-step solver and CasADi's IDAS/direct-collocation solvers, with their
    backend-specific restrictions and solver settings. NumPy and IDAS require
    a differential or quadrature state; no higher-index solvability guarantee
    is made. Converted semi-explicit systems retain their original direct
    integration path. Numerical residual support does not imply symbolic
    analysis support: ``analyse_*`` accepts supported semi-explicit index-one
    models, not ``ResidualDynamicalSystem``.

    Args:
        inputs: Input ``FunctionSpace``, or ``None``/``Noop`` for no input.
            Input callbacks receive the trajectory and evaluate it as ``u(t)``.
        parameters: Scalar, vector or function-valued declarations, a sequence
            of declarations, or ``None``. Sequence entries become separate
            callback arguments in declaration order.
        x0: Initial-condition ``Function`` returning ``(x0, z0)``.
        F: Residual ``Function`` with the signature and layout above.
        y: Output ``Function`` returning one model output.
        differential: Finite differential-state ``Dimension``.
        algebraic: Finite algebraic-state ``Dimension``, or ``None``.
        quadrature: Finite quadrature-state ``Dimension``, or ``None``.
        solver_parameters: Backend-specific integration settings.
        legacy: Internal conversion certificate; leave at its default.

    Raises:
        TypeError: Dimensions or callbacks have unsupported types, or residual
            callback arity does not match the parameter declarations.
        ValueError: Dimensions are invalid or callbacks use different backends.

    Examples:
        Define the implicit equation ``xdot + x = 0`` without conversion:

        >>> import numpy as np
        >>> from coker import Dimension, Scalar, VectorSpace, function
        >>> from coker.dynamics import ResidualDynamicalSystem
        >>> t, x = Scalar("t"), VectorSpace("x", 1)
        >>> residual = ResidualDynamicalSystem(
        ...     inputs=None, parameters=None,
        ...     x0=function([None, None, None],
        ...                 lambda z, u, p: (np.ones(1), None), backend="numpy"),
        ...     F=function(
        ...         [t, x, VectorSpace("xdot", 1), VectorSpace("z", 0), None, None],
        ...         lambda t, w, wdot, z, u, p: wdot + w, backend="numpy"),
        ...     y=function([t, x, None, None, None, None],
        ...                lambda t, x, z, u, p, q: x, backend="numpy"),
        ...     differential=Dimension(1), algebraic=None, quadrature=None,
        ... )
        >>> np.testing.assert_allclose(
        ...     residual.F(
        ...         0.0, np.ones(1), -np.ones(1), np.empty(0), None, None),
        ...     0.0, atol=1e-12,
        ... )
        >>> np.testing.assert_allclose(residual(0.0), 1.0)
        >>> np.testing.assert_allclose(residual(0.1), np.exp(-0.1), rtol=1e-5)
    """

    inputs: FunctionSpace | Noop | None
    parameters: DynamicsParameters
    x0: Function
    F: Function
    y: Function
    differential: Dimension
    algebraic: Dimension | None
    quadrature: Dimension | None
    solver_parameters: object | None = None
    legacy: LegacyFormCertificate | None = None

    def __post_init__(self) -> None:
        if self.inputs is None:
            object.__setattr__(self, "inputs", Noop())
        _validate_residual_system(self)

    def get_state_dimensions(
        self,
    ) -> tuple[Dimension, Dimension | None, Dimension | None]:
        """Return the differential, algebraic, and quadrature dimensions.

        Returns:
            A ``(differential, algebraic, quadrature)`` dimension tuple.

        Examples:
            >>> from coker.dynamics import (
            ...     create_autonomous_ode, to_residual_dynamical_system,
            ... )
            >>> residual = to_residual_dynamical_system(
            ...     create_autonomous_ode([1.0], lambda x, p: -x)
            ... )
            >>> residual.get_state_dimensions()[0].flat()
            1
        """
        return self.differential, self.algebraic, self.quadrature

    @property
    def backend(self) -> str:
        """Return the backend used by the residual callback.

        Returns:
            Registered backend name for this model.

        Examples:
            >>> from coker.dynamics import (
            ...     create_autonomous_ode, to_residual_dynamical_system,
            ... )
            >>> residual = to_residual_dynamical_system(
            ...     create_autonomous_ode([1.0], lambda x, p: -x)
            ... )
            >>> residual.backend
            'numpy'
        """
        return self.F.backend

    def output_as_function_space(self) -> FunctionSpace:
        """Return the public output function space.

        Returns:
            Function space accepting time, declared input, and declared
            parameters and returning the model output.

        Examples:
            >>> from coker import FunctionSpace
            >>> from coker.dynamics import (
            ...     create_autonomous_ode, to_residual_dynamical_system,
            ... )
            >>> residual = to_residual_dynamical_system(
            ...     create_autonomous_ode([1.0], lambda x, p: -x)
            ... )
            >>> isinstance(residual.output_as_function_space(), FunctionSpace)
            True
        """
        return _output_function_space(self.y, self.inputs, self.parameters)

    def __call__(self, *args):
        """Evaluate outputs after integration from time zero.

        Args:
            *args: Time followed by the declared input trajectory, if present,
                then one value per parameter declaration. Do not supply absent
                input/parameter placeholders. A function-valued parameter must
                match its declared ``FunctionSpace``; supplied ``Function``
                objects must also use this system's backend.

        Returns:
            The model output at a Python scalar endpoint, or outputs stacked
            over a supported one-dimensional time grid. Larger vector outputs
            are vertically stacked. This returns outputs, not an analysis
            result or a state-solution object.

        Raises:
            ValueError: Argument count or a function-valued parameter differs
                from the declarations, or the backend rejects times, shapes
                or solver settings.
            TypeError: A function-valued parameter is not callable.
            NotImplementedError: The backend does not support this residual
                model or its state partition.
            RuntimeError: The selected numerical solver fails.

        Notes:
            NumPy squeezes one-component outputs to zero-dimensional values
            at scalar endpoints. Time-grid evaluation of these outputs
            currently raises ``ValueError`` when concatenating them; evaluate
            separate Python scalar endpoints instead.

        Examples:
            >>> import numpy as np
            >>> from coker.dynamics import (
            ...     create_autonomous_ode, to_residual_dynamical_system,
            ... )
            >>> residual = to_residual_dynamical_system(
            ...     create_autonomous_ode([1.0], lambda x, p: -x)
            ... )
            >>> np.testing.assert_allclose(residual(0.0), 1.0)
            >>> np.testing.assert_allclose(
            ...     residual(0.1), np.exp(-0.1), rtol=1e-5,
            ... )
        """
        return _evaluate_residual_trajectory(self, *args)


def _evaluate_residual_trajectory(system: ResidualDynamicalSystem, *args):
    """Evaluate one residual system trajectory through its declared backend."""
    time, inputs, parameters = _map_residual_arguments(system, args)
    from coker.backends import get_backend_by_name

    backend = get_backend_by_name(system.backend)
    inputs = _prepare_backend_value(backend, inputs)
    parameters = tuple(
        _prepare_backend_value(backend, parameter) for parameter in parameters
    )
    x0, z0 = system.x0(0, inputs, *parameters)
    q0 = _initial_residual_quadrature(system)
    x, z, q = backend.evaluate_integrals(
        system,
        [x0, z0, q0],
        time,
        [inputs, *parameters],
        solver_parameters=system.solver_parameters,
    )
    if isinstance(time, (float, int)):
        return system.y(time, x, z, inputs, *parameters, q)

    def map_arguments(index):
        x_i = x[:, index]
        z_i = z[:, index] if z is not None else None
        q_i = q[:, index] if q is not None else None
        return x_i, z_i, inputs, *parameters, q_i

    if system.y.output_shape()[0].is_scalar() or system.y.output_shape()[0].dim == (1,):
        return np.concatenate(
            [
                system.y(time_i, *map_arguments(index))
                for index, time_i in enumerate(time)
            ]
        )
    return np.vstack(
        [system.y(time_i, *map_arguments(index)) for index, time_i in enumerate(time)]
    )


def _map_residual_arguments(system: ResidualDynamicalSystem, args):
    declarations = (
        ()
        if system.parameters is None
        else (
            system.parameters
            if isinstance(system.parameters, (tuple, list))
            else (system.parameters,)
        )
    )
    has_inputs = not isinstance(system.inputs, Noop)
    expected_count = 1 + int(has_inputs) + len(declarations)
    if len(args) != expected_count:
        raise ValueError(
            "Trajectory argument count does not match the system declaration: "
            f"expected {expected_count}, got {len(args)}"
        )
    input_value = args[1] if has_inputs else None
    parameter_start = 1 + int(has_inputs)
    parameters = tuple(
        _prepare_residual_parameter(system, declaration, value, index)
        for index, (declaration, value) in enumerate(
            zip(declarations, args[parameter_start:])
        )
    )
    return args[0], input_value, parameters if declarations else (None,)


def _prepare_residual_parameter(system, declaration, value, index):
    if not isinstance(declaration, FunctionSpace):
        return value
    if isinstance(value, Function):
        if value.backend != system.backend:
            raise ValueError(
                f"Function-valued parameter {index} uses backend "
                f"{value.backend!r}, expected {system.backend!r}"
            )
        if value not in declaration:
            raise ValueError(
                f"Function-valued parameter {index} does not match declared "
                f"FunctionSpace {declaration.name!r}"
            )
        return value
    if isinstance(value, FittedFunction) and value not in declaration:
        raise ValueError(
            f"Function-valued parameter {index} does not match declared "
            f"FunctionSpace {declaration.name!r}"
        )
    if not callable(value):
        raise TypeError(f"Function-valued parameter {index} must be callable")
    prepared = function(declaration.arguments, value, backend=system.backend)
    if prepared not in declaration:
        raise ValueError(
            f"Function-valued parameter {index} does not match declared "
            f"FunctionSpace {declaration.name!r}"
        )
    return prepared


def _prepare_backend_value(backend, value):
    return (
        value if value is None or callable(value) else backend.to_backend_array(value)
    )


def _initial_residual_quadrature(system: ResidualDynamicalSystem) -> np.ndarray | None:
    if system.quadrature is None or system.quadrature.flat() == 0:
        return None
    return np.zeros(system.quadrature.flat())


_DirectIntegrationCallback: TypeAlias = Callable[..., Any]
_DirectIntegrationCallbacks: TypeAlias = tuple[
    _DirectIntegrationCallback,
    _DirectIntegrationCallback | Noop | None,
    _DirectIntegrationCallback | Noop | None,
]
_DirectIntegrationFunctionSource: TypeAlias = (
    DynamicalSystem
    | ResidualDynamicalSystem
    | _DirectIntegrationCallbacks
    | list[_DirectIntegrationCallback | Noop | None]
)


def _normalise_direct_integration_functions(
    functions: _DirectIntegrationFunctionSource,
) -> _DirectIntegrationCallbacks:
    """Return the semi-explicit callbacks supported by direct integrators."""
    if isinstance(functions, ResidualDynamicalSystem):
        certificate = functions.legacy
        if certificate is None:
            raise NotImplementedError(
                "Direct integration does not support independent residual "
                "systems without a semi-explicit certificate"
            )
        return (
            certificate.rate,
            certificate.algebraic,
            certificate.quadrature,
        )
    if isinstance(functions, DynamicalSystem):
        return functions.dxdt, functions.g, functions.dqdt

    rate, algebraic, quadrature = functions
    return (
        cast(_DirectIntegrationCallback, rate),
        cast(_DirectIntegrationCallback | Noop | None, algebraic),
        cast(_DirectIntegrationCallback | Noop | None, quadrature),
    )


def _validate_residual_system(system: ResidualDynamicalSystem) -> None:
    """Validate the invariants required to evaluate a residual system."""
    _dimension_size(system.differential, "differential")
    _optional_dimension_size(system.algebraic, "algebraic")
    _optional_dimension_size(system.quadrature, "quadrature")

    _validate_callback(system.F, "residual")
    _validate_callback(system.x0, "initial-condition", system.F.backend)
    _validate_callback(system.y, "output", system.F.backend)
    _validate_residual_input_arity(system.F, system.parameters)


def _dimension_size(dimension: object, name: str) -> int:
    if not isinstance(dimension, Dimension):
        raise TypeError(f"{name} dimension must be a Dimension")
    try:
        size = dimension.flat()
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} dimension must be finite") from ex
    if size < 0:
        raise ValueError(f"{name} dimension must not have a negative size")
    return size


def _optional_dimension_size(dimension: object, name: str) -> int:
    if dimension is None:
        return 0
    return _dimension_size(dimension, name)


def _validate_residual_input_arity(
    callback: Function, parameters: DynamicsParameters
) -> None:
    parameter_count = len(parameters) if isinstance(parameters, (tuple, list)) else 1
    expected_count = 5 + parameter_count
    actual_count = len(callback.input_spaces())
    if actual_count != expected_count:
        raise TypeError(
            f"residual callback has {actual_count} inputs; "
            f"expected {expected_count}"
        )


def _validate_callback(callback: object, name: str, backend: str | None = None) -> None:
    if not isinstance(callback, Function):
        raise TypeError(f"{name} callback must be a Function")
    if backend is not None and callback.backend != backend:
        raise ValueError(
            f"{name} callback uses backend {callback.backend!r}, "
            f"expected {backend!r}"
        )


def _compose_residual_callback(callback: Function, *arguments: Any) -> Any:
    """Record callable inputs without switching the callback's backend."""
    if not any(isinstance(space, FunctionSpace) for space in callback.input_spaces()):
        return callback(*arguments)

    tape = TraceContext.get_local_tape()
    assert tape is not None
    outputs = callback._append_symbolic_call(tape, arguments)
    return outputs[0] if callback.is_single else tuple(outputs)


def to_residual_dynamical_system(
    system: DynamicalSystem,
) -> ResidualDynamicalSystem:
    """Convert a semi-explicit system to an equivalent residual representation.

    Args:
        system: Explicit or semi-explicit ``DynamicalSystem`` to convert.

    Returns:
        A model with residual rows ``xdot - f``, then ``qdot - h`` when
        quadratures exist, then algebraic constraints ``g``. The differential,
        algebraic and quadrature dimensions, initialization, output, input and
        parameter declarations, backend and solver settings are preserved.
        Its internal certificate retains the original semi-explicit direct
        integration path; conversion is not an implicit symbolic solve.

    Raises:
        ValueError: A nonempty quadrature or algebraic partition lacks its
            corresponding ``Function`` callback.

    Examples:
        >>> import numpy as np
        >>> from coker.dynamics import (
        ...     create_autonomous_ode, to_residual_dynamical_system,
        ... )
        >>> ode = create_autonomous_ode([1.0], lambda x, p: -x)
        >>> residual = to_residual_dynamical_system(ode)
        >>> np.testing.assert_allclose(
        ...     residual.F(
        ...         0.0, np.ones(1), -np.ones(1), np.empty(0), None, None),
        ...     0.0, atol=1e-12,
        ... )
        >>> np.testing.assert_allclose(residual(0.0), 1.0)
        >>> np.testing.assert_allclose(residual(0.1), np.exp(-0.1), rtol=1e-5)
    """
    x_dim, z_dim, q_dim = system.get_state_dimensions()
    state_size = x_dim.flat()
    quadrature_size = 0 if q_dim is None else q_dim.flat()
    algebraic_size = 0 if z_dim is None else z_dim.flat()
    rate_size = state_size + quadrature_size
    state_rows = slice(0, state_size)
    quadrature_rows = slice(state_size, rate_size)
    w_space = VectorSpace("_residual_w", rate_size)
    wdot_space = VectorSpace("_residual_wdot", rate_size)
    z_space = VectorSpace("_residual_z", algebraic_size)
    time_space, _, _, input_space, *parameter_spaces = system.dxdt.input_spaces()
    quadrature = system.dqdt
    algebraic = system.g

    if quadrature_size and not isinstance(quadrature, Function):
        raise ValueError("system quadrature layout requires a quadrature callback")
    if algebraic_size and not isinstance(algebraic, Function):
        raise ValueError("system algebraic layout requires a constraint callback")

    legacy = LegacyFormCertificate(
        rate=system.dxdt,
        quadrature=quadrature if isinstance(quadrature, Function) else Noop(),
        algebraic=algebraic if isinstance(algebraic, Function) else Noop(),
    )

    def residual(t: Any, w: Any, wdot: Any, z: Any, u: Any, *parameters: Any) -> Any:
        x = w[state_rows]
        z_public = z if algebraic_size else None
        state_rate = _compose_residual_callback(
            system.dxdt, t, x, z_public, u, *parameters
        )
        state_residual = wdot[state_rows] - np.reshape(state_rate, (state_size,))
        rows = [state_residual]
        if quadrature_size:
            assert isinstance(quadrature, Function)
            quadrature_rate = _compose_residual_callback(
                quadrature, t, x, z_public, u, *parameters
            )
            rows.append(
                wdot[quadrature_rows] - np.reshape(quadrature_rate, (quadrature_size,))
            )
        if algebraic_size:
            assert isinstance(algebraic, Function)
            algebraic_residual = _compose_residual_callback(
                algebraic, t, x, z, u, *parameters
            )
            rows.append(np.reshape(algebraic_residual, (algebraic_size,)))
        return rows[0] if len(rows) == 1 else np.concatenate(rows)

    residual_function = function(
        [
            time_space,
            w_space,
            wdot_space,
            z_space,
            input_space,
            *parameter_spaces,
        ],
        cast(Callable[..., Any], residual),
        backend=system.backend,
        name="_residual",
    )
    return ResidualDynamicalSystem(
        inputs=system.inputs,
        parameters=system.parameters,
        x0=system.x0,
        F=residual_function,
        y=system.y,
        differential=x_dim,
        algebraic=z_dim,
        quadrature=q_dim,
        solver_parameters=system.solver_parameters,
        legacy=legacy,
    )
