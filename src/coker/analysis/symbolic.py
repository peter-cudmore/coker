"""Lower Coker dynamical systems to finite SymPy expressions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import sympy as sp
from sympy.core.function import AppliedUndef

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.function import Function
from coker.algebra.ops import Noop
from coker.backends.sympy import SympyBackend

from coker.dynamics.model import DynamicalSystem

__all__ = ("SymbolicSystem", "UnsupportedSystemError", "lower_system")


class UnsupportedSystemError(ValueError):
    """Raised when a model is outside the symbolic analysis domain."""


class _UnsupportedShapeError(UnsupportedSystemError):
    """Internal marker for a finite shape not covered by this analysis."""


@dataclass(frozen=True)
class SymbolicSystem:
    """Finite symbolic ODE data in the order declared by a Coker model."""

    state: tuple[sp.Symbol, ...]
    parameters: tuple[sp.Symbol, ...]
    controls: tuple[sp.Symbol, ...]
    dynamics: tuple[sp.Expr, ...]
    outputs: tuple[sp.Expr, ...]


def lower_system(system: DynamicalSystem | SymbolicSystem) -> SymbolicSystem:
    """Return finite autonomous ODE expressions suitable for rank analysis.

    The lowering is deliberately conservative.  Models outside the supported
    class raise :class:`UnsupportedSystemError`; analysis callers can turn
    that into an inconclusive result instead of applying an invalid test.
    """

    if isinstance(system, SymbolicSystem):
        return system
    if not isinstance(system, DynamicalSystem):
        raise UnsupportedSystemError(
            "system analysis requires a DynamicalSystem or SymbolicSystem"
        )

    declarations = _parameter_declarations(system.parameters)
    control_dimension = _control_dimension(system.inputs)
    parameter_slots = len(declarations) if declarations else 1

    _reject_dae_or_quadrature(system)
    for function, subject in (
        (system.dxdt, "dynamics"),
        (system.y, "outputs"),
    ):
        _validate_function(function, subject)

    dynamics_shapes = system.dxdt.input_shape()
    output_shapes = system.y.input_shape()
    for shapes, expected_size, include_quadrature, subject in (
        (
            dynamics_shapes,
            4 + parameter_slots,
            False,
            "dynamics",
        ),
        (output_shapes, 5 + parameter_slots, True, "outputs"),
    ):
        _validate_signature(
            shapes,
            expected_size=expected_size,
            parameter_declarations=declarations,
            control_dimension=control_dimension,
            include_quadrature=include_quadrature,
            subject=subject,
        )

    state_dimension = _require_finite_dimension(dynamics_shapes[1], "state")
    output_state_dimension = _require_finite_dimension(
        output_shapes[1], "output state"
    )
    if output_state_dimension != state_dimension:
        raise UnsupportedSystemError(
            "output state shape does not match the dynamics state shape"
        )

    dynamics_output_shapes = system.dxdt.output_shape()
    if len(dynamics_output_shapes) != 1:
        raise UnsupportedSystemError("dynamics must have exactly one output")
    dynamics_dimension = _require_finite_dimension(
        dynamics_output_shapes[0], "dynamics output"
    )
    if dynamics_dimension != state_dimension:
        raise UnsupportedSystemError(
            "dynamics output shape does not match the state shape"
        )
    for index, shape in enumerate(system.y.output_shape()):
        _require_finite_dimension(shape, f"output {index}")

    (
        (dynamics_args, raw_dynamics),
        (output_args, raw_outputs),
    ) = _lower_functions(
        (system.dxdt, dynamics_shapes),
        (system.y, output_shapes),
    )

    dynamics_time = _scalar_symbol(dynamics_args[0], "dynamics time")
    output_time = _scalar_symbol(output_args[0], "output time")
    state = _argument_symbols(dynamics_args[1], state_dimension, "state")
    output_state = _argument_symbols(
        output_args[1], output_state_dimension, "output state"
    )

    parameter_start = 4
    parameters = _argument_symbols_for_declarations(
        dynamics_args[parameter_start : parameter_start + parameter_slots],
        declarations,
        "parameter",
    )
    output_parameters = _argument_symbols_for_declarations(
        output_args[parameter_start : parameter_start + parameter_slots],
        declarations,
        "output parameter",
    )

    if not declarations:
        if (
            dynamics_args[parameter_start] is not None
            or output_args[parameter_start] is not None
        ):
            raise UnsupportedSystemError(
                "parameter-free model has a parameter input"
            )

    _ensure_distinct_symbols((dynamics_time,), state, parameters)

    raw_dynamics_expressions = _function_expressions(
        system.dxdt, raw_dynamics, "dynamics"
    )
    raw_output_expressions = _function_expressions(
        system.y, raw_outputs, "outputs"
    )

    if control_dimension is None:
        if dynamics_args[3] is not None or output_args[3] is not None:
            raise UnsupportedSystemError(
                "autonomous model has a control input"
            )
        controls: tuple[sp.Symbol, ...] = ()
        dynamics_control_values: tuple[sp.Expr, ...] = ()
        output_control_values: tuple[sp.Expr, ...] = ()
    else:
        dynamics_control_values = _control_values(
            dynamics_args[3], dynamics_time, control_dimension, "control"
        )
        output_control_values = _control_values(
            output_args[3], output_time, control_dimension, "output control"
        )
        controls = _control_symbols(dynamics_control_values, state, parameters)

    dynamics = _replace_expressions(
        raw_dynamics_expressions,
        dict(zip(dynamics_control_values, controls)),
    )
    output_replacements = {
        output_time: dynamics_time,
        **dict(zip(output_state, state)),
        **dict(zip(output_parameters, parameters)),
        **dict(zip(output_control_values, controls)),
    }
    outputs = _replace_expressions(raw_output_expressions, output_replacements)

    _reject_non_autonomous_or_implicit(dynamics, dynamics_time, "dynamics")
    _reject_non_autonomous_or_implicit(outputs, dynamics_time, "outputs")
    _reject_nonsmooth(dynamics, "dynamics")
    _reject_nonsmooth(outputs, "outputs")

    return SymbolicSystem(
        state=state,
        parameters=parameters,
        controls=controls,
        dynamics=dynamics,
        outputs=outputs,
    )


def _parameter_declarations(
    parameters: Scalar | VectorSpace | tuple[object, ...] | None,
) -> tuple[Scalar | VectorSpace, ...]:
    if parameters is None:
        return ()
    if isinstance(parameters, (Scalar, VectorSpace)):
        declarations = (parameters,)
    elif isinstance(parameters, tuple):
        declarations = parameters
    else:
        raise UnsupportedSystemError(
            "parameters must be scalar or vector spaces"
        )

    for index, declaration in enumerate(declarations):
        if isinstance(declaration, FunctionSpace):
            raise UnsupportedSystemError(
                f"function-valued parameter {index} is not supported"
            )
        if not isinstance(declaration, (Scalar, VectorSpace)):
            raise UnsupportedSystemError(
                f"parameter {index} is not a finite scalar or vector space"
            )
        _require_finite_dimension(
            _dimension_from_declaration(declaration), f"parameter {index}"
        )
    return declarations


def _control_dimension(inputs: object) -> Dimension | None:
    if isinstance(inputs, Noop):
        return None
    if not isinstance(inputs, FunctionSpace):
        raise UnsupportedSystemError(
            "controls must be declared as a FunctionSpace"
        )
    if len(inputs.arguments) != 1 or not isinstance(
        inputs.arguments[0], Scalar
    ):
        raise UnsupportedSystemError(
            "controls must be functions of one scalar time argument"
        )
    dimensions = inputs.output_dimensions()
    if len(dimensions) != 1:
        raise UnsupportedSystemError("controls must have exactly one output")
    return _require_finite_dimension(dimensions[0], "control")


def _reject_dae_or_quadrature(system: DynamicalSystem) -> None:
    if system.g is not None and not isinstance(system.g, Noop):
        raise UnsupportedSystemError("DAE constraints are not supported")
    if system.dqdt is not None and not isinstance(system.dqdt, Noop):
        raise UnsupportedSystemError("quadrature states are not supported")


def _validate_function(value: object, subject: str) -> None:
    if not isinstance(value, Function):
        raise UnsupportedSystemError(f"{subject} must be a Coker Function")


def _lower_functions(
    *functions: tuple[Function, Sequence[object]],
) -> tuple[tuple[Sequence[object], object], ...]:
    lowered: list[tuple[Sequence[object], object]] = []
    try:
        backend = SympyBackend()
        for function, _shapes in functions:
            arguments, values = backend.lower_to_symbolic(function)
            lowered.append((arguments, values))
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise UnsupportedSystemError(
            f"SymPy cannot lower this system: {exc}"
        ) from exc

    if any(
        len(arguments) != len(shapes)
        for (arguments, _values), (_function, shapes) in zip(
            lowered, functions
        )
    ):
        raise UnsupportedSystemError(
            "lowered function arguments are inconsistent"
        )
    return tuple(lowered)


def _validate_signature(
    shapes: Sequence[Dimension | FunctionSpace | None],
    *,
    expected_size: int,
    parameter_declarations: Sequence[Scalar | VectorSpace],
    control_dimension: Dimension | None,
    include_quadrature: bool,
    subject: str,
) -> None:
    if len(shapes) != expected_size:
        raise UnsupportedSystemError(
            f"{subject} has an unsupported argument signature"
        )
    if not isinstance(shapes[0], Dimension) or not shapes[0].is_scalar():
        raise UnsupportedSystemError(f"{subject} time input must be scalar")
    _require_finite_dimension(shapes[1], f"{subject} state")
    if shapes[2] is not None:
        raise UnsupportedSystemError("algebraic states are not supported")

    control_shape = shapes[3]
    if control_dimension is None:
        if (
            not isinstance(control_shape, FunctionSpace)
            or control_shape.name != "noop"
        ):
            raise UnsupportedSystemError(
                "autonomous model has a control input"
            )
    elif not isinstance(control_shape, FunctionSpace):
        raise UnsupportedSystemError(
            "control input does not match its declaration"
        )

    parameter_start = 4
    parameter_slots = (
        len(parameter_declarations) if parameter_declarations else 1
    )
    for index in range(parameter_slots):
        shape = shapes[parameter_start + index]
        if not parameter_declarations:
            if shape is not None:
                raise UnsupportedSystemError(
                    "parameter-free model has a parameter input"
                )
            continue
        expected = _dimension_from_declaration(parameter_declarations[index])
        if shape != expected:
            raise UnsupportedSystemError(
                f"parameter {index} input does not match its declaration"
            )

    if include_quadrature and shapes[-1] is not None:
        raise UnsupportedSystemError("quadrature states are not supported")


def _dimension_from_declaration(value: Scalar | VectorSpace) -> Dimension:
    return (
        Dimension(None)
        if isinstance(value, Scalar)
        else Dimension(value.dimension)
    )


def _require_finite_dimension(value: object, subject: str) -> Dimension:
    if not isinstance(value, Dimension):
        raise _UnsupportedShapeError(
            f"{subject} must be a scalar or one-dimensional vector"
        )
    if value.is_scalar():
        return value
    if (
        not value.is_vector()
        or not isinstance(value.dim[0], int)
        or isinstance(value.dim[0], bool)
        or value.dim[0] < 1
    ):
        raise _UnsupportedShapeError(
            f"{subject} must be a finite one-dimensional vector"
        )
    return value


def _scalar_symbol(value: object, subject: str) -> sp.Symbol:
    if not isinstance(value, sp.Symbol):
        raise UnsupportedSystemError(f"{subject} did not lower to a symbol")
    return value


def _argument_symbols(
    value: object, dimension: Dimension, subject: str
) -> tuple[sp.Symbol, ...]:
    values = _values_for_dimension(
        value,
        dimension,
        f"{subject} did not lower to the declared scalar symbols",
    )
    if not all(isinstance(symbol, sp.Symbol) for symbol in values):
        raise UnsupportedSystemError(
            f"{subject} did not lower to the declared scalar symbols"
        )
    return tuple(values)  # type: ignore[return-value]


def _argument_symbols_for_declarations(
    arguments: Sequence[object],
    declarations: Sequence[Scalar | VectorSpace],
    subject: str,
) -> tuple[sp.Symbol, ...]:
    if not declarations:
        return ()
    symbols: list[sp.Symbol] = []
    for index, (argument, declaration) in enumerate(
        zip(arguments, declarations)
    ):
        symbols.extend(
            _argument_symbols(
                argument,
                _dimension_from_declaration(declaration),
                f"{subject} {index}",
            )
        )
    return tuple(symbols)


def _function_expressions(
    function: Function, values: object, subject: str
) -> tuple[sp.Expr, ...]:
    raw_values = (values,) if function.is_single else tuple(values)
    shapes = function.output_shape()
    if len(raw_values) != len(shapes):
        raise UnsupportedSystemError(
            f"{subject} lowering returned invalid outputs"
        )

    expressions: list[sp.Expr] = []
    for index, (value, shape) in enumerate(zip(raw_values, shapes)):
        dimension = _require_finite_dimension(
            shape, f"{subject} output {index}"
        )
        expressions.extend(
            _expressions_for_dimension(
                value, dimension, f"{subject} output {index}"
            )
        )
    return tuple(expressions)


def _control_values(
    control: object,
    time: sp.Symbol,
    dimension: Dimension,
    subject: str,
) -> tuple[sp.Expr, ...]:
    if not callable(control):
        raise UnsupportedSystemError(f"{subject} did not lower to a function")
    try:
        return _expressions_for_dimension(control(time), dimension, subject)
    except (TypeError, ValueError) as exc:
        raise UnsupportedSystemError(
            f"{subject} cannot be represented symbolically"
        ) from exc


def _values_for_dimension(
    value: object, dimension: Dimension, invalid_shape_message: str
) -> tuple[sp.Expr, ...]:
    values = _flatten(value)
    if len(values) != dimension.flat():
        raise UnsupportedSystemError(invalid_shape_message)
    return values


def _expressions_for_dimension(
    value: object, dimension: Dimension, subject: str
) -> tuple[sp.Expr, ...]:
    return _values_for_dimension(
        value,
        dimension,
        f"{subject} has an unsupported symbolic shape",
    )


def _control_symbols(
    control_values: Sequence[sp.Expr],
    state: Sequence[sp.Symbol],
    parameters: Sequence[sp.Symbol],
) -> tuple[sp.Symbol, ...]:
    symbols: list[sp.Symbol] = []
    used = set((*state, *parameters))
    for index, value in enumerate(control_values):
        if not isinstance(value, AppliedUndef):
            raise UnsupportedSystemError(
                "controls must lower to applied scalar functions"
            )
        name = value.func.__name__
        symbol = sp.Symbol(name)
        if symbol in used:
            symbol = sp.Symbol(f"{name}_control_{index}")
        while symbol in used:
            symbol = sp.Symbol(f"{symbol.name}_")
        used.add(symbol)
        symbols.append(symbol)
    return tuple(symbols)


def _replace_expressions(
    expressions: Iterable[sp.Expr], replacements: dict[object, object]
) -> tuple[sp.Expr, ...]:
    return tuple(
        expression.xreplace(replacements) for expression in expressions
    )


def _ensure_distinct_symbols(
    *groups: Sequence[sp.Symbol],
) -> None:
    flattened = tuple(symbol for group in groups for symbol in group)
    if len(set(flattened)) != len(flattened):
        raise UnsupportedSystemError(
            "time, state, and parameter symbols must have distinct names"
        )


def _reject_non_autonomous_or_implicit(
    expressions: Iterable[sp.Expr], time: sp.Symbol, subject: str
) -> None:
    for expression in expressions:
        if expression.has(time):
            raise UnsupportedSystemError(
                f"explicit time dependence in {subject} is not supported"
            )
        if expression.atoms(AppliedUndef):
            raise UnsupportedSystemError(
                f"implicit functions in {subject} are not supported"
            )


def _reject_nonsmooth(expressions: Iterable[sp.Expr], subject: str) -> None:
    nonsmooth = (
        sp.Abs,
        sp.Heaviside,
        sp.Piecewise,
        sp.sign,
        sp.ceiling,
        sp.floor,
        sp.Max,
        sp.Min,
    )
    if any(expression.has(*nonsmooth) for expression in expressions):
        raise UnsupportedSystemError(
            f"nonsmooth expressions in {subject} are not supported"
        )


def _flatten(value: object) -> tuple[sp.Expr, ...]:
    if isinstance(value, sp.MatrixBase):
        return tuple(
            sp.sympify(value[row, column])
            for row in range(value.rows)
            for column in range(value.cols)
        )
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            return (sp.sympify(value),)
        return tuple(
            sp.sympify(value[index])
            for index in product(*(range(length) for length in shape))
        )
    if isinstance(value, (list, tuple)):
        return tuple(element for item in value for element in _flatten(item))
    return (sp.sympify(value),)
