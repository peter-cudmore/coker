"""Closure-backed function parameters."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace
from coker.algebra.function import Function, function
from coker.parameters import (
    BoundVector,
    BoundedVariable,
    DenseTensorVariable,
    ParameterVariable,
    UnboundedVariable,
)

from .base import FunctionParameter, _concrete_parameter_space


class ClosureParameter(FunctionParameter):
    """Bind leading concrete decisions to a Coker function.

    Args:
        function: Coker function taking concrete blocks first, followed by one
            public input. Its name becomes the declaration's name.
        parameters: Non-empty scalar or dense declarations in leading-input order.

    Raises:
        TypeError: The function is not a Coker ``Function`` or a concrete
            declaration is not a supported scalar or dense parameter.
        ValueError: The parameter sequence is empty.

    Shapes, bounds, guesses and units belong to the concrete declarations.
    Signature compatibility is checked by ``validate_target``.

    Examples:
        >>> from coker import Scalar
        >>> from coker.parameters import UnboundedVariable
        >>> from coker.parameters.function_parameters import ClosureParameter
        >>> scale = ClosureParameter.bind_callable(
        ...     lambda gain, x: gain * x, [UnboundedVariable("gain", 1.0)],
        ...     [Scalar("x")], name="scale")
        >>> float(scale.evaluate([2.0], 3.0))
        6.0
        >>> scale.list_concrete_parameters()[0].name
        'gain'
    """

    def __init__(
        self, function: Function, parameters: Sequence[ParameterVariable]
    ) -> None:
        if not isinstance(function, Function):
            raise TypeError("function must be a Coker Function")
        if not parameters:
            raise ValueError("parameters must not be empty")
        if not all(
            isinstance(
                parameter,
                (
                    BoundedVariable,
                    BoundVector,
                    DenseTensorVariable,
                    UnboundedVariable,
                ),
            )
            for parameter in parameters
        ):
            raise TypeError("parameters must be scalar or dense parameter declarations")
        self.function = function
        self.parameters = tuple(parameters)
        self.name = function.name

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        """Validate concrete-first inputs and the public output signature.

        Args:
            target: Public function space; use one public input.

        Returns:
            FunctionSpace: The unchanged compatible target.

        Raises:
            TypeError: The target is not a ``FunctionSpace``.
            ValueError: Source input or output dimensions are incompatible.
        """
        if not isinstance(target, FunctionSpace):
            raise TypeError("target must be a FunctionSpace")
        expected_inputs = FunctionSpace(
            "closure_parameter",
            arguments=[
                *(
                    _concrete_parameter_space(parameter)
                    for parameter in self.parameters
                ),
                *target.arguments,
            ],
            output=target.output,
        ).input_dimensions()
        if tuple(self.function.input_shape()) != tuple(expected_inputs):
            raise ValueError(
                "function inputs must be the concrete parameters followed by "
                "the target arguments"
            )
        if tuple(self.function.output_shape()) != tuple(target.output_dimensions()):
            raise ValueError("function output must match the target output")
        return target

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        """Return the supplied declarations as a tuple in source-input order."""
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        """Evaluate with concrete values before the single public input.

        Args:
            parameters: Values matching concrete declaration order and shapes.
            argument: Public input value.

        Returns:
            Any: Source function result, without numeric conversion.
        """
        return self.function(*parameters, argument)

    @staticmethod
    def bind_callable(
        implementation: Callable[..., Any],
        parameters: Sequence[ParameterVariable],
        inputs: Sequence[Scalar | VectorSpace],
        *,
        name: str | None = None,
        backend: str = "numpy",
    ) -> ClosureParameter:
        """Trace an implementation whose concrete blocks precede its input.

        Args:
            implementation: Callable accepting concrete values then public input,
                supporting graph tracing with the selected backend.
            parameters: Non-empty scalar or dense concrete declarations.
            inputs: Public input spaces; use one input for closure evaluation.
            name: Function name; defaults to ``"bound_"`` followed by the
                implementation's ``__name__``.
            backend: Backend used to trace the source function.

        Returns:
            ClosureParameter: Declaration wrapping the traced source function.

        Raises:
            TypeError: A concrete declaration has an unsupported type.
            ValueError: The concrete declaration sequence is empty.
        """
        function_name = name or f"bound_{implementation.__name__}"
        return ClosureParameter(
            function(
                [
                    *(_concrete_parameter_space(parameter) for parameter in parameters),
                    *inputs,
                ],
                implementation,
                name=function_name,
                backend=backend,
            ),
            parameters,
        )
