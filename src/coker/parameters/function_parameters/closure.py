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
    """Declare a function whose leading inputs become solver decisions.

    Args:
        function: Coker function with concrete blocks before public inputs.
            Its name becomes the fitted-function name.
        parameters: Nonempty scalar/dense decision declarations, matching the
            leading input shapes in order.

    Raises:
        TypeError: The function or concrete declaration types are unsupported.
        ValueError: No concrete parameters are supplied.

    Example:
        Use ``ClosureParameter(f, [UnboundedVariable("gain", 1.0)])`` when
        ``f`` takes a scalar gain followed by the public function argument.
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
        """Return the target after checking ordered input and output shapes."""
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
        """Return the leading concrete decision declarations in input order."""
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        """Evaluate the function with concrete blocks before its public input."""
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
        """Trace an implementation whose concrete blocks precede its inputs.

        Args:
            implementation: Python callable accepting decision blocks first.
            parameters: Ordered scalar/dense decision declarations.
            inputs: Public input spaces following the concrete blocks.
            name: Function name; defaults to the implementation name prefixed
                by ``"bound_"``.
            backend: Execution backend, defaulting to ``"numpy"``.

        Returns:
            A closure declaration wrapping the traced Coker function.

        Example:
            Bind ``lambda gain, x: gain * x`` with one scalar gain declaration
            and ``inputs=[Scalar("x")]``.
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
