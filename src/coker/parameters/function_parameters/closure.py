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
    """Realize a function parameter by binding decisions to a function."""

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
            raise TypeError(
                "parameters must be scalar or dense parameter declarations"
            )
        self.function = function
        self.parameters = tuple(parameters)
        self.name = function.name

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
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
        if tuple(self.function.output_shape()) != tuple(
            target.output_dimensions()
        ):
            raise ValueError("function output must match the target output")
        return target

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
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
        """Trace an implementation whose concrete blocks precede its inputs."""
        function_name = name or f"bound_{implementation.__name__}"
        return ClosureParameter(
            function(
                [
                    *(
                        _concrete_parameter_space(parameter)
                        for parameter in parameters
                    ),
                    *inputs,
                ],
                implementation,
                name=function_name,
                backend=backend,
            ),
            parameters,
        )
