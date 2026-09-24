"""Function-valued parameter declarations for dynamical systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
from coker.interfaces import FunctionSignatureValue


class FittedFunction(FunctionSignatureValue):
    """A concrete function reconstructed from fitted parameter decisions."""

    def __init__(
        self,
        specification: FunctionParameter,
        space: FunctionSpace,
        function: Callable[..., Any],
        parameters: Any,
    ) -> None:
        self.specification = specification
        self.space = space
        self.function = function
        self.parameters = parameters

    def __call__(self, *arguments: Any, **kwargs: Any) -> Any:
        return self.function(*arguments, **kwargs)

    def input_shape(self):
        return tuple(self.space.input_dimensions())

    def output_shape(self):
        return tuple(self.space.output_dimensions())


class FunctionParameter(ABC):
    """Declare concrete parameter blocks that realize a function parameter."""

    name: str | None

    @abstractmethod
    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        """Validate and return the compatible function parameter space."""

    @abstractmethod
    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        """Return the named parameter blocks used to realize this function."""

    @abstractmethod
    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        """Evaluate the declared function from structured parameter values."""

    def build_function(
        self, target: FunctionSpace, backend: str | None
    ) -> Function:
        """Build a function of the target argument and concrete blocks."""
        target = self.validate_target(target)
        declarations = self.list_concrete_parameters()
        parameter_spaces = [
            _concrete_parameter_space(declaration)
            for declaration in declarations
        ]
        argument_count = len(target.arguments)
        return function(
            [*target.arguments, *parameter_spaces],
            lambda *arguments: self.evaluate(
                tuple(arguments[argument_count:]), arguments[0]
            ),
            backend=backend,
        )


def _concrete_parameter_space(
    declaration: ParameterVariable,
) -> Scalar | VectorSpace:
    """Return the algebraic space represented by one concrete declaration."""
    if isinstance(declaration, (BoundedVariable, UnboundedVariable)):
        return Scalar(declaration.name)
    if isinstance(declaration, (BoundVector, DenseTensorVariable)):
        return VectorSpace(declaration.name, declaration.shape)
    raise TypeError(
        "function parameter declarations must be scalar or dense variables"
    )
