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
    """Retain a solved finite realization as a callable.

    Args:
        specification: Original function-parameter declaration.
        space: Validated public argument/output spaces.
        function: Backend-native callable using the fitted blocks.
        parameters: Structured fitted blocks in declaration order.

    Attributes:
        specification: Original declaration, not a new solver decision.
        space: Public callable domain and codomain.
        function: Native callable retained for subsequent evaluation.
        parameters: Backend-specific fitted arrays/tensors or scalar values.

    Example:
        >>> from coker import FunctionSpace, Scalar
        >>> from coker.parameters.function_parameters import (
        ...     FittedFunction, RadialBasisFunction,
        ... )
        >>> declaration = RadialBasisFunction([0.0], 1.0, name="response")
        >>> space = FunctionSpace("response", [Scalar("t")], [Scalar("value")])
        >>> fitted = FittedFunction(
        ...     declaration, space, lambda t: 2 * t, (2.0,),
        ... )
        >>> fitted(3.0)
        6.0
    """

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
        """Evaluate arguments through the retained backend-native callable.

        Args:
            *arguments: Concrete values in public argument order.
            **kwargs: Keyword arguments supported by the native callable.

        Returns:
            Native callable results; shape and value types follow ``space``
            and the selected backend.
        """
        return self.function(*arguments, **kwargs)

    def input_shape(self):
        """Return input dimensional metadata in public argument order."""
        return tuple(self.space.input_dimensions())

    def output_shape(self):
        """Return output dimensional metadata in public result order."""
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
