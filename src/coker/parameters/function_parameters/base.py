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
    """Wrap a concrete callable reconstructed from fitted decisions.

    Args:
        specification: Original function-parameter declaration.
        space: Public function space, excluding concrete decision blocks.
        function: Backend callable to invoke without input or output conversion.
        parameters: Reconstructed concrete values in declaration order.
            Solver-produced values are tuples; numeric types depend on backend.

    Attributes:
        specification: Original declaration.
        space: Validated public function space.
        function: Concrete backend callable.
        parameters: Fitted concrete values, retaining their scalar or array shapes.

    Examples:
        >>> from coker import FunctionSpace, Scalar
        >>> from coker.parameters.function_parameters import (
        ...     FittedFunction, RadialBasisFunction)
        >>> declaration = RadialBasisFunction([0.0], 1.0, name="response")
        >>> space = FunctionSpace("response", [Scalar("x")], [Scalar("y")])
        >>> fitted = FittedFunction(declaration, space, lambda x: 2 * x, ([2, 0],))
        >>> fitted(3)
        6
        >>> fitted.specification.name
        'response'
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
        """Forward a call to the concrete backend callable.

        Args:
            *arguments: Public positional inputs in function-space order.
            **kwargs: Keywords accepted by the concrete callable.

        Returns:
            Any: The callable's result without conversion.

        Raises:
            Exception: Any exception raised by the concrete callable propagates.
        """
        return self.function(*arguments, **kwargs)

    def input_shape(self):
        """Return public input dimensions, excluding concrete decision blocks.

        Returns:
            tuple: Dimensions reported by the public function space.
        """
        return tuple(self.space.input_dimensions())

    def output_shape(self):
        """Return public output dimensions.

        Returns:
            tuple: Dimensions reported by the public function space.
        """
        return tuple(self.space.output_dimensions())


class FunctionParameter(ABC):
    """Extension interface for finite realisations of function decisions.

    Implement target validation, ordered concrete declarations and evaluation.
    Concrete blocks must be scalar or dense parameter declarations. Keep their
    names and order stable: builders use them to reconstruct fitted results.
    The default builder supports evaluation with one public argument; override
    it if the declaration needs a different argument-passing convention.
    """

    name: str | None

    @abstractmethod
    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        """Validate the public function signature.

        Args:
            target: Function space requested by the consuming builder.

        Returns:
            FunctionSpace: The compatible, validated target.

        Raises:
            TypeError: The target is not a supported function-space type.
            ValueError: Its input or output signature is incompatible.
        """

    @abstractmethod
    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        """Return the concrete decisions in reconstruction order.

        Returns:
            tuple[ParameterVariable, ...]: Named scalar or dense declarations.
                Evaluation receives fitted values in this same order.
        """

    @abstractmethod
    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        """Evaluate the declared function from structured concrete values.

        Args:
            parameters: Scalar or array values in concrete declaration order.
            argument: Public input value in the target's shape.

        Returns:
            Any: The value with the target output signature. Implementations
            must also support graph values when used by a tracing backend.
        """

    def build_function(self, target: FunctionSpace, backend: str | None) -> Function:
        """Build a Coker function with public inputs before concrete blocks.

        Args:
            target: Public function space to validate.
            backend: Backend used to trace the function, or ``None`` for the
                normal backend selection.

        Returns:
            Function: Function taking target arguments then concrete values.
                The default implementation passes only the first public
                argument to :meth:`evaluate`.

        Raises:
            TypeError: The target or a concrete declaration has an invalid type.
            ValueError: The target is incompatible with the declaration.
        """
        target = self.validate_target(target)
        declarations = self.list_concrete_parameters()
        parameter_spaces = [
            _concrete_parameter_space(declaration) for declaration in declarations
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
    raise TypeError("function parameter declarations must be scalar or dense variables")
