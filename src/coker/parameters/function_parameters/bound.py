"""Function parameters formed by binding selected Coker function inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.function import Function, function
from coker.parameters import (
    BoundVector,
    BoundedVariable,
    DenseTensorVariable,
    ParameterVariable,
    UnboundedVariable,
)

from .base import FunctionParameter, _concrete_parameter_space


_ConcreteParameter = (
    BoundedVariable | UnboundedVariable | BoundVector | DenseTensorVariable
)
_BindingInput = (
    Mapping[int, _ConcreteParameter] | Sequence[tuple[int, _ConcreteParameter]]
)


class BoundFunctionParameter(FunctionParameter):
    """Bind selected Coker function inputs to solver decision declarations.

    Args:
        function: Source function whose selected inputs are solver decisions.
        bindings: Mapping or sequence of ``(input_index, declaration)`` pairs.
        name: Public function-parameter name. Defaults to ``function.name``.

    Every unbound source input remains a public target input. The target output
    space remains identical to the source function output space.
    """

    def __init__(
        self,
        function: Function,
        bindings: _BindingInput,
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(function, Function):
            raise TypeError("function must be a Coker Function")

        self.function = function
        self.bindings = self._normalise_bindings(bindings)
        self.parameters = tuple(
            declaration for _, declaration in self.bindings
        )
        self.name = function.name if name is None else name
        self._unbound_indices = tuple(
            index
            for index in range(len(function.input_spaces()))
            if index not in {index for index, _ in self.bindings}
        )

    def _normalise_bindings(
        self, bindings: _BindingInput
    ) -> tuple[tuple[int, _ConcreteParameter], ...]:
        if isinstance(bindings, Mapping):
            values = tuple(bindings.items())
        elif isinstance(bindings, Sequence) and not isinstance(
            bindings, (str, bytes)
        ):
            values = tuple(bindings)
        else:
            raise TypeError(
                "bindings must be a mapping or sequence of "
                "(input index, parameter declaration) pairs"
            )
        if not values:
            raise ValueError("bindings must not be empty")

        input_shapes = self.function.input_shape()
        normalised = []
        seen_indices = set()
        for binding in values:
            if (
                not isinstance(binding, Sequence)
                or isinstance(binding, (str, bytes))
                or len(binding) != 2
            ):
                raise TypeError(
                    "each binding must be an "
                    "(input index, parameter declaration) pair"
                )
            index, declaration = binding
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("binding input index must be an integer")
            if index < 0 or index >= len(input_shapes):
                raise ValueError(
                    f"binding input index {index} is outside the function "
                    f"signature with {len(input_shapes)} inputs"
                )
            if index in seen_indices:
                raise ValueError(
                    f"duplicate binding for function input {index}"
                )
            if not isinstance(
                declaration,
                (
                    BoundedVariable,
                    UnboundedVariable,
                    BoundVector,
                    DenseTensorVariable,
                ),
            ):
                raise TypeError(
                    "binding declarations must be scalar or dense parameter "
                    "declarations"
                )

            declaration_space = _concrete_parameter_space(declaration)
            declaration_dimension = _space_dimension(declaration_space)
            if declaration_dimension != input_shapes[index]:
                raise ValueError(
                    f"binding declaration for function input {index} has "
                    "an incompatible space"
                )
            seen_indices.add(index)
            normalised.append((index, declaration))

        return tuple(sorted(normalised, key=lambda binding: binding[0]))

    def validate_target(self, target: FunctionSpace) -> FunctionSpace:
        """Validate the function space formed by the unbound source inputs."""
        if not isinstance(target, FunctionSpace):
            raise TypeError("target must be a FunctionSpace")

        expected_inputs = tuple(
            self.function.input_shape()[index]
            for index in self._unbound_indices
        )
        if tuple(target.input_dimensions()) != expected_inputs:
            raise ValueError(
                "target inputs must match the unbound function inputs"
            )
        if tuple(target.output_dimensions()) != tuple(
            self.function.output_shape()
        ):
            raise ValueError("target output must match the function output")
        return target

    def list_concrete_parameters(self) -> tuple[ParameterVariable, ...]:
        """Return declarations bound to the source function inputs."""
        return self.parameters

    def evaluate(self, parameters: Sequence[Any], argument: Any) -> Any:
        """Evaluate the source function with bound and public inputs."""
        return self._evaluate_with_arguments(
            parameters, self._normalise_public_argument(argument)
        )

    def build_function(
        self, target: FunctionSpace, backend: str | None
    ) -> Function:
        """Build the public target followed by bound concrete declarations."""
        target = self.validate_target(target)
        parameter_spaces = [
            _concrete_parameter_space(declaration)
            for declaration in self.parameters
        ]
        argument_count = len(target.arguments)

        def implementation(*arguments: Any) -> Any:
            return self._evaluate_with_arguments(
                arguments[argument_count:], arguments[:argument_count]
            )

        return function(
            [*target.arguments, *parameter_spaces],
            implementation,
            backend=backend,
            name=self.name,
        )

    def _normalise_public_argument(self, argument: Any) -> tuple[Any, ...]:
        if not self._unbound_indices:
            if argument is not None:
                raise ValueError("a fully bound function has no public inputs")
            return ()
        if len(self._unbound_indices) == 1:
            return (argument,)
        if not isinstance(argument, Sequence) or isinstance(
            argument, (str, bytes)
        ):
            raise TypeError(
                "a function with multiple public inputs requires an argument "
                "sequence"
            )
        if len(argument) != len(self._unbound_indices):
            raise ValueError(
                "public argument count does not match unbound function inputs"
            )
        return tuple(argument)

    def _evaluate_with_arguments(
        self, parameters: Sequence[Any], public_arguments: Sequence[Any]
    ) -> Any:
        if len(parameters) != len(self.bindings):
            raise ValueError(
                "parameter count does not match bound function declarations"
            )
        if len(public_arguments) != len(self._unbound_indices):
            raise ValueError(
                "public argument count does not match unbound function inputs"
            )

        bound_values = {
            index: value
            for (index, _), value in zip(self.bindings, parameters)
        }
        public_values = iter(public_arguments)
        return self.function(
            *(
                (
                    bound_values[index]
                    if index in bound_values
                    else next(public_values)
                )
                for index in range(len(self.function.input_spaces()))
            )
        )


def _space_dimension(space: Scalar | VectorSpace) -> Dimension:
    """Return the graph dimension represented by one concrete declaration."""
    return (
        Dimension.scalar()
        if isinstance(space, Scalar)
        else Dimension(space.dimension)
    )
