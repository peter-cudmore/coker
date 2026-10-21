"""Dependency-free public interfaces."""

import abc


class SolverParameters(metaclass=abc.ABCMeta):
    """Interface for backend-specific ODE solver configuration."""


class SymbolicCallable(metaclass=abc.ABCMeta):
    """Define symbolic calling and domain-specific lowering.

    Construct a callable through its public factory rather than this abstract
    class. A ``Function`` lowers to an executable handle; a mathematical
    program lowers to a solver-pinned ``Function``.

    Example:
        >>> from coker import Scalar, SymbolicCallable, function
        >>> square = function([Scalar("x")], lambda x: x * x)
        >>> isinstance(square, SymbolicCallable)
        True
        >>> assert square(3.) == 9.
    """

    @abc.abstractmethod
    def __call__(self, *args):
        """Evaluate positional arguments or record a call in an active trace.

        Args:
            *args: Values in the callable's declared argument order.

        Returns:
            Domain-specific values or symbolic results in declaration order.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def lower(self):
        """Return the owning domain's documented lowering representation."""
        raise NotImplementedError


class FunctionSignatureValue(metaclass=abc.ABCMeta):
    """Value that declares the shapes of a callable function."""

    @abc.abstractmethod
    def input_shape(self):
        """Return declared input shapes."""
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self):
        """Return declared output shapes."""
        raise NotImplementedError
