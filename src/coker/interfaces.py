"""Dependency-free public interfaces."""

import abc


class SolverParameters(metaclass=abc.ABCMeta):
    """Mark backend-specific ODE or independent residual-solver settings.

    This marker defines no fields, validation, copying or solver-selection
    policy. Use the concrete options type accepted by the selected backend.
    These settings are separate from lowering options, backend device/dtype
    configuration and mathematical-program optimiser options. Backends may
    use defaults for unrecognised parameter types.

    Examples:
        >>> from coker.interfaces import SolverParameters
        >>> from coker.backends.numpy import NumpySolverParameters
        >>> isinstance(NumpySolverParameters(), SolverParameters)
        True
    """


class SymbolicCallable(metaclass=abc.ABCMeta):
    """Extend the positional call and lowering interface for graph callables.

    Implementations define whether calls evaluate concrete values or record
    symbolic graph operations. This interface does not itself declare shapes,
    a backend, output packing or ownership. Shape declarations belong to
    ``FunctionSignatureValue``.

    Examples:
        >>> from coker import Scalar, function
        >>> from coker.interfaces import SymbolicCallable
        >>> increment = function([Scalar("x")], lambda x: x + 1)
        >>> isinstance(increment, SymbolicCallable)
        True
        >>> float(increment(2.0))
        3.0
    """

    @abc.abstractmethod
    def __call__(self, *args):
        """Evaluate or record a positional call.

        Args:
            *args: Inputs in the concrete callable's declared order.

        Returns:
            Any: Concrete results or symbolic values, with packing defined by
            the implementation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def lower(self):
        """Lower the callable using its implementation's configuration.

        Returns:
            Any: The implementation's lowered representation. This interface
            does not require a common handle type; ``Function.lower`` returns
            a ``LoweredFunction``.
        """
        raise NotImplementedError


class FunctionSignatureValue(metaclass=abc.ABCMeta):
    """Declare ordered logical shapes for a callable value.

    Scalar metadata uses ``Dimension(None)``, not an absent output ``None``
    slot. Callable-valued entries use ``FunctionSpace`` declarations.

    Examples:
        >>> from coker import Scalar, function
        >>> from coker.interfaces import FunctionSignatureValue
        >>> identity = function([Scalar("x")], lambda x: x)
        >>> isinstance(identity, FunctionSignatureValue)
        True
        >>> identity.input_shape()[0].is_scalar()
        True
        >>> identity.output_shape()[0].shape
        (1,)
    """

    @abc.abstractmethod
    def input_shape(self):
        """Return logical input dimensions in positional argument order.

        Returns:
            Sequence: ``Dimension`` or ``FunctionSpace`` declarations, with
            ``None`` for absent slots where supported by the implementation.
            These are declarations, not inferred shapes of runtime values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self):
        """Return logical output dimensions in result order.

        Returns:
            Sequence: ``Dimension`` or ``FunctionSpace`` declarations, with
            ``None`` for absent slots. Scalar entries are ``Dimension(None)``.
        """
        raise NotImplementedError
