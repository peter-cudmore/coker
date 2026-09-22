"""Dependency-free public interfaces."""

import abc


class SolverParameters(metaclass=abc.ABCMeta):
    """Interface for backend-specific ODE solver configuration."""


class SymbolicCallable(metaclass=abc.ABCMeta):
    """Common call/lowering contract for symbolic callable graph values."""

    @abc.abstractmethod
    def __call__(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def lower(self):
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
