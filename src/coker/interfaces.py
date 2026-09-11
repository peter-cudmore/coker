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
