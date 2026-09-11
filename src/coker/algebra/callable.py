"""Dependency-free symbolic callable contract."""

from abc import ABC, abstractmethod


class SymbolicCallable(ABC):
    """Common call/lowering contract for symbolic callable graph values."""

    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError

    @abstractmethod
    def lower(self):
        raise NotImplementedError
