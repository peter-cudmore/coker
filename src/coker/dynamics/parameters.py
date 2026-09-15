"""Parameter declarations for dynamical systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace

ParameterElement = Scalar | VectorSpace | FunctionSpace


@dataclass(frozen=True)
class ParameterSpace:
    """Ordered heterogeneous parameter declarations.

    Elements retain their declaration order.  Function-valued elements are
    represented by a :class:`FunctionSpace`; finite elements are scalars or
    vector spaces.  The order is the public positional ``p[index]`` order
    used by model callbacks.
    """

    elements: Tuple[ParameterElement, ...]

    def __init__(self, elements: Iterable[ParameterElement]):
        values = tuple(elements)
        if not all(isinstance(v, (Scalar, VectorSpace, FunctionSpace)) for v in values):
            raise TypeError("parameters must contain Scalar, VectorSpace, or FunctionSpace elements")
        object.__setattr__(self, "elements", values)

    def __iter__(self) -> Iterator[ParameterElement]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    @property
    def heterogeneous(self) -> bool:
        return any(isinstance(v, FunctionSpace) for v in self.elements)

    @property
    def numeric_dimension(self) -> int:
        return sum(v.size if isinstance(v, VectorSpace) else 1 for v in self.elements if not isinstance(v, FunctionSpace))

    @property
    def dimension(self) -> int:
        """Finite packed size, retained for compatibility with numeric callers."""
        return self.numeric_dimension

    def as_tuple(self) -> Tuple[ParameterElement, ...]:
        return self.elements


def normalize_parameters(parameters):
    """Normalize a positional declaration sequence without changing legacy values."""
    if isinstance(parameters, ParameterSpace):
        return parameters
    if isinstance(parameters, (list, tuple)):
        return ParameterSpace(parameters)
    return parameters
