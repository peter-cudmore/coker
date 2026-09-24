"""General parameter and decision declarations."""

from .dense import BoundVector, DenseTensorVariable
from .scalar import (
    BoundedVariable,
    Constant,
    ParameterMixin,
    UnboundedVariable,
    ValueType,
)

ParameterVariable = (
    BoundedVariable
    | UnboundedVariable
    | BoundVector
    | DenseTensorVariable
    | Constant
)

__all__ = [
    "BoundedVariable",
    "BoundVector",
    "Constant",
    "DenseTensorVariable",
    "ParameterMixin",
    "ParameterVariable",
    "UnboundedVariable",
    "ValueType",
]
