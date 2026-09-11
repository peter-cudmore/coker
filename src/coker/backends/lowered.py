"""Common executable handles produced by backend lowering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)


InputSpace: TypeAlias = Scalar | VectorSpace | FunctionSpace
OutputShape: TypeAlias = (
    Dimension | Scalar | VectorSpace | FunctionSpace | None
)


@dataclass(frozen=True)
class FunctionInputSpec:
    """One canonical input, preserving its declaration name and space."""

    name: str
    space: InputSpace


@dataclass(frozen=True)
class FunctionOutputSpec:
    """One canonical output, preserving its declaration order and shape."""

    name: str
    shape: OutputShape


@dataclass(frozen=True)
class FunctionSignature:
    """Backend-independent ordered input and output declaration."""

    inputs: tuple[FunctionInputSpec, ...]
    outputs: tuple[FunctionOutputSpec, ...]


@dataclass(frozen=True)
class LoweringOptions:
    """Immutable options selecting a backend lowering configuration."""

    backend: str | None = None


@dataclass(frozen=True)
class LoweringCapabilities:
    """Execution semantics exposed by a lowered handle."""

    eager_execution: bool = False
    symbolic_execution: bool = False
    autograd: bool = False


class LoweredFunction(ABC):
    """Backend-specific executable with a common packed execution ABI."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Registered name of the backend that created this handle."""

    @property
    @abstractmethod
    def signature(self) -> FunctionSignature:
        """Ordered input and output declaration accepted by :meth:`execute`."""

    @property
    @abstractmethod
    def capabilities(self) -> LoweringCapabilities:
        """Execution, lifecycle, and sharing guarantees for this handle."""

    @abstractmethod
    def execute(self, inputs: Sequence[Any]) -> tuple[Any | None, ...]:
        """Execute ordered inputs and return ordered outputs as a tuple."""

    def __call__(self, *inputs: Any) -> Any:
        """Execute positional inputs, unwrapping a single declared output."""
        outputs = self.execute(inputs)
        return outputs[0] if len(outputs) == 1 else outputs
