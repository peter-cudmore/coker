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
OutputShape: TypeAlias = Dimension | Scalar | VectorSpace | FunctionSpace | None


@dataclass(frozen=True)
class FunctionInputSpec:
    """Declare one named argument without discarding callable-space metadata.

    Args:
        name: Argument name in the native/imported signature.
        space: Named scalar, vector/tensor, or function argument space.
    """

    name: str
    space: InputSpace


@dataclass(frozen=True)
class FunctionOutputSpec:
    """Declare one ordered output slot.

    Args:
        name: Output name in the native/imported signature.
        shape: Numeric dimension or callable space. ``None`` retains an absent
            output slot rather than removing it from the public signature.
    """

    name: str
    shape: OutputShape


@dataclass(frozen=True)
class FunctionSignature:
    """Declare backend-independent ordered arguments and results.

    Args:
        inputs: Input declarations in positional call order.
        outputs: Output declarations in result order.
    """

    inputs: tuple[FunctionInputSpec, ...]
    outputs: tuple[FunctionOutputSpec, ...]


@dataclass(frozen=True)
class LoweringOptions:
    """Select an immutable lowering configuration and cache key.

    Args:
        backend: Registry name overriding the function's execution backend.
            ``None`` retains that backend; it does not select an ambient one.
    """

    backend: str | None = None


@dataclass(frozen=True)
class LoweringCapabilities:
    """Describe execution support, not ownership or thread-safety guarantees.

    Args:
        eager_execution: Support concrete numeric evaluation.
        symbolic_execution: Support the backend's native symbolic values.
        autograd: Preserve the backend's automatic differentiation graph.
    """

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
        """Return eager, symbolic and autograd execution support flags."""

    @abstractmethod
    def execute(self, inputs: Sequence[Any]) -> tuple[Any | None, ...]:
        """Execute the packed positional ABI.

        Args:
            inputs: One backend-compatible value per signature input.

        Returns:
            A tuple in signature output order, retaining absent ``None`` slots.
            Conversion, aliasing and differentiability depend on the backend.
        """

    def restore_public_outputs(
        self, outputs: Sequence[Any | None]
    ) -> tuple[Any | None, ...]:
        """Adapt native execution results to the public function boundary."""
        return tuple(outputs)

    def __call__(self, *inputs: Any) -> Any:
        """Execute positional inputs, unwrapping a single declared output."""
        outputs = self.execute(inputs)
        return outputs[0] if len(outputs) == 1 else outputs
