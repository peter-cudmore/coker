"""Common executable handles produced by backend lowering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)

if TYPE_CHECKING:
    from coker.algebra.kernel import Function
    from coker.backends.backend import Backend


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
    """Declared execution and lifecycle properties of a lowered handle."""

    eager_execution: bool
    symbolic_execution: bool
    autograd: bool
    serializable_artifact: bool
    caller_owned_workspace: bool
    supports_prepare: bool
    supports_module_adapter: bool
    thread_safe: bool


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

    def prepare(self) -> None:
        """Perform optional deferred setup; eager handles need no work."""

    def close(self) -> None:
        """Release optional host resources without touching caller values."""


class EvaluatedLoweredFunction(LoweredFunction):
    """Generic handle without a reusable compiled representation."""

    def __init__(
        self,
        backend: "Backend",
        function: "Function",
        signature: FunctionSignature,
    ) -> None:
        self._backend = backend
        self._function = function
        self._signature = signature

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def signature(self) -> FunctionSignature:
        return self._signature

    @property
    def capabilities(self) -> LoweringCapabilities:
        return LoweringCapabilities(
            True, True, False, False, False, False, False, True
        )

    def execute(self, inputs: Sequence[Any]) -> tuple[Any | None, ...]:
        from coker.backends.evaluator import evaluate_inner

        workspace: dict[int, Any] = {}
        return tuple(
            evaluate_inner(
                self._function.tape,
                list(inputs),
                self._function.output,
                self._backend,
                workspace,
            )
        )
