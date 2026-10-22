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
    """Declare one positional input.

    Args:
        name: Descriptive declaration name, not a keyword-dispatch key.
        space: Scalar, array or callable argument-space declaration.
    """

    name: str
    space: InputSpace


@dataclass(frozen=True)
class FunctionOutputSpec:
    """Declare one output slot in result order.

    Args:
        name: Descriptive output name.
        shape: Logical dimension or space declaration. ``None`` means an
            absent output slot, whereas ``Dimension(None)`` means a scalar.
    """

    name: str
    shape: OutputShape


@dataclass(frozen=True)
class FunctionSignature:
    """Describe the ordered public interface independently of a backend.

    Args:
        inputs: Input records in positional argument order.
        outputs: Output records in result order, including absent slots.

    Examples:
        >>> from coker import Dimension, Scalar
        >>> from coker.backends.lowered import (
        ...     FunctionInputSpec, FunctionOutputSpec, FunctionSignature)
        >>> signature = FunctionSignature(
        ...     (FunctionInputSpec("x", Scalar("x")),),
        ...     (FunctionOutputSpec("y", Dimension.scalar()),))
        >>> signature.inputs[0].name, signature.outputs[0].shape.is_scalar()
        ('x', True)

    The record is frozen, but referenced space declarations are not copied
    or made immutable.
    """

    inputs: tuple[FunctionInputSpec, ...]
    outputs: tuple[FunctionOutputSpec, ...]


@dataclass(frozen=True)
class LoweringOptions:
    """Select a backend for lowering without changing global selection.

    Args:
        backend: Registered backend name, or ``None`` to retain the function's
            backend. Device, dtype and solver configuration belong to their
            respective backend or solver options, not this record.

    Examples:
        >>> from coker.backends.lowered import LoweringOptions
        >>> LoweringOptions().backend is None
        True
    """

    backend: str | None = None


@dataclass(frozen=True)
class LoweringCapabilities:
    """Report execution mechanisms advertised by a lowered handle.

    Args:
        eager_execution: Whether the handle advertises eager execution.
        symbolic_execution: Whether it advertises symbolic execution.
        autograd: Whether it advertises automatic differentiation.

    These flags are metadata, not correctness, device, thread-safety or
    lifetime guarantees. In particular, JAX derivative correctness has known
    unresolved limitations (issue 99), even when ``autograd`` is true.
    """

    eager_execution: bool = False
    symbolic_execution: bool = False
    autograd: bool = False


class LoweredFunction(ABC):
    """Execute a backend-owned callable through an ordered packed ABI.

    Obtain handles from ``Function.lower()`` rather than instantiating this
    abstract class. ``execute`` always returns a tuple, including absent
    ``None`` slots. Calling the handle unwraps exactly one output, independent
    of whether the source function returned a single expression or a sequence.
    A public ``Function`` call additionally restores its own output convention.

    Concrete handles retain backend-specific execution state and may retain
    imported callables, modules and closure values. This interface promises
    neither a deep copy nor ownership transfer, resource release, concurrent
    execution safety or independence from changes to retained native objects.
    Consult the concrete backend for its lifetime and configuration contract.

    Examples:
        >>> import numpy as np
        >>> from coker import Scalar, function
        >>> twice = function([Scalar("x")], lambda x: 2 * x)
        >>> handle = twice.lower()
        >>> packed = handle.execute((3.0,))
        >>> len(packed), bool(np.allclose(packed[0], 6.0))
        (1, True)
        >>> bool(np.allclose(handle(4.0), 8.0))
        True
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """str: Registered name of the backend that created this handle."""

    @property
    @abstractmethod
    def signature(self) -> FunctionSignature:
        """FunctionSignature: Ordered declarations accepted by ``execute``."""

    @property
    @abstractmethod
    def capabilities(self) -> LoweringCapabilities:
        """LoweringCapabilities: Advertised execution mechanisms, not guarantees."""

    @abstractmethod
    def execute(self, inputs: Sequence[Any]) -> tuple[Any | None, ...]:
        """Execute one packed argument sequence.

        Args:
            inputs: Values in ``signature.inputs`` order, using the concrete
                backend's accepted representations.

        Returns:
            tuple: Native results in ``signature.outputs`` order, retaining
            absent ``None`` slots, including for zero or one output.
        """

    def restore_public_outputs(
        self, outputs: Sequence[Any | None]
    ) -> tuple[Any | None, ...]:
        """Adapt native results for a public ``Function`` call.

        Args:
            outputs: Ordered packed results from ``execute``.

        Returns:
            tuple: Public result representations. The base implementation
            only converts the sequence to a tuple; backends may convert values.
            This hook does not unwrap a single output.
        """
        return tuple(outputs)

    def __call__(self, *inputs: Any) -> Any:
        """Execute separate positional arguments and unwrap exactly one result.

        Args:
            *inputs: Values in signature order.

        Returns:
            Any: The native value for one result, otherwise the packed tuple.
            Unlike a public ``Function`` call, this does not invoke
            ``restore_public_outputs``.
        """
        outputs = self.execute(inputs)
        return outputs[0] if len(outputs) == 1 else outputs
