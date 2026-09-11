"""NumPy reusable-plan lowering handle."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coker.algebra.kernel import Function
    from coker.backends.backend import Backend
    from coker.backends.evaluator import CompiledPlan
    from coker.backends.lowered import FunctionSignature

from coker.backends.evaluator import _cast_outputs
from coker.backends.lowered import LoweredFunction, LoweringCapabilities


class NumpyLoweredFunction(LoweredFunction):
    """Execute one immutable NumPy plan with a fresh per-call workspace."""

    def __init__(
        self,
        backend: "Backend",
        function: "Function",
        plan: "CompiledPlan",
    ) -> None:
        self._backend = backend
        self._function = function
        self._plan = plan

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def signature(self) -> "FunctionSignature":
        return self._function.signature

    @property
    def capabilities(self) -> LoweringCapabilities:
        return LoweringCapabilities(
            True, True, False, False, False, False, False, True
        )

    def execute(self, inputs: Sequence[Any]) -> tuple[Any | None, ...]:
        workspace = self._plan.execute(inputs, self._backend)
        return tuple(
            _cast_outputs(
                self._function.output,
                self._function.tape,
                workspace,
                self._backend,
            )
        )
