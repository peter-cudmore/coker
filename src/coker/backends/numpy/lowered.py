"""NumPy reusable-plan lowering handle."""

from collections.abc import Sequence

from coker.backends.evaluator import _cast_outputs
from coker.backends.lowered import LoweredFunction, LoweringCapabilities


class NumpyLoweredFunction(LoweredFunction):
    """Execute one immutable NumPy plan with a fresh per-call workspace."""

    def __init__(self, backend, function, plan):
        self._backend = backend
        self._function = function
        self._plan = plan

    @property
    def backend_name(self):
        return self._backend.name

    @property
    def signature(self):
        return self._function.signature

    @property
    def capabilities(self):
        return LoweringCapabilities(
            True, True, False, False, False, False, False, True
        )

    def execute(self, inputs: Sequence):
        workspace = self._plan.execute(inputs, self._backend)
        return tuple(
            _cast_outputs(
                self._function.output,
                self._function.tape,
                workspace,
                self._backend,
            )
        )
