"""CasADi lowering handles retaining the native function when available."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import casadi as ca

if TYPE_CHECKING:
    from coker.algebra.kernel import Function
    from coker.backends.backend import Backend
    from coker.backends.lowered import FunctionSignature

import numpy as np

from coker.backends.lowered import LoweredFunction, LoweringCapabilities


class CasadiLoweredFunction(LoweredFunction):
    """Execute a native ``ca.Function`` or the required evaluate fallback."""

    def __init__(
        self,
        backend: "Backend",
        function: "Function",
        ca_function: ca.Function | None = None,
    ) -> None:
        self._backend = backend
        self._function = function
        self._ca_function = ca_function

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def signature(self) -> "FunctionSignature":
        return self._function.signature

    @property
    def capabilities(self) -> LoweringCapabilities:
        return LoweringCapabilities(
            eager_execution=True,
            symbolic_execution=True,
        )

    @property
    def ca_function(self) -> ca.Function | None:
        """Native CasADi function, or ``None`` for an evaluate fallback."""
        return self._ca_function

    def execute(self, inputs: Sequence[Any]) -> tuple[Any | None, ...]:
        if self._ca_function is None:
            return tuple(self._backend.evaluate(self._function, list(inputs)))
        dm_inputs = [
            self._backend.to_backend_array(
                np.asarray(value) if isinstance(value, list) else value
            )
            for value in inputs
            if value is not None
        ]
        values = self._ca_function(*dm_inputs)
        if not isinstance(values, (list, tuple)):
            values = [values]
        # Native execution is intentionally not converted through the
        # backend's public NumPy boundary.  CasADi callers need to retain
        # symbolic/numeric native values (and their sparse representation).
        return tuple(values)
