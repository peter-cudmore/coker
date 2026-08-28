"""Host orchestration for Coker optimisation modules.

The wrapped solver owns host-side setup state only. Its compiled QP runtime
keeps coefficient, solver, and warm-start buffers in the runtime object.
"""

from __future__ import annotations

from typing import Any


class CokerModule:
    """Expose a Coker solver as a numerical module in a composed function.

    This wrapper deliberately forwards each invocation to the prebuilt solver;
    it does not reconstruct a QP or allocate a solver per call.
    """

    def __init__(self, solver: Any):
        self._solver = solver
        self.last_solve_info = None

    def __call__(self, *args):
        try:
            return self._solver(*args)
        finally:
            self.last_solve_info = getattr(
                self._solver, "last_solve_info", None
            )
