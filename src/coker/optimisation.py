from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SolveInfo:
    backend: str
    solver: str
    success: bool
    return_status: str
    unified_return_status: str | None = None
    iteration_count: int | None = None


class SolveFailure(RuntimeError):
    def __init__(self, message: str, solve_info: SolveInfo):
        super().__init__(message)
        self.solve_info = solve_info


def solve_info_from_casadi_stats(
    stats: Mapping[str, Any], *, solver: str = "ipopt"
) -> SolveInfo:
    iteration_count = stats.get("iter_count")
    if iteration_count is not None:
        iteration_count = int(iteration_count)

    unified_status = stats.get("unified_return_status")
    if unified_status is not None:
        unified_status = str(unified_status)

    return SolveInfo(
        backend="casadi",
        solver=solver,
        success=bool(stats.get("success", False)),
        return_status=str(stats.get("return_status", "unknown")),
        unified_return_status=unified_status,
        iteration_count=iteration_count,
    )
