"""Typed trajectory requirement records retained for backend lowering."""

from dataclasses import dataclass
from typing import Any

from coker.algebra.kernel import Tracer


@dataclass(frozen=True)
class PathSite:
    pass


@dataclass(frozen=True)
class InitialSite:
    pass


@dataclass(frozen=True)
class TerminalSite:
    pass


_StateSignal = PathSite
_InputSignal = PathSite
_OutputSignal = TerminalSite


@dataclass(frozen=True)
class TrajectoryRequirement:
    expression: Tracer
    site: PathSite | InitialSite | TerminalSite


def normalize_trajectory_expression(expression: Tracer, site: Any):
    if not isinstance(expression, Tracer):
        raise TypeError("trajectory expression must be a tracer")
    return TrajectoryRequirement(expression, site)
