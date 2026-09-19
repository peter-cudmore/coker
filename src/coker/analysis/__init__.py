"""Generic local symbolic analysis for Coker dynamical systems."""

from __future__ import annotations

from importlib import import_module as _import_module

from .model import AnalysisResult, AnalysisStatus
from .symbolic import SymbolicSystem, UnsupportedSystemError, lower_system

__all__ = (
    "AnalysisResult",
    "AnalysisStatus",
    "SymbolicSystem",
    "UnsupportedSystemError",
    "analyse_controllability",
    "analyse_identifiability",
    "lower_system",
)


def __getattr__(name: str):
    if name == "analyse_controllability":
        return _import_module(
            ".controllability", __name__
        ).analyse_controllability
    if name == "analyse_identifiability":
        return _import_module(
            ".identifiability", __name__
        ).analyse_identifiability
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
