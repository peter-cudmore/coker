"""Public CasADi variational solver API."""

from . import transcription as _transcription
from .transcription import (
    CasadiVariationalSolver,
    ControlFactory,
    InterpolatingPoly,
    InterpolatingPolyCollection,
    SymbolicPoly,
    SymbolicPolyCollection,
    _create_solver,
    _derive_objective_scale,
    _derive_variable_scaling,
    _resolve_options,  # noqa: F401
)


def create_variational_solver(problem):
    _transcription._derive_objective_scale = _derive_objective_scale
    _transcription._derive_variable_scaling = _derive_variable_scaling
    _transcription._create_solver = _create_solver
    return _transcription.create_variational_solver(problem)


__all__ = [
    "CasadiVariationalSolver",
    "ControlFactory",
    "InterpolatingPoly",
    "InterpolatingPolyCollection",
    "SymbolicPoly",
    "SymbolicPolyCollection",
    "create_variational_solver",
]
