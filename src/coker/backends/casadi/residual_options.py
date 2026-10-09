"""Public solver selection for CasADi residual DAE integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from coker.interfaces import SolverParameters


class CasadiResidualSolver(Enum):
    """Select the direct solver for independent residual DAEs."""

    IDAS = "idas"
    VARIATIONAL = "variational"


@dataclass(frozen=True)
class CasadiResidualSolverOptions(SolverParameters):
    """Configure independent residual DAE integration on CasADi.

    Args:
        solver: Select native SUNDIALS IDAS or the CasADi variational
            collocation solver.
        idas_options: CasADi IDAS options used when ``solver`` is ``IDAS``.
        variational_options: CasADi IPOPT options used when ``solver`` is
            ``VARIATIONAL``.

    Examples:
        Select direct collocation explicitly::

            options = CasadiResidualSolverOptions(
                solver=CasadiResidualSolver.VARIATIONAL
            )
    """

    solver: CasadiResidualSolver
    idas_options: Mapping[str, Any] = field(default_factory=dict)
    variational_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.solver, CasadiResidualSolver):
            raise TypeError("solver must be a CasadiResidualSolver")
        if not isinstance(self.idas_options, Mapping):
            raise TypeError("idas_options must be a mapping")
        if not isinstance(self.variational_options, Mapping):
            raise TypeError("variational_options must be a mapping")
