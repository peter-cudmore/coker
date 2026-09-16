"""CasADi-specific configuration for variational transcription."""

from dataclasses import dataclass, field
from typing import Optional

from coker.dynamics.variational.problem import (
    BackendTranscriptionOptions,
    VariationalIterationCallback,
)


@dataclass
class CasadiVariationalOptions(BackendTranscriptionOptions):
    """Configure CasADi variational transcription and adaptive refinement.

    These options apply only when a :class:`~coker.dynamics.VariationalProblem`
    is solved by the CasADi backend. Leave ``refinement_enabled`` false to
    retain the single-transcription solve path.

    Attributes:
        verbose: Enable IPOPT and CasADi solver output.
        optimiser_options: Options passed directly to the CasADi NLP solver.
        initialise_near_guess: Run the feasibility initialiser before the
            optimisation solve.
        enable_scaling: Normalize decision variables, constraints, and the
            objective before solving.
        interation_callback: Receive each IPOPT iterate as a variational
            solution. The field name preserves the existing public spelling.
        refinement_enabled: Re-solve using p-then-h mesh refinement until the
            local state defect meets ``mesh_tolerance``.
        mesh_tolerance: Maximum scaled relative state defect permitted in each
            collocation interval.
        maximum_degree: Largest local collocation polynomial degree selected by
            p-refinement before the interval is split.
        maximum_iterations: Maximum number of mesh-refinement iterations after
            the initial transcription solve.
        minimum_interval_duration: Smallest normalized interval width allowed
            when h-refinement bisects an interval.
    """

    verbose: bool = False
    optimiser_options: dict = field(default_factory=dict)
    initialise_near_guess: bool = True
    enable_scaling: bool = True
    interation_callback: Optional[VariationalIterationCallback] = None
    refinement_enabled: bool = False
    mesh_tolerance: float = 1e-6
    maximum_degree: int = 12
    maximum_iterations: int = 8
    minimum_interval_duration: float = 1e-8

    @property
    def backend_name(self) -> str:
        """Return the backend identifier that consumes these options."""
        return "casadi"

    def __post_init__(self) -> None:
        """Validate refinement bounds before the solver builds a mesh."""
        if self.mesh_tolerance <= 0:
            raise ValueError("mesh_tolerance must be positive")
        if self.maximum_degree < 1:
            raise ValueError("maximum_degree must be positive")
        if self.maximum_iterations < 0:
            raise ValueError("maximum_iterations must be non-negative")
        if self.minimum_interval_duration <= 0:
            raise ValueError("minimum_interval_duration must be positive")
