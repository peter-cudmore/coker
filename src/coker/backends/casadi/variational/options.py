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
    is solved by the CasADi backend.  Leave ``refinement_enabled`` false to
    retain the single-transcription solve path.
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
