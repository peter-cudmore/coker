"""Public Coker QP optimisation entry points.

The generic optimisation interface remains backend-neutral: CasADi may lower
broader nonlinear problems, while this embedded Coker implementation currently
accepts only QPs.
"""

from coker.backends.coker.qp_extract import (
    build_optimisation_problem,
    compile_qp_problem,
    extract_qp_program,
    validate_qp_problem,
)
from coker.backends.coker.qp_types import (
    CokerSolver,
    ExtractedQpProgram,
    OutputSlice,
    RuntimeQpProgram,
)

__all__ = [
    "CokerSolver",
    "ExtractedQpProgram",
    "OutputSlice",
    "RuntimeQpProgram",
    "build_optimisation_problem",
    "compile_qp_problem",
    "extract_qp_program",
    "validate_qp_problem",
]
