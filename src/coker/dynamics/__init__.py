"""Public dynamics, residual simulation, symbolic analysis and optimisation.

Import public residual and analysis contracts from ``coker.dynamics``.
``ResidualDynamicalSystem`` describes implicit numerical trajectories;
``to_residual_dynamical_system`` preserves semi-explicit models when converting
them to residual form. Numerical support depends on the selected backend.

``lower_system`` and ``lower_dae_system`` expose finite autonomous symbolic
representations. ``analyse_controllability``, ``analyse_observability`` and
``analyse_identifiability`` return typed rank evidence, not booleans or
simulation results. Their supported DAE domain is semi-explicit index one,
not general implicit residual systems. Inspect ``AnalysisStatus``, ``reason``
and nonvanishing ``rank_conditions`` before interpreting a result.
"""

from coker.dynamics.variables import (
    ConstantControlSolution,
    ConstantControlVariable,
    ControlLaw,
    ControlSolution,
    ControlVariable,
    LossFunction,
    PiecewiseConstantVariable,
    PiecewiseControlSolution,
    Solution,
    SpikeControlSolution,
    SpikeVariable,
)
from coker.dynamics.analysis import (
    AnalysisResult,
    AnalysisStatus,
    ControllabilityResult,
    IdentifiabilityResult,
    ObservabilityResult,
    SymbolicDAESystem,
    SymbolicSystem,
    UnsupportedSystemError,
    analyse_controllability,
    analyse_identifiability,
    analyse_observability,
    lower_dae_system,
    lower_system,
)
from coker.dynamics.system import (
    CompositionOperator,
    ProjectionSet,
    StructuredParameterProjection,
    create_autonomous_ode,
    create_control_system,
    create_dynamics_from_spec,
    direct_sum,
)

from coker.dynamics.model import DynamicsSpec, DynamicalSystem
from coker.dynamics.residual import (
    ResidualDynamicalSystem,
    to_residual_dynamical_system,
)
from coker.dynamics.variational.builder import VariationalProblemBuilder
from coker.dynamics.variational.problem import (
    TranscriptionOptions,
    VariationalIterationCallback,
    VariationalProblem,
)
from coker.dynamics.variational.solution import VariationalSolution

__all__ = [
    "AnalysisResult",
    "AnalysisStatus",
    "ControllabilityResult",
    "ConstantControlSolution",
    "ConstantControlVariable",
    "CompositionOperator",
    "ControlLaw",
    "ControlSolution",
    "ControlVariable",
    "DynamicalSystem",
    "DynamicsSpec",
    "ResidualDynamicalSystem",
    "to_residual_dynamical_system",
    "SymbolicDAESystem",
    "SymbolicSystem",
    "UnsupportedSystemError",
    "IdentifiabilityResult",
    "ObservabilityResult",
    "LossFunction",
    "PiecewiseConstantVariable",
    "PiecewiseControlSolution",
    "ProjectionSet",
    "Solution",
    "SpikeControlSolution",
    "SpikeVariable",
    "StructuredParameterProjection",
    "TranscriptionOptions",
    "VariationalProblem",
    "VariationalProblemBuilder",
    "VariationalSolution",
    "VariationalIterationCallback",
    "analyse_controllability",
    "analyse_identifiability",
    "analyse_observability",
    "create_autonomous_ode",
    "create_control_system",
    "create_dynamics_from_spec",
    "direct_sum",
    "lower_dae_system",
    "lower_system",
]
