"""Dynamical-system modelling and variational optimisation API."""

from coker.dynamics.variables import (
    BoundedVariable,
    BoundVector,
    Constant,
    ConstantControlSolution,
    ConstantControlVariable,
    DenseTensorVariable,
    ControlLaw,
    ControlSolution,
    ControlVariable,
    LossFunction,
    ParameterMixin,
    ParameterVariable,
    PiecewiseConstantVariable,
    PiecewiseControlSolution,
    Solution,
    SpikeControlSolution,
    SpikeVariable,
    UnboundedVariable,
    ValueType,
)
from coker.dynamics.function_parameters import (
    FittedFunction,
    FunctionParameter,
    MonotonePiecewiseLinear,
    Perceptron,
    RadialBasisFunction,
)
from coker.dynamics.analysis import (
    AnalysisResult,
    AnalysisStatus,
    SymbolicDAESystem,
    SymbolicSystem,
    UnsupportedSystemError,
    analyse_controllability,
    analyse_identifiability,
    lower_dae_system,
    lower_system,
)
from coker.dynamics.system import create_autonomous_ode, direct_sum
from coker.interfaces import SolverParameters

from coker.dynamics.model import DynamicsSpec, DynamicalSystem
from coker.dynamics.transcription.collocation import (
    InterpolatingPoly,
    evaluate_legendre_polynomial,
    expand_coefficients,
    generate_discritisation_operators,
    legendre_coefficient,
    lgr_points,
)
from coker.dynamics.transcription.intervals import (
    split_at_non_differentiable_points,
)
from coker.dynamics.variational.builder import VariationalProblemBuilder
from coker.dynamics.variational.polynomials import InterpolatingPolyCollection
from coker.dynamics.variational.problem import (
    ConstraintSpec,
    QuadratureSpec,
    TranscriptionOptions,
    VariationalIterationCallback,
    VariationalProblem,
)
from coker.dynamics.variational.solution import VariationalSolution
from coker.toolkits.codesign.optimisation import SolveFailure, SolveInfo

__all__ = [
    "AnalysisResult",
    "AnalysisStatus",
    "BoundedVariable",
    "BoundVector",
    "Constant",
    "ConstantControlSolution",
    "ConstantControlVariable",
    "ConstraintSpec",
    "ControlLaw",
    "ControlSolution",
    "ControlVariable",
    "DynamicalSystem",
    "DynamicsSpec",
    "SymbolicDAESystem",
    "SymbolicSystem",
    "UnsupportedSystemError",
    "DenseTensorVariable",
    "InterpolatingPoly",
    "InterpolatingPolyCollection",
    "LossFunction",
    "FittedFunction",
    "FunctionParameter",
    "MonotonePiecewiseLinear",
    "Perceptron",
    "RadialBasisFunction",
    "ParameterMixin",
    "ParameterVariable",
    "PiecewiseConstantVariable",
    "PiecewiseControlSolution",
    "QuadratureSpec",
    "Solution",
    "SolverParameters",
    "SpikeControlSolution",
    "SpikeVariable",
    "TranscriptionOptions",
    "UnboundedVariable",
    "ValueType",
    "VariationalProblem",
    "VariationalProblemBuilder",
    "VariationalSolution",
    "VariationalIterationCallback",
    "SolveFailure",
    "SolveInfo",
    "analyse_controllability",
    "analyse_identifiability",
    "create_autonomous_ode",
    "direct_sum",
    "evaluate_legendre_polynomial",
    "expand_coefficients",
    "generate_discritisation_operators",
    "legendre_coefficient",
    "lgr_points",
    "split_at_non_differentiable_points",
    "lower_dae_system",
    "lower_system",
]
