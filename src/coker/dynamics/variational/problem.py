from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace
from coker.algebra.function import Function, InequalityExpression, function
from coker.algebra.graph import Tracer
from coker.algebra.ops import Noop
from coker.dynamics.variables import ControlVariable, LossFunction
from coker.parameters import (
    BoundedVariable,
    ParameterMixin,
    ParameterVariable,
)
from coker.dynamics.system import DynamicalSystem
from coker.dynamics.variational.solution import VariationalSolution


def _space_size(
    space: Scalar | VectorSpace | FunctionSpace | tuple[object, ...],
) -> int:
    """Return the flattened scalar width of a finite parameter space."""
    if isinstance(space, tuple):
        return sum(_space_size(element) for element in space)
    if isinstance(space, FunctionSpace):
        raise TypeError(
            "Function-valued parameters must be specialized by "
            "VariationalProblemBuilder"
        )
    assert isinstance(space, (Scalar, VectorSpace))
    return space.size


class VariationalIterationCallback:
    """Extension hook receiving numerical variational solver iterates."""

    def __call__(
        self,
        iterate: int,
        solution: VariationalSolution,
        **kwargs,
    ) -> bool:
        """Handle callbacks at the end of each optimisation step.

        Args:
            iterate: Solver iteration index.
            solution: Current candidate; this is not a success certificate.
            **kwargs: Solver-specific arguments.

        Returns:
            True if the solver should continue.

        The shapes of each argument should line up with the
        `VariationalProblem` class attributes.
        """

        return True


@dataclass
class TranscriptionOptions:
    """Control the collocation mesh and solve behavior.

    Attributes:
        minimum_n_intervals: Minimum number of intervals in the initial mesh.
        minimum_degree: Collocation polynomial degree for each initial
            interval.
        absolute_tolerance: Default permitted absolute residual for
            transcription equality constraints.
        segment_defect_tolerance: Reference tolerance retained with physical
            post-solve segment-defect diagnostics. ``None`` uses
            ``absolute_tolerance``; it does not add NLP constraints.
        derivative_defect_tolerance: Override for collocation derivative
            defects. ``None`` uses ``absolute_tolerance``; zero enforces
            equality.
        verbose: Show CasADi/IPOPT solver output when no explicit backend
            options are supplied.
        optimiser_options: CasADi/IPOPT solver settings when no explicit
            backend options are supplied.
        initialise_near_guess: Run CasADi's feasibility initializer before
            optimization when no explicit backend options are supplied.
        enable_scaling: Scale CasADi decision variables, constraints, and
            objective when no explicit backend options are supplied.
        interation_callback: Receive CasADi solver iterations when no explicit
            backend options are supplied.
        backend_options: Backend-specific solve policy. Use
            ``CasadiVariationalOptions`` to configure CasADi, including
            adaptive mesh refinement.

    """

    minimum_n_intervals: int = 4
    minimum_degree: int = 7
    absolute_tolerance: float = 1e-12
    segment_defect_tolerance: Optional[float] = None
    derivative_defect_tolerance: Optional[float] = None
    verbose: bool = False
    optimiser_options: dict = field(default_factory=dict)
    initialise_near_guess: bool = True
    enable_scaling: bool = True
    interation_callback: Optional[VariationalIterationCallback] = None
    backend_options: Optional[object] = None


@dataclass(frozen=True)
class ConstraintSpec:
    """Backend-neutral normalized comparison constraint."""

    residual: Function | Tracer
    lower_bound: object
    upper_bound: object

    @classmethod
    def from_expression(
        cls, constraint: InequalityExpression
    ) -> "ConstraintSpec":
        """Normalize a comparison into residual and lower/upper bounds.

        Args:
            constraint: An algebraic ``InequalityExpression``.

        Returns:
            A backend-neutral constraint record.
        """
        return cls(
            residual=constraint.value,
            lower_bound=constraint.lower,
            upper_bound=constraint.upper,
        )


def _normalize_constraints(constraints):
    return [
        (
            item
            if isinstance(item, ConstraintSpec)
            else ConstraintSpec.from_expression(item)
        )
        for item in constraints
    ]


@dataclass(frozen=True)
class QuadratureSpec:
    """A builder-owned running integral channel."""

    integrand: Tracer
    initial_state: float = 0.0
    channel: int = 0


@dataclass
class VariationalProblem:
    """Describe a variational objective, declarations, and transcription.

    Prefer ``VariationalProblemBuilder`` for symbolic objectives and
    constraints. The direct constructor also accepts a loss callback whose
    arguments are ``(trajectory, parameters)`` without controls or
    ``(trajectory, control_function, parameters)`` with controls.

    Args:
        loss: Scalar loss callback, traced ``Function``, or builder tracer.
        t_final: Fixed physical duration or a bounded duration decision.
        system: Traced model defining initial conditions and outputs.
        control: Optional ordered scalar control declarations.
        parameters: Optional parameter decisions. Finite vector spaces use
            scalar declarations; heterogeneous tuples are specialized from
            their positional scalar/vector/function declarations.
        system_parameter_map: Matrix of shape ``(system_width,
            declaration_width)`` mapping finite solver parameters to system
            parameters.
        quadratures: Builder-owned running-integral records.
        transcription_options: Mesh and solve configuration.
        backend: Variational solver backend, default ``"casadi"``.
        path_constraints: Comparisons enforced at transcription path nodes.
        terminal_constraints: Comparisons enforced at terminal time.
        initial_constraints: Comparisons enforced at initial time.

    Construction traces/normalizes declarations but does not solve.
    CasADi supports collocation with controls, DAEs, quadratures, and free
    duration. PyTorch supports fixed-duration ODE fitting with parameter
    bounds and quadratures, not declared controls, DAEs, or comparisons.
    NumPy, JAX, and SymPy do not supply a variational solver.
    """

    loss: LossFunction | Tracer
    t_final: float | BoundedVariable
    system: DynamicalSystem
    control: Optional[List[ControlVariable]] = None
    parameters: Optional[List[ParameterVariable]] = None
    system_parameter_map: Optional[np.ndarray] = None
    parameter_layout: Optional[object] = field(default=None, init=False)
    quadratures: List[QuadratureSpec] = field(default_factory=list)
    transcription_options: TranscriptionOptions = field(
        default_factory=TranscriptionOptions
    )
    backend: Optional[str] = "casadi"
    path_constraints: List[InequalityExpression] = field(default_factory=list)
    terminal_constraints: List[InequalityExpression] = field(
        default_factory=list
    )
    initial_constraints: List[InequalityExpression] = field(
        default_factory=list
    )

    @property
    def horizon_decision(self) -> Optional[BoundedVariable]:
        """Return the duration declaration when duration is optimized."""
        return (
            self.t_final if isinstance(self.t_final, BoundedVariable) else None
        )

    @property
    def decision_declarations(self) -> List[ParameterMixin]:
        """Return horizon and control declarations in decision order."""
        decisions: List[ParameterMixin] = []
        if self.horizon_decision is not None:
            decisions.append(self.horizon_decision)
        decisions.extend(self.control or [])
        return decisions

    def __post_init__(self):
        if isinstance(self.system.parameters, tuple):
            from coker.dynamics.variational.function_binding import (
                specialize_system_parameters,
            )

            (
                self.system,
                self.parameters,
                self.parameter_layout,
            ) = specialize_system_parameters(
                self.system, self.parameters or []
            )
        self.path_constraints = _normalize_constraints(self.path_constraints)
        self.terminal_constraints = _normalize_constraints(
            self.terminal_constraints
        )
        self.initial_constraints = _normalize_constraints(
            self.initial_constraints
        )
        system_space = self.system.parameters
        system_width = (
            _space_size(system_space) if system_space is not None else 0
        )
        declaration_width = len(self.parameters or [])
        if self.system_parameter_map is not None:
            expected_shape = (system_width, declaration_width)
            if expected_shape != self.system_parameter_map.shape:
                raise ValueError(
                    "Parameter map is invalid. Expected an "
                    f"{expected_shape} matrix, but got "
                    f"{self.system_parameter_map.shape}."
                )
        elif self.parameters is not None and system_width != declaration_width:
            # Heterogeneous spaces are bound by the builder before reaching
            # the backend; at this point a normal numeric vector is expected.
            raise ValueError(
                "Number of parameters does not match: expected "
                f"{system_width} but got {declaration_width}. Please provide "
                "a parameter map or specify the same number of parameters."
            )
        if self.control is not None:
            assert self.system.inputs is not Noop()

        if not isinstance(self.loss, (Tracer, Function)):
            solution_space = self.system.output_as_function_space()
            parameter_space = VectorSpace("p", len(self.parameters or []))
            if self.parameters:
                solution_space.arguments[-1] = parameter_space
            loss_arguments = [solution_space]
            if self.control:
                loss_arguments.append(self.system.inputs)
            loss_arguments.append(parameter_space)
            self.loss = function(
                arguments=loss_arguments,
                implementation=self.loss,
            )

    def get_solver(self, backend: Optional[str] = None):
        """Create a backend solver for this problem.

        Args:
            backend: Optional override of the problem's solver backend.

        Returns:
            A solver exposing ``parameters`` (free scalar fixing names) and
            ``solve(**fixed_parameters)``. Retain this object for repeated
            solves; each call here creates a new solver.
        """
        from coker.backends import get_backend_by_name

        backend_name = self.backend if backend is None else backend
        return get_backend_by_name(backend_name).create_variational_solver(
            self
        )

    def __call__(self) -> VariationalSolution:
        """Create the configured solver and solve with no fixed overrides.

        Returns:
            A ``VariationalSolution`` carrying solve metadata.

        Raises:
            SolveFailure: The backend reports numerical solve failure.
            NotImplementedError: The backend does not support the problem.
        """
        return self.get_solver().solve()
