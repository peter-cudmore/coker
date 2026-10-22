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
from coker.dynamics.model import DynamicalSystem
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
    """Receive a reconstructed solution at each supported solver iteration.

    Subclass and override :meth:`__call__`, or supply a callable with the same
    signature. The default continues unconditionally. Iteration callbacks are
    supported by CasADi; other backends need not support them.

    Examples:
        Configure a callback without starting a solve:

        >>> from coker.dynamics import (
        ...     TranscriptionOptions, VariationalIterationCallback,
        ... )
        >>> class RecordCost(VariationalIterationCallback):
        ...     def __init__(self):
        ...         self.costs = []
        ...     def __call__(self, iterate, solution, **kwargs):
        ...         self.costs.append((iterate, solution.cost))
        ...         return True
        >>> recorder = RecordCost()
        >>> options = TranscriptionOptions(interation_callback=recorder)
        >>> options.interation_callback is recorder
        True
    """

    def __call__(
        self,
        iterate: int,
        solution: VariationalSolution,
        **kwargs,
    ) -> bool:
        """Handle one solver iteration and choose whether to continue.

        Args:
            iterate: Zero-based callback count for the current CasADi solve,
                not a physical time or a collocation index.
            solution: Current :class:`VariationalSolution`, with the same
                channel dimensions and parameter declarations as the problem.
                Evaluate its trajectories at scalar physical times on
                ``[0, solution.t_final]``. Its scalar ``cost`` and constraint
                residuals describe the current, possibly infeasible iterate.
            **kwargs: Optional solver-specific arguments. The current CasADi
                wrapper supplies none.

        Returns:
            bool: ``True`` to continue, ``False`` to request solver termination.
            A termination request is not a successful solve result.

        Examples:
            Use the result constructed by the problem example:

            .. testcode:: variational-problem-source

                from coker.dynamics import VariationalIterationCallback

                assert VariationalIterationCallback()(iterate=0, solution=solution)
        """

        return True


@dataclass
class TranscriptionOptions:
    """Control the collocation mesh and solve behaviour.

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
        verbose: Show CasADi/IPOPT solver output.
        optimiser_options: CasADi/IPOPT solver settings.
        initialise_near_guess: Run CasADi's feasibility initializer before
            optimization.
        enable_scaling: Scale CasADi decision variables, constraints and
            objective.
        interation_callback: Receive CasADi solver iterations.
        backend_options: Backend-specific solve policy. Use
            ``CasadiVariationalOptions`` to configure CasADi, including
            adaptive mesh refinement.

    Notes:
        Explicit ``backend_options`` replace the CasADi policy supplied by
        ``verbose``, ``optimiser_options``, ``initialise_near_guess``,
        ``enable_scaling`` and ``interation_callback``.

    Examples:
        >>> from coker.dynamics import TranscriptionOptions
        >>> options = TranscriptionOptions(
        ...     minimum_n_intervals=2, minimum_degree=3,
        ...     absolute_tolerance=1e-8, initialise_near_guess=False,
        ... )
        >>> (options.minimum_n_intervals, options.minimum_degree)
        (2, 3)

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
    def from_expression(cls, constraint: InequalityExpression) -> "ConstraintSpec":
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
    """Represent a backend-neutral dynamic optimisation problem.

    Args:
        loss: Scalar symbolic objective, or a callback ``loss(trajectory, p)``
            without controls and ``loss(trajectory, u, p)`` with controls.
            ``trajectory`` evaluates declared outputs as
            ``trajectory(t, p)`` or ``trajectory(t, u, p)`` respectively.
            ``t`` is physical time and ``p`` is the solver parameter vector.
        t_final: Fixed horizon in the model's time units, measured from zero,
            or a bounded scalar duration decision in those same units.
        system: Dynamic system to transcribe.
        control: Scalar control declarations in the system input's order.
            Their time coordinates are normalised by the solved horizon.
        parameters: Fitted parameter declarations and fixed numeric values in
            system parameter order. Function-valued parameters require
            declarations compatible with the system's parameter spaces.
        system_parameter_map: Optional numeric matrix of shape
            ``(system_parameter_width, solver_parameter_count)`` mapping the
            finite solver parameter vector into system parameters.
        quadratures: Builder-owned running-integral records. Use
            :meth:`VariationalProblemBuilder.integrate` rather than creating
            these internal records.
        transcription_options: Collocation and solve configuration.
        backend: Backend name used to create the solver; defaults to CasADi.
        path_constraints: Comparisons evaluated across the trajectory.
        terminal_constraints: Comparisons evaluated at the final physical time.
        initial_constraints: Comparisons evaluated at physical time zero.

    Examples:
        This example requires the CasADi extra. The initial state is a fitted
        scalar and the objective compares the final output against one.

        .. testcode:: variational-problem-source

            import numpy as np
            from coker import VectorSpace
            from coker.dynamics import VariationalProblem, create_autonomous_ode
            from coker.parameters import BoundedVariable

            system = create_autonomous_ode(
                x0=lambda p: p[0],
                xdot=lambda x, p: 0 * x,
                parameters=VectorSpace("p", 1),
                backend="casadi",
            )
            problem = VariationalProblem(
                loss=lambda trajectory, p: (trajectory(1.0, p) - 1.0) ** 2,
                t_final=1.0,
                system=system,
                parameters=[BoundedVariable("initial", 0.0, 2.0, guess=0.5)],
            )
            assert problem.horizon_decision is None
            assert problem.decision_declarations == []
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
    terminal_constraints: List[InequalityExpression] = field(default_factory=list)
    initial_constraints: List[InequalityExpression] = field(default_factory=list)

    @property
    def horizon_decision(self) -> Optional[BoundedVariable]:
        """Return the bounded duration declaration, or ``None`` if fixed.

        Returns:
            BoundedVariable | None: Horizon decision in physical time units.

        Examples:
            .. testcode:: variational-problem-source

                assert problem.horizon_decision is None
        """
        return self.t_final if isinstance(self.t_final, BoundedVariable) else None

    @property
    def decision_declarations(self) -> List[ParameterMixin]:
        """Return the horizon and controls in solver declaration order.

        Returns:
            list[ParameterMixin]: The bounded duration first, when present,
            followed by control declarations. Fitted system parameters are
            not included; this is not a flattened coefficient vector.

        Examples:
            .. testcode:: variational-problem-source

                assert problem.decision_declarations == []
        """
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
            ) = specialize_system_parameters(self.system, self.parameters or [])
        self.path_constraints = _normalize_constraints(self.path_constraints)
        self.terminal_constraints = _normalize_constraints(self.terminal_constraints)
        self.initial_constraints = _normalize_constraints(self.initial_constraints)
        system_space = self.system.parameters
        system_width = _space_size(system_space) if system_space is not None else 0
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
        """Create a solver for this problem.

        Args:
            backend: Optional backend name overriding :attr:`backend` for this
                solver. It does not change the stored problem configuration.

        Returns:
            VariationalSolver: Backend solver whose ``solve()`` method returns
            a :class:`VariationalSolution`. Construction does not solve.

        Examples:
            .. testcode:: variational-problem-source

                solver = problem.get_solver()
                assert callable(solver.solve)
        """
        from coker.backends import get_backend_by_name

        backend_name = backend or self.backend
        return get_backend_by_name(backend_name).create_variational_solver(self)

    def __call__(self) -> VariationalSolution:
        """Solve with the configured backend and return its trajectory result.

        Returns:
            VariationalSolution: Objective, reconstructed parameters and
            trajectory channels; see its class contract for evaluation.

        Raises:
            SolveFailure: The backend reports an unsuccessful solve.
                Configuration and native backend errors may propagate
                separately.

        Examples:
            .. testcode:: variational-problem-source

                solution = problem()
                assert solution.cost < 1e-6
                assert np.isclose(solution.parameters["initial"], 1.0, atol=1e-3)
        """
        return self.get_solver().solve()
