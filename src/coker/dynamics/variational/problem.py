from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from coker.algebra.kernel import (
    Function,
    InequalityExpression,
    Noop,
    Tracer,
    VectorSpace,
    function,
)
from coker.dynamics.controls import (
    BoundedVariable,
    ControlVariable,
    LossFunction,
    ParameterMixin,
    ParameterVariable,
)
from coker.dynamics.system import DynamicalSystem
from coker.dynamics.variational.solution import VariationalSolution


class VariationalIterationCallback:

    def __call__(
        self,
        iterate: int,
        solution: VariationalSolution,
        **kwargs,
    ) -> bool:
        """Handle callbacks at the end of each optimisation step.

        Args:
            **kwargs: Solver-specific arguments.

        Returns:
            True if the solver should continue.

        The shapes of each argument should line up with the
        `VariationalProblem` class attributes.
        """

        return True


@dataclass
class TranscriptionOptions:
    minimum_n_intervals: int = 4
    minimum_degree: int = 7
    absolute_tolerance: float = 1e-12
    verbose: bool = False
    optimiser_options: dict = field(default_factory=dict)
    initialise_near_guess: bool = True
    interation_callback: Optional[VariationalIterationCallback] = None


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
    loss: LossFunction | Tracer
    t_final: float | BoundedVariable
    system: DynamicalSystem
    control: Optional[List[ControlVariable]] = None
    parameters: Optional[List[ParameterVariable]] = None
    system_parameter_map: Optional[np.ndarray] = None
    quadratures: List[QuadratureSpec] = field(default_factory=list)
    path_constraints: List[InequalityExpression] = field(default_factory=list)
    terminal_constraints: List[InequalityExpression] = field(
        default_factory=list
    )
    initial_constraints: List[InequalityExpression] = field(
        default_factory=list
    )
    transcription_options: TranscriptionOptions = field(
        default_factory=TranscriptionOptions
    )
    backend: Optional[str] = "casadi"

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
        self.path_constraints = _normalize_constraints(self.path_constraints)
        self.terminal_constraints = _normalize_constraints(
            self.terminal_constraints
        )
        self.initial_constraints = _normalize_constraints(
            self.initial_constraints
        )
        if self.system_parameter_map is not None:
            expected_shape = (
                self.system.parameters.size,
                len(self.parameters),
            )
            assert expected_shape == self.system_parameter_map.shape, (
                "Parameter map is invalid. Expected an "
                f"{expected_shape} matrix, but got "
                f"{self.system_parameter_map.shape}."
            )
        elif (
            self.parameters is not None
            and self.system.parameters.size != len(self.parameters)
        ):
            raise ValueError(
                "Number of parameters does not match: expected "
                f"{self.system.parameters.size} but got "
                f"{len(self.parameters)}. Please provide a "
                "parameter map or specify the same number of "
                "parameters."
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
        from coker.backends import get_backend_by_name

        backend_name = self.backend if backend is None else backend
        return get_backend_by_name(backend_name).create_variational_solver(
            self
        )

    def __call__(self) -> VariationalSolution:
        return self.get_solver().solve()
