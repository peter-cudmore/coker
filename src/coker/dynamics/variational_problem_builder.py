"""Context-managed construction of :class:`VariationalProblem` values."""

from dataclasses import dataclass
import warnings
from typing import Optional, Sequence

import numpy as np

from coker.algebra.kernel import (
    Function,
    FunctionSpace,
    InequalityExpression,
    Noop,
    OP,
    Scalar,
    Tape,
    TraceContext,
    Tracer,
    VectorSpace,
)
from coker.dynamics.types import (
    BoundedVariable,
    ConstraintSpec,
    ControlVariable,
    DynamicalSystem,
    FinalTimeMapping,
    LossFunction,
    ParameterVariable,
    QuadratureSpec,
    TranscriptionOptions,
    TemporalBinding,
    VariationalProblem,
)
from coker.toolkits.codesign import Minimise


@dataclass(frozen=True)
class _Quadrature:
    """Symbolic dynamic quadrature channel registered by the builder."""

    integrand: Tracer
    state: Tracer
    initial_state: float
    channel: int
    trace_id: int


@dataclass
class _LegacyProblemState:
    loss: Optional[LossFunction] = None
    path_constraints: list[InequalityExpression] = None
    terminal_constraints: list[InequalityExpression] = None
    initial_constraints: list[InequalityExpression] = None

    def __post_init__(self):
        self.path_constraints = list(self.path_constraints or [])
        self.terminal_constraints = list(self.terminal_constraints or [])
        self.initial_constraints = list(self.initial_constraints or [])


class VariationalProblemBuilder:
    """Build a variational problem from one, context-owned symbolic trace."""

    def __init__(
        self,
        system: DynamicalSystem,
        t_final: float | BoundedVariable,
        *,
        control: Optional[Sequence[ControlVariable]] = None,
        parameters: Optional[Sequence[ParameterVariable]] = None,
        backend: Optional[str] = "casadi",
        transcription_options: Optional[TranscriptionOptions] = None,
        system_parameter_map: Optional[np.ndarray] = None,
    ):
        if isinstance(t_final, bool):
            raise TypeError(
                "t_final must be a positive float or BoundedVariable"
            )
        if isinstance(t_final, (int, float, np.number)):
            if float(t_final) <= 0:
                raise ValueError("t_final must be positive")
        elif isinstance(t_final, BoundedVariable):
            if t_final.lower_bound <= 0:
                raise ValueError(
                    "t_final BoundedVariable lower_bound must be positive"
                )
            if t_final.upper_bound < t_final.lower_bound:
                raise ValueError(
                    "t_final BoundedVariable upper_bound must not be below "
                    "lower_bound"
                )
            if not (
                t_final.lower_bound <= t_final.guess <= t_final.upper_bound
            ):
                raise ValueError(
                    "t_final BoundedVariable guess must be within bounds"
                )
        else:
            raise TypeError(
                "t_final must be a positive float or BoundedVariable"
            )
        self.system = system
        self._base_system = system
        self.t_final_declaration = t_final
        self.backend = backend
        self.transcription_options = transcription_options
        self.system_parameter_map = system_parameter_map
        self._legacy = _LegacyProblemState()
        self.control = list(control or [])
        self._parameter_declarations = list(parameters or [])
        self._lowered_constraints: list[ConstraintSpec] = []
        self._quadratures: list[_Quadrature] = []
        self._quadrature_derivative: list[Tracer] = []
        self._quadrature_initial: list[float] = []
        self._trace = Tape(backend)
        self._context: Optional[TraceContext] = None
        self._timed_values: dict[int, TemporalBinding] = {}
        self._closed = False
        self._make_symbols()

    @property
    def loss(self):
        return self._legacy.loss

    @loss.setter
    def loss(self, value):
        self._legacy.loss = value

    @property
    def path_constraints(self):
        return self._legacy.path_constraints

    @property
    def terminal_constraints(self):
        return self._legacy.terminal_constraints

    @property
    def initial_constraints(self):
        return self._legacy.initial_constraints

    def _make_symbols(self) -> None:
        x_dim, z_dim, _q_dim = self.system.get_state_dimensions()

        # Canonical private trajectory descriptions.  Function-valued symbols
        # are evaluated by the public accessors; downstream lowering still
        # accepts concrete values as constant trajectory arguments.
        self._state_trajectory = FunctionSpace(
            "x",
            arguments=[Scalar("t")],
            output=[VectorSpace("x", x_dim.flat())],
        )
        self._input_trajectory = (
            self.system.inputs
            if isinstance(self.system.inputs, FunctionSpace)
            else None
        )

        t = self._trace.input(Scalar("t"))
        terminal = self._trace.input(Scalar("t_final"))

        initial = self._trace.input(Scalar("t_0"))
        state = self._trace.input(self._state_trajectory)
        u = (
            self._trace.input(self.system.inputs)
            if not isinstance(self.system.inputs, Noop)
            else Noop()
        )
        p = (
            self._trace.input(self.system.parameters)
            if self.system.parameters is not None
            else Noop()
        )
        (
            self._t,
            self._t_final,
            self._t_initial,
            self._state,
            self._input,
            self._parameters,
        ) = (
            t,
            terminal,
            initial,
            state,
            u,
            p,
        )
        self._algebraic = (
            self._trace.input(VectorSpace("z", z_dim.flat()))
            if z_dim is not None and not z_dim.is_scalar() and z_dim.flat()
            else Noop()
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "symbolic access is only valid inside the builder context"
            )

    @property
    def t(self) -> Tracer:
        self._require_open()
        return self._t

    @property
    def t_final(self) -> Tracer:
        self._require_open()
        return self._t_final

    def minimise(self, loss: LossFunction) -> None:
        """Set the loss for the legacy imperative API.

        .. deprecated::
           Use :meth:`build` with a :class:`~coker.toolkits.codesign.Minimise`
           objective instead.
        """
        warnings.warn(
            "VariationalProblemBuilder.minimise() is deprecated; use build()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.loss = loss

    def add_input(self, control: ControlVariable) -> None:
        """Add a control using the deprecated imperative API."""
        warnings.warn(
            "VariationalProblemBuilder.add_input() is deprecated; "
            "pass control= to the constructor",
            DeprecationWarning,
            stacklevel=2,
        )
        self.control.append(control)

    def add_parameter(self, parameter: ParameterVariable) -> None:
        """Add a parameter using the deprecated imperative API."""
        warnings.warn(
            "VariationalProblemBuilder.add_parameter() is deprecated; "
            "pass parameters= to the constructor",
            DeprecationWarning,
            stacklevel=2,
        )
        self._parameter_declarations.append(parameter)

    def add_path_constraint(self, constraint: InequalityExpression) -> None:
        """Add a path constraint using the deprecated imperative API."""
        warnings.warn(
            "VariationalProblemBuilder.add_path_constraint() is deprecated; "
            "pass subject_to= to build()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.path_constraints.append(constraint)

    def add_terminal_constraint(
        self, constraint: InequalityExpression
    ) -> None:
        """Add a terminal constraint using the deprecated imperative API."""
        warnings.warn(
            "VariationalProblemBuilder.add_terminal_constraint() is "
            "deprecated; pass subject_to= to build()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.terminal_constraints.append(constraint)

    def _with_time(self, value: Tracer, time: object) -> Tracer:
        binding = self._time_binding(time)
        marker = {
            TemporalBinding.PATH: self._t,
            TemporalBinding.TERMINAL: self._t_final,
            TemporalBinding.INITIAL: self._t_initial,
        }[binding]
        tagged = value + 0 * marker
        self._timed_values[tagged.index] = binding
        return tagged

    def state(self, time: Optional[object] = None) -> Tracer:
        self._require_open()
        time = self._t if time is None else time
        value = self._state(
            time if isinstance(time, Tracer) else self._t_initial
        )
        return self._with_time(value, time)

    def input(self, time: Optional[object] = None) -> Tracer:
        self._require_open()
        if isinstance(self._input.dim, FunctionSpace):
            value = self._input(
                time if isinstance(time, Tracer) else self._t_initial
            )
            return self._with_time(value, self._t if time is None else time)
        return self._with_time(self._input, self._t if time is None else time)

    def output(self, time: Optional[object] = None) -> Tracer:
        self._require_open()
        time = self._t if time is None else time
        self._time_binding(time)
        if isinstance(self.system.y, Function):
            return self.system.y.call_inline(
                time if isinstance(time, Tracer) else self._t_initial,
                self.state(time),
                self._algebraic,
                self.input(time),
                self._parameters,
                Noop(),
            )
        return self.state(time)

    def parameters_symbol(self) -> Tracer:
        self._require_open()
        return self._parameters

    def parameters(self) -> Tracer:
        return self.parameters_symbol()

    def _time_binding(self, time: object) -> TemporalBinding:
        if isinstance(time, (int, float, np.number)):
            if float(time) == 0:
                return TemporalBinding.INITIAL
            raise ValueError(
                "unsupported concrete time; allowed bindings are "
                "0, t, and t_final"
            )
        if not isinstance(time, Tracer) or time.tape is not self._trace:
            raise ValueError(
                "time marker belongs to a foreign or unrecognised trace"
            )
        if time.index == self._t.index:
            return TemporalBinding.PATH
        if time.index == self._t_final.index:
            return TemporalBinding.TERMINAL
        raise ValueError(
            "unsupported time binding; allowed bindings are 0, t, and t_final"
        )

    def _validate_time(self, time: object) -> None:
        self._time_binding(time)

    def integrate(self, expression: Tracer) -> Tracer:
        """Register a scalar integrand and return its accumulated state.

        The returned tracer is a distinct quadrature channel.  Its initial
        value is zero and its derivative is the supplied expression; the
        derivative channels are exposed on ``system.dqdt`` for the dynamics
        transcription.
        """
        self._require_open()
        if not isinstance(expression, Tracer):
            raise TypeError("integrand must be a symbolic scalar expression")
        self._validate_trace(expression, "integrand")
        if not expression.dim.is_scalar():
            raise ValueError("integrand must be scalar")

        channel = len(self._quadratures)
        state = self._trace.input(Scalar(f"q_{channel}"))
        self._quadratures.append(
            QuadratureSpec(
                integrand=expression,
                initial_state=0.0,
                channel=channel,
            )
        )
        self._quadrature_derivative.append(expression)
        self._quadrature_initial.append(0.0)
        return state

    def build(
        self,
        objective: Optional[Minimise] = None,
        *,
        subject_to: Optional[Sequence[object]] = None,
    ) -> VariationalProblem:
        """Build a problem from a ``Minimise`` objective and constraints."""
        if objective is None:
            if self.loss is None:
                raise ValueError(
                    "A variational problem requires a loss functional or "
                    "Minimise objective"
                )
            loss = self.loss
        else:
            if not isinstance(objective, Minimise):
                raise TypeError("build requires a Minimise objective")
            loss = objective.expression
            if not isinstance(loss, Tracer):
                raise TypeError(
                    "Minimise cost must be a scalar symbolic expression"
                )
            self._validate_trace(loss, "cost")
            if not loss.dim.is_scalar():
                raise ValueError("Minimise cost must be scalar")

        constraints = list(subject_to or [])
        lowered: list[ConstraintSpec] = []
        for constraint in constraints:
            if not isinstance(constraint, Tracer):
                raise TypeError("constraints must be symbolic comparisons")
            self._validate_trace(constraint, "constraint")
            op = constraint.tape.op(constraint.index)
            if op not in {OP.EQUAL, OP.LESS_EQUAL, OP.LESS_THAN}:
                raise TypeError("constraints must be symbolic comparisons")
            residual, lower, upper = Tracer(
                self._trace, constraint.index
            ).as_halfplane_bound()
            binding = self._classify_time(residual)
            lowered.append(
                ConstraintSpec(
                    residual=residual,
                    lower_bound=lower,
                    upper_bound=upper,
                    temporal_binding=binding,
                )
            )

        path = list(self.path_constraints)
        terminal = list(self.terminal_constraints)
        initial = list(self.initial_constraints)
        for record in lowered:
            if record.temporal_binding is TemporalBinding.PATH:
                path.append(record)
            elif record.temporal_binding is TemporalBinding.INITIAL:
                initial.append(record)
            else:
                terminal.append(record)
        self._lowered_constraints = lowered

        if isinstance(loss, Tracer):
            self._validate_trace(loss, "cost")

        return VariationalProblem(
            path_constraints=path,
            loss=loss,
            system=self.system,
            t_final=self.t_final_declaration,
            control=self.control or None,
            parameters=self._parameter_declarations or None,
            quadratures=list(self._quadratures),
            system_parameter_map=self.system_parameter_map,
            final_time_map=(
                FinalTimeMapping(
                    declaration=self.t_final_declaration,
                    decision_index=0,
                )
                if isinstance(self.t_final_declaration, BoundedVariable)
                else FinalTimeMapping(value=float(self.t_final_declaration))
            ),
            terminal_constraints=terminal,
            initial_constraints=initial,
            transcription_options=self.transcription_options
            or TranscriptionOptions(),
            backend=self.backend,
        )

    def _classify_time(self, expression: Tracer) -> TemporalBinding:
        if expression.tape is not self._trace:
            raise ValueError("constraint contains a foreign trace")

        # Scope is inferred from ordinary trajectory-evaluation arguments.
        # TemporalBinding tags remain attached for backend lowering, but are
        # deliberately not consulted here.
        if self._trace.depends_on(expression, self._t):
            return TemporalBinding.PATH
        if self._trace.depends_on(expression, self._t_initial):
            return TemporalBinding.INITIAL
        return TemporalBinding.TERMINAL

    def _validate_trace(self, expression: Tracer, label: str) -> None:
        if expression.tape is not self._trace:
            raise ValueError(f"{label} belongs to a foreign trace")

    def __enter__(self) -> "VariationalProblemBuilder":
        self._context = TraceContext(self._trace)
        self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._context is not None:
            self._context.__exit__(exc_type, exc_val, exc_tb)
            self._context = None
        self._closed = True
