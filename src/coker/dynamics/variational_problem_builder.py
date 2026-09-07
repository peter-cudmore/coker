"""Context-managed construction of :class:`VariationalProblem` values."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from coker.algebra.kernel import (
    Function,
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
    ControlVariable,
    DynamicalSystem,
    LossFunction,
    ParameterVariable,
    TranscriptionOptions,
    VariationalProblem,
)
from coker.toolkits.codesign import Minimise


@dataclass(frozen=True)
class _LoweredConstraint:
    """Immutable symbolic constraint record produced by the builder."""

    operation: OP
    residual: Tracer
    lower_bound: float
    upper_bound: float
    trace_id: int
    temporal_binding: str

    @property
    def trace_identity(self) -> int:
        return self.trace_id


class VariationalProblemBuilder:
    """Build a variational problem from one, context-owned symbolic trace.

    The symbolic accessors are intentionally scoped to the builder context.
    The imperative ``minimise`` and ``add_*`` methods remain available while
    callers migrate to :meth:`build`.
    """

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
        if isinstance(t_final, (int, float)) and t_final <= 0:
            raise ValueError("t_final must be positive")
        if not isinstance(t_final, (int, float, BoundedVariable)):
            raise TypeError(
                "t_final must be a positive float or BoundedVariable"
            )

        self.system = system
        self.t_final_declaration = t_final
        self.backend = backend
        self.transcription_options = transcription_options
        self.system_parameter_map = system_parameter_map
        self.loss: Optional[LossFunction] = None
        self.control = list(control or [])
        self._parameter_declarations = list(parameters or [])
        self.path_constraints: list[InequalityExpression] = []
        self.terminal_constraints: list[InequalityExpression] = []
        self.initial_constraints: list[InequalityExpression] = []
        self._lowered_constraints: list[_LoweredConstraint] = []
        self._trace = Tape(backend)
        self._context: Optional[TraceContext] = None
        self._closed = False
        self._make_symbols()

    def _make_symbols(self) -> None:
        t = self._trace.input(Scalar("t"))
        terminal = self._trace.input(Scalar("t_final"))
        initial = self._trace.input(Scalar("t_0"))
        x_dim, z_dim, _q_dim = self.system.get_state_dimensions()
        x = self._trace.input(VectorSpace("x", x_dim.flat()))
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
            x,
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
        """Set the loss for the legacy imperative API."""
        self.loss = loss

    def add_input(self, control: ControlVariable) -> None:
        self.control.append(control)

    def add_parameter(self, parameter: ParameterVariable) -> None:
        self._parameter_declarations.append(parameter)

    def add_path_constraint(self, constraint: InequalityExpression) -> None:
        self.path_constraints.append(constraint)

    def add_terminal_constraint(
        self, constraint: InequalityExpression
    ) -> None:
        self.terminal_constraints.append(constraint)

    def _with_time(self, value: Tracer, time: object) -> Tracer:
        binding = self._time_binding(time)
        marker = {
            "path": self._t,
            "terminal": self._t_final,
            "initial": self._t_initial,
        }[binding]
        # Preserve the endpoint dependency without changing the value.
        return value + 0 * marker

    def state(self, time: Optional[object] = None) -> Tracer:
        self._require_open()
        return self._with_time(self._state, self._t if time is None else time)

    def input(self, time: Optional[object] = None) -> Tracer:
        self._require_open()
        if isinstance(self._input, Noop):
            return self._input
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

    def _time_binding(self, time: object) -> str:
        if isinstance(time, (int, float, np.number)):
            if float(time) == 0:
                return "initial"
            raise ValueError(
                "unsupported concrete time; allowed bindings are "
                "0, t, and t_final"
            )
        if not isinstance(time, Tracer) or time.tape is not self._trace:
            raise ValueError(
                "time marker belongs to a foreign or unrecognised trace"
            )
        if time.index == self._t.index:
            return "path"
        if time.index == self._t_final.index:
            return "terminal"
        raise ValueError(
            "unsupported time binding; allowed bindings are 0, t, and t_final"
        )

    def _validate_time(self, time: object) -> None:
        self._time_binding(time)

    def integrate(self, expression: Tracer) -> Tracer:
        raise NotImplementedError("integrate is implemented in Phase 3")

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
        lowered: list[_LoweredConstraint] = []
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
                _LoweredConstraint(
                    operation=op,
                    residual=residual,
                    lower_bound=lower,
                    upper_bound=upper,
                    trace_id=id(self._trace),
                    temporal_binding=binding,
                )
            )

        path = list(self.path_constraints)
        terminal = list(self.terminal_constraints)
        initial = list(self.initial_constraints)
        for record in lowered:
            if record.temporal_binding == "path":
                path.append(record)
            elif record.temporal_binding == "initial":
                initial.append(record)
            else:
                terminal.append(record)
        self._lowered_constraints = lowered

        if isinstance(loss, Tracer):
            self._validate_trace(loss, "cost")

        return VariationalProblem(
            loss=loss,
            system=self.system,
            t_final=self.t_final_declaration,
            control=self.control or None,
            parameters=self._parameter_declarations or None,
            system_parameter_map=self.system_parameter_map,
            path_constraints=path,
            terminal_constraints=terminal,
            initial_constraints=initial,
            transcription_options=self.transcription_options
            or TranscriptionOptions(),
            backend=self.backend,
        )

    def _classify_time(self, expression: Tracer) -> str:
        found: set[str] = set()

        def visit(value: object) -> None:
            if isinstance(value, Tracer):
                if value.tape is not self._trace:
                    raise ValueError("constraint contains a foreign trace")
                if value.index == self._t.index:
                    found.add("path")
                elif value.index == self._t_final.index:
                    found.add("terminal")
                elif value.index == self._t_initial.index:
                    found.add("initial")
                else:
                    node = value.tape.nodes[value.index]
                    for arg in node[1:]:
                        visit(arg)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        visit(expression)
        if "path" in found:
            return "path"
        if "initial" in found:
            return "initial"
        return "terminal"

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
