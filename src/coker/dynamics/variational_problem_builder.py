"""Context-managed construction of :class:`VariationalProblem` values."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from coker.algebra.kernel import (
    Function,
    InequalityExpression,
    Noop,
    Scalar,
    Tape,
    TraceContext,
    Tracer,
    VectorSpace,
    OP,
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
        self._trace = Tape(backend)
        self._context: Optional[TraceContext] = None
        self._closed = False
        self._make_symbols()

    def _make_symbols(self) -> None:
        t = self._trace.input(Scalar("t"))
        terminal = self._trace.input(Scalar("t_final"))
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
        self._t, self._t_final, self._state, self._input, self._parameters = (
            t,
            terminal,
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

    def state(self, time: Optional[Tracer] = None) -> Tracer:
        self._require_open()
        if time is not None:
            self._validate_time(time)
        return self._state

    def input(self, time: Optional[Tracer] = None) -> Tracer:
        self._require_open()
        if time is not None:
            self._validate_time(time)
        return self._input

    def output(self, time: Optional[Tracer] = None) -> Tracer:
        self._require_open()
        if time is None:
            time = self._t
        self._validate_time(time)
        if isinstance(self.system.y, Function):
            return self.system.y.call_inline(
                time,
                self._state,
                self._algebraic,
                self._input,
                self._parameters,
                Noop(),
            )
        return self._state

    def parameters_symbol(self) -> Tracer:
        self._require_open()
        return self._parameters

    def parameters(self) -> Tracer:
        return self.parameters_symbol()

    def _validate_time(self, time: Tracer) -> None:
        if not isinstance(time, Tracer) or time.tape is not self._trace:
            raise ValueError("time marker belongs to a foreign trace")

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
        for constraint in constraints:
            if not isinstance(constraint, Tracer):
                raise TypeError("constraints must be symbolic comparisons")
            self._validate_trace(constraint, "constraint")
            op = constraint.tape.op(constraint.index)
            if op not in {OP.EQUAL, OP.LESS_EQUAL, OP.LESS_THAN}:
                raise TypeError("constraints must be symbolic comparisons")

        if isinstance(loss, Tracer):
            self._validate_trace(loss, "cost")

        return VariationalProblem(
            loss=loss,
            system=self.system,
            t_final=self.t_final_declaration,
            control=self.control or None,
            parameters=self._parameter_declarations or None,
            system_parameter_map=self.system_parameter_map,
            path_constraints=self.path_constraints,
            terminal_constraints=self.terminal_constraints,
            transcription_options=self.transcription_options
            or TranscriptionOptions(),
            backend=self.backend,
        )

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
