"""Context-managed construction of :class:`VariationalProblem` values."""

from typing import Optional, Sequence

import numpy as np

from coker.algebra.dimensions import FunctionSpace, Scalar, VectorSpace
from coker.algebra.graph import Tape, TraceContext, Tracer
from coker.algebra.ops import Noop, OP
from coker.dynamics.controls import (
    BoundedVariable,
    ControlVariable,
    ParameterVariable,
)
from coker.dynamics.model import DynamicalSystem
from coker.dynamics.variational.problem import (
    ConstraintSpec,
    QuadratureSpec,
    TranscriptionOptions,
    VariationalProblem,
)
from coker.toolkits.codesign import Minimise


_PATH_SITE = object()
_INITIAL_SITE = object()
_TERMINAL_SITE = object()


def _validation_t_final(t_final: float | BoundedVariable) -> None:
    if isinstance(t_final, bool):
        raise TypeError("t_final must be a positive float or BoundedVariable")
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
        if not t_final.lower_bound <= t_final.guess <= t_final.upper_bound:
            raise ValueError(
                "t_final BoundedVariable guess must be within bounds"
            )
    else:
        raise TypeError("t_final must be a positive float or BoundedVariable")


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
        _validation_t_final(t_final)
        self.system = system
        self.t_final_declaration = t_final
        self.backend = backend
        self.transcription_options = transcription_options
        self.system_parameter_map = system_parameter_map
        self.control = list(control or [])
        self._parameter_declarations = list(parameters or [])
        self._quadratures: list[QuadratureSpec] = []
        self._trace = Tape(backend)
        self._context: Optional[TraceContext] = None
        self._closed = False
        self._make_symbols()

    def _make_symbols(self) -> None:
        x_dim, z_dim, _q_dim = self.system.get_state_dimensions()
        state_trajectory = FunctionSpace(
            "_state", [Scalar("t")], [VectorSpace("x", x_dim.flat())]
        )
        input_trajectory = (
            self.system.inputs
            if isinstance(self.system.inputs, FunctionSpace)
            else None
        )
        output_dim = self.system.y.output_shape()[0]
        output_size = (
            output_dim.flat()
            if hasattr(output_dim, "flat")
            else output_dim[0] if isinstance(output_dim, tuple) else output_dim
        )
        output_trajectory = FunctionSpace(
            "_output", [Scalar("t")], [VectorSpace("y", output_size)]
        )
        t = self._trace.input(Scalar("t"))
        terminal = self._trace.input(Scalar("t_final"))
        initial = self._trace.input(Scalar("t_0"))
        state = self._trace.input(state_trajectory)
        u = (
            self._trace.input(input_trajectory)
            if input_trajectory is not None
            else Noop()
        )
        p = (
            self._trace.input(self.system.parameters)
            if self.system.parameters is not None
            else Noop()
        )
        output = self._trace.input(output_trajectory)
        self._t, self._t_final, self._t_initial = t, terminal, initial
        self._state, self._input, self._parameters, self._output = (
            state,
            u,
            p,
            output,
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

    def state(self, time: Optional[object] = None) -> Tracer:
        self._require_open()
        time = self._t if time is None else time
        self._validate_time(time)
        return self._state(
            time if isinstance(time, Tracer) else self._t_initial
        )

    def input(self, time: Optional[object] = None) -> Tracer:
        self._require_open()
        time = self._t if time is None else time
        self._validate_time(time)
        if isinstance(self._input, Tracer):
            return self._input(
                time if isinstance(time, Tracer) else self._t_initial
            )
        return self._input

    def output(self, time: Optional[object] = None) -> Tracer:
        self._require_open()
        time = self._t if time is None else time
        self._validate_time(time)
        return self._output(
            time if isinstance(time, Tracer) else self._t_initial
        )

    @property
    def parameters(self) -> Tracer:
        self._require_open()
        return self._parameters

    def _time_binding(self, time: object) -> object:
        if isinstance(time, (int, float, np.number)):
            if float(time) == 0:
                return _INITIAL_SITE
            raise ValueError(
                "unsupported concrete time; allowed bindings are "
                "0, t, and t_final"
            )
        if not isinstance(time, Tracer) or time.tape is not self._trace:
            raise ValueError(
                "time marker belongs to a foreign or unrecognised trace"
            )
        if time.index == self._t.index:
            return _PATH_SITE
        if time.index == self._t_final.index:
            return _TERMINAL_SITE
        if time.index == self._t_initial.index:
            return _INITIAL_SITE
        raise ValueError(
            "unsupported time marker; allowed bindings are 0, t, and t_final"
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
        return state

    def build(
        self,
        objective: Minimise,
        *,
        subject_to: Optional[Sequence[object]] = None,
    ) -> VariationalProblem:
        """Build a problem from a ``Minimise`` objective and constraints."""
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
        lowered: list[tuple[ConstraintSpec, object]] = []
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
            lowered.append((ConstraintSpec(residual, lower, upper), binding))

        path = []
        terminal = []
        initial = []
        for record, binding in lowered:
            if binding is _PATH_SITE:
                path.append(record)
            elif binding is _INITIAL_SITE:
                initial.append(record)
            else:
                terminal.append(record)
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
            terminal_constraints=terminal,
            initial_constraints=initial,
            transcription_options=self.transcription_options
            or TranscriptionOptions(),
            backend=self.backend,
        )

    def _classify_time(self, expression: Tracer) -> object:
        if expression.tape is not self._trace:
            raise ValueError("constraint contains a foreign trace")
        if self._trace.depends_on(expression, self._t):
            return _PATH_SITE
        if self._trace.depends_on(expression, self._t_initial):
            return _INITIAL_SITE
        return _TERMINAL_SITE

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
