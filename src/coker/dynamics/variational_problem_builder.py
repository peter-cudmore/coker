"""Context-managed construction of :class:`VariationalProblem` values."""

from dataclasses import dataclass, replace
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
    function,
)
from coker.dynamics.types import (
    BoundedVariable,
    ConstraintSpec,
    ControlVariable,
    DynamicalSystem,
    FinalTimeMapping,
    LossFunction,
    ParameterVariable,
    TranscriptionOptions,
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
        self.loss: Optional[LossFunction] = None
        self.control = list(control or [])
        self._parameter_declarations = list(parameters or [])
        self.path_constraints: list[InequalityExpression] = []
        self.terminal_constraints: list[InequalityExpression] = []
        self.initial_constraints: list[InequalityExpression] = []
        self._lowered_constraints: list[ConstraintSpec] = []
        self._quadratures: list[_Quadrature] = []
        self._quadrature_derivative: list[Tracer] = []
        self._quadrature_initial: list[float] = []
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
        if isinstance(self._input.dim, FunctionSpace):
            return self._input(
                time if isinstance(time, Tracer) else self._t_initial
            )
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
        record = _Quadrature(
            integrand=expression,
            state=state,
            initial_state=0.0,
            channel=channel,
            trace_id=id(self._trace),
        )
        self._quadratures.append(record)
        self._quadrature_derivative.append(expression)
        self._quadrature_initial.append(record.initial_state)
        self._sync_quadrature_system()
        return state

    def _clone_expression(self, expression: Tracer, target: Tape, mapping):
        """Copy an expression graph onto the dynamics function tape."""
        key = (id(expression.tape), expression.index)
        if key in mapping:
            return mapping[key]
        op, *args = expression.tape.nodes[expression.index]
        cloned_args = [
            (
                self._clone_expression(arg, target, mapping)
                if isinstance(arg, Tracer)
                else arg
            )
            for arg in args
        ]
        result = target.append(op, *cloned_args)
        cloned = Tracer(target, result)
        mapping[key] = cloned
        return cloned

    def _sync_quadrature_system(self) -> None:
        """Create copied system with appended quadrature channels."""
        if not self._quadrature_derivative:
            return

        base = self._base_system
        spaces = base.dxdt.input_spaces()
        tape = Tape(self.backend)
        args = [tape.input(space) for space in spaces]
        mapping = {
            (id(self._trace), self._t.index): args[0],
            (id(self._trace), self._state.index): args[1],
        }
        for source, target in zip(
            (self._algebraic, self._input, self._parameters), args[2:]
        ):
            if isinstance(source, Tracer):
                mapping[(id(self._trace), source.index)] = target

        outputs = []
        if isinstance(base.dqdt, Function):
            existing = base.dqdt.call_inline(*args)
            outputs.append(existing)
        outputs.extend(
            self._clone_expression(expression, tape, mapping)
            for expression in self._quadrature_derivative
        )
        with TraceContext(tape):
            combined = np.concatenate(
                [np.reshape(output, (1,)) for output in outputs]
            )
        derivative = Function(
            tape, combined, self.backend, name="builder_quadratures"
        )
        self.system = self._system_with_quadratures(base, derivative)

    def _system_with_quadratures(
        self, base: DynamicalSystem, derivative: Function
    ) -> DynamicalSystem:
        """Return a copied system with an augmented q output."""

        _t, _x, _z, _u, _p, q_dim = base.y.input_shape()
        existing_q_size = 0 if q_dim is None else q_dim.flat()
        q_size = existing_q_size + len(self._quadrature_derivative)
        y_spaces = base.y.input_spaces()
        y_spaces[-1] = VectorSpace("q", q_size)

        def output(*values):
            q = values[-1]
            if existing_q_size:
                q = q[:existing_q_size]
                if q_dim.is_scalar():
                    q = q[0]
            else:
                q = None
            return base.y.call_inline(*values[:-1], q)

        output_function = function(
            y_spaces, output, backend=self.backend, name="builder_output"
        )
        return replace(base, dqdt=derivative, y=output_function)

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
            path_constraints=path,
            loss=loss,
            system=self.system,
            t_final=self.t_final_declaration,
            control=self.control or None,
            parameters=self._parameter_declarations or None,
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
                    if isinstance(node, Tracer):
                        return
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
