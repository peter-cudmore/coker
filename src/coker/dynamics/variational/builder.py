"""Context-managed construction of :class:`VariationalProblem` values."""

from typing import Optional, Sequence

import numpy as np

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.graph import Tape, TraceContext, Tracer
from coker.algebra.ops import Noop, OP
from coker.dynamics.variables import ControlVariable
from coker.parameters import (
    BoundVector,
    BoundedVariable,
    DenseTensorVariable,
    ParameterVariable,
)
from coker.parameters.function_parameters import FunctionParameter
from coker.dynamics.residual import ResidualDynamicalSystem
from coker.dynamics.model import DynamicalSystem
from coker.dynamics.variational.problem import (
    ConstraintSpec,
    QuadratureSpec,
    TranscriptionOptions,
    VariationalProblem,
)
from coker.dynamics.variational.function_binding import (
    specialize_system_parameters,
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
            raise ValueError("t_final BoundedVariable lower_bound must be positive")
        if t_final.upper_bound < t_final.lower_bound:
            raise ValueError(
                "t_final BoundedVariable upper_bound must not be below " "lower_bound"
            )
        if not t_final.lower_bound <= t_final.guess <= t_final.upper_bound:
            raise ValueError("t_final BoundedVariable guess must be within bounds")
    else:
        raise TypeError("t_final must be a positive float or BoundedVariable")


class VariationalProblemBuilder:
    """Build a variational problem from one context-owned symbolic trace.

    Args:
        system: Explicit, semi-explicit, or residual dynamic system.
        t_final: Positive horizon measured from physical time zero in the
            model's time units, or a bounded duration decision with positive
            lower bound and a guess within its bounds.
        control: Scalar control declarations in system input order. Coefficient
            rates and spike locations use normalised time ``t / t_final``.
        parameters: Fitted declarations or fixed numeric values in system
            parameter order. For heterogeneous parameter tuples, supply one
            declaration per space, including ``FunctionParameter`` for function
            spaces and vector/tensor declarations for vector spaces.
        backend: Name of the variational backend; defaults to CasADi.
        transcription_options: Collocation and backend solve configuration.
        system_parameter_map: Numeric map from the finite solver parameter
            vector into the system parameter vector.

    Symbolic access is context-owned. Use only ``0``, :attr:`t` and
    :attr:`t_final` as evaluation sites; arbitrary concrete sample times are
    not supported. Build inside the context and solve the returned problem
    after leaving it. Constraint and quadrature records are internal lowering
    details, not ordinary user declarations.

    Examples:
        Construct a terminal objective and a running integral with NumPy
        tracing. This constructs a problem; it does not request a NumPy
        variational solver.

        .. testcode:: variational-builder-source

            import numpy as np
            from coker import VectorSpace
            from coker.dynamics import (
                VariationalProblemBuilder, create_autonomous_ode,
            )
            from coker.parameters import BoundedVariable
            from coker.toolkits.codesign import Minimise

            system = create_autonomous_ode(
                x0=lambda p: p,
                xdot=lambda x, p: 0 * x,
                parameters=VectorSpace("p", 1),
                backend="numpy",
            )
            declarations = [BoundedVariable("initial", 0.0, 2.0, guess=0.5)]
            with VariationalProblemBuilder(
                system, t_final=1.0, parameters=declarations, backend="numpy",
            ) as builder:
                terminal = builder.output(builder.t_final)[0]
                running = builder.integrate(builder.state()[0] ** 2)
                built = builder.build(
                    Minimise((terminal - 1.0) ** 2 + running),
                    subject_to=[builder.state(0)[0] >= 0.0],
                )
            assert built.t_final == 1.0
            assert len(built.initial_constraints) == 1
    """

    def __init__(
        self,
        system: DynamicalSystem | ResidualDynamicalSystem,
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
        self._parameter_layout = None
        if isinstance(self.system.parameters, tuple):
            self._validate_parameters()
            (
                self.system,
                self._parameter_declarations,
                self._parameter_layout,
            ) = specialize_system_parameters(self.system, self._parameter_declarations)
        self._trace = Tape(backend)
        self._context: Optional[TraceContext] = None
        self._closed = False
        self._make_symbols()

    def _validate_parameters(self) -> None:
        """Validate positional declarations against the parameter tuple."""
        declarations = self._parameter_declarations
        space = self.system.parameters
        assert isinstance(space, tuple)
        if len(declarations) != len(space):
            raise ValueError(
                "Number of parameters does not match tuple: expected "
                f"{len(space)} but got {len(declarations)}"
            )
        for index, (element, declaration) in enumerate(zip(space, declarations)):
            if isinstance(element, FunctionSpace):
                if isinstance(declaration, FunctionParameter):
                    continue
                expected = "FunctionParameter"
            elif isinstance(element, VectorSpace):
                if isinstance(declaration, (BoundVector, DenseTensorVariable)):
                    continue
                expected = "BoundVector or DenseTensorVariable"
            elif isinstance(declaration, (ParameterVariable, float, np.number)):
                continue
            else:
                expected = "a finite parameter declaration"
            raise TypeError(
                f"Parameter {index} must be {expected}, got "
                f"{type(declaration).__name__}"
            )

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
            if isinstance(output_dim, Dimension)
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
        """Return the scalar symbolic physical-time marker for path sites.

        Returns:
            Tracer: Physical time varying across ``[0, t_final]``, not a
            numerical sample or normalised control coordinate.

        Examples:
            .. testcode:: variational-builder-source

                with VariationalProblemBuilder(system, 1.0, backend="numpy") as builder:
                    assert builder.t.dim.is_scalar()
        """
        self._require_open()
        return self._t

    @property
    def t_final(self) -> Tracer:
        """Return the scalar symbolic terminal physical-time marker.

        Returns:
            Tracer: Horizon in the model's time units. Use it as the terminal
            evaluation site for a fixed or optimised duration.

        Examples:
            .. testcode:: variational-builder-source

                with VariationalProblemBuilder(system, 1.0, backend="numpy") as builder:
                    assert builder.t_final.dim.is_scalar()
        """
        self._require_open()
        return self._t_final

    def state(self, time: Optional[object] = None) -> Tracer:
        """Return the symbolic differential state at ``time``.

        Args:
            time: ``0``, :attr:`t`, or :attr:`t_final`. Omitting it uses
                :attr:`t`.

        Returns:
            Tracer: Flat differential-state vector, excluding algebraic and
            quadrature channels, with the system's differential-state width
            and physical state units.

        Examples:
            .. testcode:: variational-builder-source

                with VariationalProblemBuilder(system, 1.0, backend="numpy") as builder:
                    assert builder.state(0).shape == (1,)
        """
        self._require_open()
        time = self._t if time is None else time
        self._validate_time(time)
        return self._state(time if isinstance(time, Tracer) else self._t_initial)

    def input(self, time: Optional[object] = None) -> Tracer:
        """Return the symbolic input at ``time``.

        Args:
            time: ``0``, :attr:`t`, or :attr:`t_final`. Omitting it uses
                :attr:`t`.

        Returns:
            Tracer | Noop: Scalar or vector input expression matching the
            system input space and units, or the absent-input marker when the
            system has no control trajectory. Evaluation sites use physical
            time; control declarations handle their own normalised coordinate.

        Examples:
            .. testcode:: variational-builder-source

                from coker import FunctionSpace, Scalar
                from coker.dynamics import create_control_system

                controlled = create_control_system(
                    x0=np.array([0.0]),
                    xdot=lambda t, x, u, p: u + 0 * x,
                    control=FunctionSpace("u", [Scalar("t")], [Scalar("u(t)")]),
                    backend="numpy",
                )
                with VariationalProblemBuilder(
                    controlled, 1.0, backend="numpy"
                ) as builder:
                    assert builder.input(0).dim.is_scalar()
        """
        self._require_open()
        time = self._t if time is None else time
        self._validate_time(time)
        if isinstance(self._input, Tracer):
            return self._input(time if isinstance(time, Tracer) else self._t_initial)
        return self._input

    def output(self, time: Optional[object] = None) -> Tracer:
        """Return the symbolic system output at ``time``.

        Args:
            time: ``0``, :attr:`t`, or :attr:`t_final`. Omitting it uses
                :attr:`t`.

        Returns:
            Tracer: Flat vector of declared observables, not the internal
            differential/algebraic/quadrature state tuple. Its width is the
            system output's flattened size; entries retain output units.

        Examples:
            .. testcode:: variational-builder-source

                with VariationalProblemBuilder(system, 1.0, backend="numpy") as builder:
                    assert builder.output(builder.t_final).shape == (1,)
        """
        self._require_open()
        time = self._t if time is None else time
        self._validate_time(time)
        return self._output(time if isinstance(time, Tracer) else self._t_initial)

    @property
    def parameters(self) -> Tracer:
        """Return the context-owned symbolic system parameters.

        Returns:
            Tracer | Noop: System parameter vector or heterogeneous tuple in
            declaration order, or the absent-parameter marker. Entries retain
            their declared dimensions and units; these are not control
            coefficients or the duration decision.

        Examples:
            .. testcode:: variational-builder-source

                with VariationalProblemBuilder(system, 1.0, backend="numpy") as builder:
                    assert builder.parameters.shape == (1,)
        """
        self._require_open()
        return self._parameters

    def _time_binding(self, time: object) -> object:
        if isinstance(time, (int, float, np.number)):
            if float(time) == 0:
                return _INITIAL_SITE
            raise ValueError(
                "unsupported concrete time; allowed bindings are " "0, t, and t_final"
            )
        if not isinstance(time, Tracer) or time.tape is not self._trace:
            raise ValueError("time marker belongs to a foreign or unrecognised trace")
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

        Args:
            expression: Scalar path expression from this builder's trace,
                integrated from physical time zero to :attr:`t_final`.

        Returns:
            Tracer: Scalar terminal integral with zero initial value and units
            equal to integrand units times the model's time unit. It is
            available in the objective and terminal constraints, not as a
            freely sampled trajectory.

        Raises:
            TypeError: The integrand is not a symbolic expression.
            ValueError: The integrand is nonscalar or belongs to another trace.

        Examples:
            .. testcode:: variational-builder-source

                with VariationalProblemBuilder(system, 1.0, backend="numpy") as builder:
                    running = builder.integrate(builder.state()[0] ** 2)
                    assert running.dim.is_scalar()
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
        """Build a problem from an objective and symbolic constraints.

        Args:
            objective: Scalar :class:`~coker.toolkits.codesign.Minimise`
                wrapping an expression from this builder's trace.
            subject_to: Symbolic ``==``, ``<=``, ``>=``, ``<`` or ``>``
                comparisons from this trace. A comparison depending on
                :attr:`t` is a path constraint; otherwise one depending on
                time zero is initial; all others are terminal. Mixed-site
                expressions follow this precedence. Strict comparisons are
                lowered to bound constraints, not strict numerical margins.

        Returns:
            VariationalProblem: Backend-neutral problem ready for
            :meth:`~VariationalProblem.get_solver`. No solve is performed.

        Raises:
            TypeError: The objective is not ``Minimise`` around a symbolic
                expression, or a constraint is not a symbolic comparison.
            ValueError: The objective is nonscalar or an expression belongs to
                another trace.

        Examples:
            .. testcode:: variational-builder-source

                with VariationalProblemBuilder(system, 1.0, backend="numpy") as builder:
                    problem = builder.build(
                        Minimise(builder.output(builder.t_final)[0] ** 2),
                        subject_to=[builder.output()[0] <= 2.0],
                    )
                assert len(problem.path_constraints) == 1
        """
        self._require_open()
        if not isinstance(objective, Minimise):
            raise TypeError("build requires a Minimise objective")
        loss = objective.expression
        if not isinstance(loss, Tracer):
            raise TypeError("Minimise cost must be a scalar symbolic expression")
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

        problem = VariationalProblem(
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
            transcription_options=self.transcription_options or TranscriptionOptions(),
            backend=self.backend,
        )
        problem.parameter_layout = self._parameter_layout
        return problem

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
        """Enter the symbolic trace used by the ``with`` block.

        Returns:
            VariationalProblemBuilder: This builder, bound as the context value.

        Examples:
            .. testcode:: variational-builder-source

                with VariationalProblemBuilder(system, 1.0, backend="numpy") as builder:
                    assert builder.state().shape == (1,)
        """
        self._context = TraceContext(self._trace)
        self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the trace and close symbolic access without suppressing errors.

        Args:
            exc_type: Exception type supplied by the context manager, or
                ``None`` on normal exit.
            exc_val: Exception instance, or ``None`` on normal exit.
            exc_tb: Exception traceback, or ``None`` on normal exit.

        Examples:
            .. testcode:: variational-builder-source

                with VariationalProblemBuilder(system, 1.0, backend="numpy") as builder:
                    problem = builder.build(Minimise(builder.state()[0] ** 2))
                assert problem.system is system
        """
        if self._context is not None:

            self._context.__exit__(exc_type, exc_val, exc_tb)
            self._context = None
        self._closed = True
