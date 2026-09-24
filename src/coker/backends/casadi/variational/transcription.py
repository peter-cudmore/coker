import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

from coker.backends.backend import VariationalSolver
from coker.backends.casadi.variational.layout import DecisionLayout
from coker.backends.casadi.variational.variable_scaling import (
    _derive_constraint_scaling,
    _derive_variable_scaling,
)
from coker.dynamics import VariationalProblem, VariationalSolution
from coker.parameters import BoundedVariable, ParameterVariable
from coker.dynamics.transcription.collocation import (
    InterpolatingPoly,  # noqa: F401
    InterpolatingPolyCollection,  # noqa: F401
    _predict_refined_degree,
    _split_refined_interval,
)
from coker.dynamics.variational.mesh import (
    split_at_non_differentiable_points,
)
from coker.toolkits.codesign.optimisation import (
    SolveFailure,
    solve_info_from_casadi_stats,
)

from .bindings import ControlFactory  # noqa: F401
from .factory import _TranscriptionFactory, _resolve_options
from .loss import _lower_loss
from .symbolic_path import (
    CallbackWrapper,
    CasadiSolutionAssembler,
    SymbolicPoly,  # noqa: F401
    SymbolicPolyCollection,
)


def _derive_objective_scale(nominal_cost: object, tolerance: object) -> float:
    """Return a finite scale that never amplifies a sub-unit objective."""
    try:
        nominal = float(nominal_cost)
        tol = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        return 1.0

    if not (math.isfinite(nominal) and math.isfinite(tol)):
        return 1.0
    return max(1.0, abs(nominal), tol)


class CasadiVariationalSolver(VariationalSolver):
    def __init__(
        self,
        *,
        problem: VariationalProblem,
        parameters: List[str],
        map_arguments: Callable[..., Dict[str, ca.DM]],
        solver: ca.Function,
        assemble_solution: Callable[
            [ca.DM, float, object], VariationalSolution
        ],
        initialiser: Optional[ca.Function] = None,
        warm_start: bool = False,
        unscale_objective: Callable[[float], float] = float,
    ):
        self.problem = problem
        self._parameters = parameters
        self._map_arguments = map_arguments
        self._solver = solver
        self._assemble_solution = assemble_solution
        self._initialiser = initialiser
        self._warm_start = warm_start
        self._unscale_objective = unscale_objective
        self._last_primal: Optional[ca.DM] = None
        self._last_lam_x: Optional[ca.DM] = None
        self._last_lam_g: Optional[ca.DM] = None
        self._adaptive_solve: Optional[
            Callable[[Dict[str, float]], VariationalSolution]
        ] = None

    @property
    def parameters(self) -> List[str]:
        return list(self._parameters)

    def solve(self, **fixed_parameters) -> VariationalSolution:
        if self._adaptive_solve is not None:
            return self._adaptive_solve(fixed_parameters)
        return self._solve_once(**fixed_parameters)

    def _solve_once(
        self,
        previous_solution: Optional[VariationalSolution] = None,
        **fixed_parameters,
    ) -> VariationalSolution:
        """Solve one fixed transcription without adaptive mesh dispatch."""
        solver_arguments = self._map_arguments(
            fixed_parameters, previous_solution
        )
        x0 = solver_arguments["x0"]
        solver_kwargs = {
            "lbx": solver_arguments["lbx"],
            "ubx": solver_arguments["ubx"],
            "lbg": solver_arguments["lbg"],
            "ubg": solver_arguments["ubg"],
        }

        if (
            previous_solution is None
            and self._warm_start
            and self._last_primal is not None
        ):
            x0 = ca.fmin(
                ca.fmax(self._last_primal, solver_kwargs["lbx"]),
                solver_kwargs["ubx"],
            )
            if self._last_lam_x is not None:
                solver_kwargs["lam_x0"] = self._last_lam_x
            if self._last_lam_g is not None:
                solver_kwargs["lam_g0"] = self._last_lam_g
        elif self._initialiser is not None:
            initialised = self._initialiser(
                x0=x0,
                lbx=solver_arguments["lbx"],
                ubx=solver_arguments["ubx"],
                lbg=solver_arguments["init_lbg"],
                ubg=solver_arguments["init_ubg"],
            )
            x0 = initialised["x"]
            initial_cost = float(initialised["f"])
            assert (
                initial_cost <= ca.inf
            ), f"Cost at guess {initial_cost} is not finite"

        result = self._solver(
            x0=x0,
            **solver_kwargs,
        )
        solve_info = solve_info_from_casadi_stats(self._solver.stats())
        if not solve_info.success:
            raise SolveFailure(
                "CasADi variational solve failed with status "
                f"{solve_info.return_status}",
                solve_info,
            )
        if self._warm_start:
            self._last_primal = result["x"]
            self._last_lam_x = result["lam_x"]
            self._last_lam_g = result["lam_g"]
        return self._assemble_solution(
            result["x"],
            self._unscale_objective(float(result["f"])),
            solve_info,
        )


class _ConstraintAccumulator:
    """Collect equality and ranged constraints with their bounds."""

    def __init__(self, factory: _TranscriptionFactory):
        self._factory = factory
        self._equality_values = []
        self._equality_tolerances = []
        self._ranged_values = []
        self._ranged_lowers = []
        self._ranged_uppers = []

    def add_equality(self, value, tolerance) -> None:
        if value is not None:
            self._equality_values.append(value)
            self._equality_tolerances.append(tolerance)

    def add_constraint(
        self, constraint, args, *, validate_shape: bool = False
    ) -> None:
        (value,) = self._factory.casadi.evaluate(constraint.residual, args)
        lower = self._factory.casadi.to_backend_array(constraint.lower_bound)
        upper = self._factory.casadi.to_backend_array(constraint.upper_bound)
        if validate_shape:
            assert lower.shape == value.shape == upper.shape
        self._ranged_values.append(value)
        self._ranged_lowers.append(lower)
        self._ranged_uppers.append(upper)

    def build(self):
        equality_lowers = [
            -tolerance * ca.DM.ones(value.shape[0], 1)
            for value, tolerance in zip(
                self._equality_values, self._equality_tolerances
            )
        ]
        equality_uppers = [-lower for lower in equality_lowers]
        return (
            ca.vertcat(*self._equality_values, *self._ranged_values),
            ca.vertcat(*equality_lowers, *self._ranged_lowers),
            ca.vertcat(*equality_uppers, *self._ranged_uppers),
        )


@dataclass
class _NlpScaling:
    decision_variables: ca.MX
    raw_decision_variables: ca.MX
    physical_variables: ca.MX
    normalized_cost: ca.MX
    normalized_g: ca.MX
    physical_decision_variables_0: ca.DM
    physical_lower_bound_base: ca.DM
    physical_upper_bound_base: ca.DM
    lbg: ca.DM
    ubg: ca.DM
    variable_scaling: object
    objective_scale: float

    def unscale_objective(self, value: float) -> float:
        return value * self.objective_scale


def _scale_nlp(
    factory: _TranscriptionFactory,
    layout: DecisionLayout,
    poly_collection: SymbolicPolyCollection,
    decision_variables: ca.MX,
    cost: ca.MX,
    g: ca.MX,
    lbg: ca.DM,
    ubg: ca.DM,
    lower_bound_base: ca.DM,
    upper_bound_base: ca.DM,
    x0_guess,
    z0_guess,
) -> _NlpScaling:
    state_guess = ca.vertcat(
        x0_guess,
        (z0_guess if z0_guess is not None else ca.DM.zeros(factory.z_size, 1)),
        ca.DM.zeros(factory.q_size, 1),
    )
    path_guess = poly_collection.constant_guess(state_guess)
    raw_decision_variables = decision_variables
    physical_decision_variables_0 = layout.guess(
        path_guess, factory.u_guess, factory.p_guess_base
    )
    physical_lower_bound_base = ca.DM(lower_bound_base)
    physical_upper_bound_base = ca.DM(upper_bound_base)
    physical_variables = raw_decision_variables
    normalized_cost = cost
    normalized_g = g
    objective_scale = 1.0
    variable_scaling = None

    if factory.options.enable_scaling:
        variable_scaling = _derive_variable_scaling(
            np.asarray(physical_lower_bound_base).reshape(-1),
            np.asarray(physical_decision_variables_0).reshape(-1),
            np.asarray(physical_upper_bound_base).reshape(-1),
        )
        normalized_variables = ca.MX.sym(
            "normalized_decision", decision_variables.shape[0]
        )
        physical_variables = variable_scaling.decode(normalized_variables)
        normalized_cost = ca.substitute(
            cost, raw_decision_variables, physical_variables
        )
        normalized_g = ca.substitute(
            g, raw_decision_variables, physical_variables
        )
        objective_scale = _derive_objective_scale(
            float(
                ca.Function("nominal_cost", [raw_decision_variables], [cost])(
                    physical_decision_variables_0
                )
            ),
            factory.tolerance,
        )
        normalized_cost /= objective_scale
        decision_variables = normalized_variables
        constraint_scaling = _derive_constraint_scaling(
            normalized_g,
            decision_variables,
            variable_scaling.encode(physical_decision_variables_0),
            lbg,
            ubg,
        )
        inverse_constraint_scaling = ca.diag(ca.DM(1.0 / constraint_scaling))
        normalized_g = inverse_constraint_scaling @ normalized_g
        lbg = inverse_constraint_scaling @ lbg
        ubg = inverse_constraint_scaling @ ubg

    return _NlpScaling(
        decision_variables=decision_variables,
        raw_decision_variables=raw_decision_variables,
        physical_variables=physical_variables,
        normalized_cost=normalized_cost,
        normalized_g=normalized_g,
        physical_decision_variables_0=physical_decision_variables_0,
        physical_lower_bound_base=physical_lower_bound_base,
        physical_upper_bound_base=physical_upper_bound_base,
        lbg=lbg,
        ubg=ubg,
        variable_scaling=variable_scaling,
        objective_scale=objective_scale,
    )


@dataclass
class _CompiledNlp:
    nlp_solver: ca.Function
    init_solver: Optional[ca.Function]
    callback_wrapper: Optional[CallbackWrapper]
    warm_start: bool
    assemble_solution: CasadiSolutionAssembler
    unscale_objective: Callable[[float], float]
    variable_scaling: object
    physical_decision_variables_0: ca.DM
    physical_lower_bound_base: ca.DM
    physical_upper_bound_base: ca.DM
    lbg: ca.DM
    ubg: ca.DM


def _compile_nlp(
    factory: _TranscriptionFactory,
    layout: DecisionLayout,
    poly_collection: SymbolicPolyCollection,
    projectors,
    path_symbols: ca.MX,
    decision_variables: ca.MX,
    cost: ca.MX,
    g: ca.MX,
    lbg: ca.DM,
    ubg: ca.DM,
    lower_bound_base: ca.DM,
    upper_bound_base: ca.DM,
    x0_guess,
    z0_guess,
) -> _CompiledNlp:
    """Scale, compile, and configure one mesh-specific NLP."""
    problem = factory.problem
    solver_options = dict(factory.options.optimiser_options)
    warm_start = bool(solver_options.pop("warm_start", False))
    if not factory.options.verbose:
        solver_options.update(
            {
                "ipopt.print_level": 0,
                "print_time": False,
                "ipopt.sb": "yes",
            }
        )

    scaled = _scale_nlp(
        factory=factory,
        layout=layout,
        poly_collection=poly_collection,
        decision_variables=decision_variables,
        cost=cost,
        g=g,
        lbg=lbg,
        ubg=ubg,
        lower_bound_base=lower_bound_base,
        upper_bound_base=upper_bound_base,
        x0_guess=x0_guess,
        z0_guess=z0_guess,
    )
    f_out = ca.Function(
        "Output",
        [scaled.decision_variables],
        [
            ca.substitute(
                path_symbols,
                scaled.raw_decision_variables,
                scaled.physical_variables,
            ),
            ca.substitute(
                factory.u_symbols,
                scaled.raw_decision_variables,
                scaled.physical_variables,
            ),
            ca.substitute(
                factory.p,
                scaled.raw_decision_variables,
                scaled.physical_variables,
            ),
            ca.substitute(
                factory.p_symbols,
                scaled.raw_decision_variables,
                scaled.physical_variables,
            ),
            (
                ca.substitute(
                    factory.horizon_symbol,
                    scaled.raw_decision_variables,
                    scaled.physical_variables,
                )
                if factory.free_horizon
                else ca.DM(problem.t_final)
            ),
        ],
        {},
    )
    assemble_solution = CasadiSolutionAssembler(
        problem=problem,
        factory=factory,
        output_function=f_out,
        poly_collection=poly_collection,
        projectors=projectors,
        proj_p=factory.proj_p,
        parameter_indices=factory.parameter_indices,
        decode_controls=factory.control_decoder,
    )

    callback_wrapper = None
    nlp_solver_options = dict(solver_options)
    if warm_start:
        nlp_solver_options["ipopt.warm_start_init_point"] = "yes"
    if factory.options.interation_callback is not None:
        callback_wrapper = CallbackWrapper(
            "variational_iteration_callback",
            factory.options.interation_callback,
            nx=scaled.decision_variables.shape[0],
            ng=scaled.normalized_g.shape[0],
            assemble_solution=assemble_solution,
            unscale_objective=scaled.unscale_objective,
        )
        nlp_solver_options["iteration_callback"] = callback_wrapper
    init_solver = None
    if factory.options.initialise_near_guess:
        init_spec = {
            "f": scaled.normalized_cost,
            "x": scaled.decision_variables,
            "g": ca.vertcat(
                scaled.normalized_g,
                ca.substitute(
                    factory.p_symbols,
                    scaled.raw_decision_variables,
                    scaled.physical_variables,
                ),
                ca.substitute(
                    factory.u_symbols,
                    scaled.raw_decision_variables,
                    scaled.physical_variables,
                ),
                (
                    ca.substitute(
                        factory.horizon_symbol,
                        scaled.raw_decision_variables,
                        scaled.physical_variables,
                    )
                    if factory.free_horizon
                    else ca.MX.zeros(0, 1)
                ),
            ),
        }
        init_solver = ca.nlpsol(
            "initialiser",
            "ipopt",
            init_spec,
            dict(solver_options),
        )

    nlp_spec = {
        "f": scaled.normalized_cost,
        "x": scaled.decision_variables,
        "g": scaled.normalized_g,
    }
    nlp_solver = ca.nlpsol("solver", "ipopt", nlp_spec, nlp_solver_options)
    return _CompiledNlp(
        nlp_solver=nlp_solver,
        init_solver=init_solver,
        callback_wrapper=callback_wrapper,
        warm_start=warm_start,
        assemble_solution=assemble_solution,
        unscale_objective=scaled.unscale_objective,
        variable_scaling=scaled.variable_scaling,
        physical_decision_variables_0=scaled.physical_decision_variables_0,
        physical_lower_bound_base=scaled.physical_lower_bound_base,
        physical_upper_bound_base=scaled.physical_upper_bound_base,
        lbg=scaled.lbg,
        ubg=scaled.ubg,
    )


def _create_solver(
    factory: _TranscriptionFactory,
    intervals: List[Tuple[float, float]],
    degrees: List[int],
) -> CasadiVariationalSolver:
    """Compile one mesh-specific NLP from shared factory state."""
    problem = factory.problem
    duration = factory.duration
    projectors = factory._projectors

    poly_collection = SymbolicPolyCollection(
        name="x",
        dimension=factory.path_size,
        intervals=intervals,
        degrees=list(degrees),
        factory=factory,
        state_size=factory.x_size,
        algebraic_size=factory.z_size,
    )
    horizon = problem.horizon_decision
    layout = DecisionLayout(
        horizon_size=1 if factory.free_horizon else 0,
        path_size=int(poly_collection.symbols().shape[0]),
        control_size=int(factory.u_symbols.shape[0]),
        parameter_size=int(factory.p_symbols.shape[0]),
        horizon_lower=horizon.lower_bound if horizon else -ca.inf,
        horizon_guess=horizon.guess if horizon else 1.0,
        horizon_upper=horizon.upper_bound if horizon else ca.inf,
    )

    path_symbols = poly_collection.symbols()
    decision_variables = layout.vector(
        factory.horizon_symbol,
        path_symbols,
        factory.u_symbols,
        factory.p_symbols,
    )

    constraints = _ConstraintAccumulator(factory)

    t0, x0_symbol = next(poly_collection.interval_starts())
    x0_guess, z0_guess = factory.casadi.evaluate(
        problem.system.x0,
        [t0, factory.u_guess, factory.proj_p @ factory.p0_guess],
    )
    x0_val, z0_val = factory.casadi.evaluate(
        problem.system.x0,
        [t0, factory.control_eval, factory.proj_p @ factory.p],
    )

    constraints.add_equality(
        factory.proj_x @ x0_symbol - x0_val, factory.tolerance
    )
    if factory.z_size > 0:
        constraints.add_equality(
            factory.proj_z @ x0_symbol - z0_val, factory.tolerance
        )
    if factory.q_size > 0:
        constraints.add_equality(factory.proj_q @ x0_symbol, factory.tolerance)

    q_initial = ca.DM.zeros(factory.q_size, 1)
    u_initial = factory.control_eval(t0)
    initial_args = (
        t0,
        factory.proj_x @ x0_symbol,
        z0_val,
        u_initial,
        factory.p,
        q_initial,
    )
    for constraint in problem.initial_constraints:
        constraints.add_constraint(constraint, initial_args)

    system_parameters = factory.proj_p @ factory.p

    for poly in poly_collection.polys:
        for t, v, dv in poly.knot_points():
            physical_t = duration * t
            x = factory.proj_x @ v
            z = factory.proj_z @ v
            dx = factory.proj_x @ dv
            control = factory.control_eval(t)
            (dynamics_ij,) = factory.evaluate_dynamics(
                physical_t,
                x,
                z,
                control,
                system_parameters,
            )
            constraints.add_equality(
                dx - duration * dynamics_ij,
                factory.derivative_defect_tolerance,
            )

            if factory.q_size > 0:
                dq = factory.proj_q @ dv
                quadrature_values = []
                if factory.has_state_quadrature:
                    (base_quadrature,) = factory.evaluate_quadrature(
                        physical_t,
                        x,
                        z,
                        control,
                        system_parameters,
                    )
                    quadrature_values.append(base_quadrature)
                quadrature_values.extend(
                    factory.evaluate_registered_quadratures(
                        (
                            physical_t,
                            x,
                            z,
                            control,
                            system_parameters,
                            factory.proj_q @ v,
                        )
                    )
                )
                quadrature_ij = duration * ca.vertcat(*quadrature_values)
                constraints.add_equality(
                    dq - quadrature_ij,
                    factory.derivative_defect_tolerance,
                )

            if factory.z_size > 0:
                (alg,) = factory.evaluate_algebraic(
                    physical_t,
                    x,
                    z,
                    control,
                    system_parameters,
                )
                constraints.add_equality(alg, factory.tolerance)

    # Path constraints apply at interval endpoints and collocation knots.
    for poly in poly_collection.polys:
        for t, v in (poly.start_point(), poly.end_point()):
            physical_t = duration * t
            x = factory.proj_x @ v
            z = factory.proj_z @ v
            q = factory.proj_q @ v
            args = (
                physical_t,
                x,
                z,
                factory.control_eval(t),
                factory.proj_p @ factory.p,
                q,
            )
            for constraint in problem.path_constraints:
                constraints.add_constraint(constraint, args)

    path_lower_bound = -ca.DM.ones(poly_collection.size(), 1) * ca.inf
    path_upper_bound = ca.DM.ones(poly_collection.size(), 1) * ca.inf
    lower_bound_base = layout.bounds(
        path_lower_bound, factory.u_lower, factory.p_lower_base
    )
    upper_bound_base = layout.upper_bounds(
        path_upper_bound, factory.u_upper, factory.p_upper_base
    )

    cost = _lower_loss(factory, poly_collection, duration)

    t_end, v_end = poly_collection.polys[-1].end_point()
    x_end_val = factory.proj_x @ v_end
    z_end_val = factory.proj_z @ v_end
    q_end_val = factory.proj_q @ v_end
    u_end = factory.control_eval(t_end)
    end_args = (
        duration * t_end,
        x_end_val,
        z_end_val,
        u_end,
        factory.p,
        q_end_val,
    )

    for constraint in problem.terminal_constraints:
        constraints.add_constraint(constraint, end_args, validate_shape=True)

    g, lbg, ubg = constraints.build()

    compilation = _compile_nlp(
        factory=factory,
        layout=layout,
        poly_collection=poly_collection,
        projectors=projectors,
        path_symbols=path_symbols,
        decision_variables=decision_variables,
        cost=cost,
        g=g,
        lbg=lbg,
        ubg=ubg,
        lower_bound_base=lower_bound_base,
        upper_bound_base=upper_bound_base,
        x0_guess=x0_guess,
        z0_guess=z0_guess,
    )
    nlp_solver = compilation.nlp_solver
    init_solver = compilation.init_solver
    callback_wrapper = compilation.callback_wrapper
    warm_start = compilation.warm_start
    assemble_solution = compilation.assemble_solution
    unscale_objective = compilation.unscale_objective
    variable_scaling = compilation.variable_scaling
    physical_decision_variables_0 = compilation.physical_decision_variables_0
    physical_lower_bound_base = compilation.physical_lower_bound_base
    physical_upper_bound_base = compilation.physical_upper_bound_base
    lbg = compilation.lbg
    ubg = compilation.ubg

    parameter_offset = layout.parameter_slice.start
    path_offset = layout.horizon_size
    path_slice = slice(path_offset, path_offset + layout.path_size)
    control_slice = slice(
        path_slice.stop, path_slice.stop + layout.control_size
    )

    def interpolate_path_guess(previous_path) -> ca.DM:
        return poly_collection.collect_guess(previous_path)

    def compatible_control_guess(previous_solution) -> Optional[ca.DM]:
        if factory.control_factory is None:
            return None
        previous_controls = previous_solution.control_solutions
        if len(previous_controls) != len(factory.control_factory.sizes):
            return None
        values = []
        for control, size in zip(
            previous_controls, factory.control_factory.sizes
        ):
            value = np.asarray(control.value, dtype=float).reshape((-1,))
            if value.size != size:
                return None
            values.append(ca.DM(value).reshape((size, 1)))
        return ca.vertcat(*values)

    def map_arguments(
        fixed_parameters: Dict[str, ParameterVariable],
        previous_solution: Optional[VariationalSolution] = None,
    ) -> Dict[str, ca.DM]:
        unknown = sorted(
            set(fixed_parameters) - set(factory.parameter_indices)
        )
        if unknown:
            raise KeyError(f"Unknown solver parameters: {', '.join(unknown)}")

        physical_x0 = ca.DM(physical_decision_variables_0)
        physical_lbx = ca.DM(physical_lower_bound_base)
        physical_ubx = ca.DM(physical_upper_bound_base)
        p_guess = ca.DM(factory.p_guess_base)

        if previous_solution is not None:
            physical_x0[path_slice] = interpolate_path_guess(
                previous_solution.path
            )
            if factory.free_horizon:
                physical_x0[layout.horizon_slice] = previous_solution.t_final
            control_guess = compatible_control_guess(previous_solution)
            if control_guess is not None:
                physical_x0[control_slice] = control_guess
            previous_values = previous_solution._solver_parameter_vector
            if previous_values is not None:
                for index in factory.parameter_indices.values():
                    if index >= previous_values.size:
                        continue
                    value = previous_values[index]
                    physical_x0[parameter_offset + index] = value
                    p_guess[index] = value

        for name, value in fixed_parameters.items():
            index = factory.parameter_indices[name]
            decision_index = parameter_offset + index
            if isinstance(value, BoundedVariable):
                p_guess[index] = value.guess
                physical_lbx[decision_index] = value.lower_bound
                physical_ubx[decision_index] = value.upper_bound
                physical_x0[decision_index] = value.guess
            else:
                scalar = float(value)
                p_guess[index] = scalar
                physical_lbx[decision_index] = scalar
                physical_ubx[decision_index] = scalar
                physical_x0[decision_index] = scalar

        if variable_scaling is None:
            x0 = physical_x0
            lbx = physical_lbx
            ubx = physical_ubx
        else:
            x0 = ca.DM(variable_scaling.encode(physical_x0))
            lbx, ubx = variable_scaling.encode_bounds(
                physical_lbx, physical_ubx
            )
        init_control_guess = physical_x0[control_slice]
        init_horizon_guess = (
            physical_x0[layout.horizon_slice]
            if factory.free_horizon
            else ca.DM.zeros(0, 1)
        )
        init_lbg = ca.vertcat(
            lbg, p_guess, init_control_guess, init_horizon_guess
        )
        init_ubg = ca.vertcat(
            ubg, p_guess, init_control_guess, init_horizon_guess
        )
        return {
            "x0": x0,
            "lbx": lbx,
            "ubx": ubx,
            "lbg": lbg,
            "ubg": ubg,
            "init_lbg": init_lbg,
            "init_ubg": init_ubg,
        }

    solver = CasadiVariationalSolver(
        problem=problem,
        parameters=factory.parameter_names,
        map_arguments=map_arguments,
        solver=nlp_solver,
        assemble_solution=assemble_solution,
        initialiser=init_solver,
        warm_start=warm_start,
        unscale_objective=unscale_objective,
    )
    solver._callback_wrapper = callback_wrapper
    return solver


def create_variational_solver(
    problem: VariationalProblem,
) -> CasadiVariationalSolver:
    """Create a CasADi solver, optionally applying p-then-h refinement."""
    options = _resolve_options(problem)
    if (
        options.refinement_enabled
        and options.maximum_degree
        < problem.transcription_options.minimum_degree
    ):
        raise ValueError(
            "maximum_degree must be at least transcription minimum_degree"
        )
    if (
        options.refinement_enabled
        and problem.transcription_options.minimum_degree <= 1
    ):
        raise ValueError(
            "transcription minimum_degree must be greater than one "
            "for adaptive refinement"
        )

    factory = _TranscriptionFactory(problem)
    initial_intervals = split_at_non_differentiable_points(
        problem.control or [], 1.0, problem.transcription_options
    )
    initial_degrees = [problem.transcription_options.minimum_degree] * len(
        initial_intervals
    )

    def mesh_signature(
        intervals: List[Tuple[float, float]], degrees: List[int]
    ) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[int, ...]]:
        return (
            tuple((float(start), float(stop)) for start, stop in intervals),
            tuple(int(degree) for degree in degrees),
        )

    @lru_cache(maxsize=8)
    def compile_transcription(
        signature: Tuple[Tuple[Tuple[float, float], ...], Tuple[int, ...]],
    ) -> CasadiVariationalSolver:
        return _create_solver(factory, list(signature[0]), list(signature[1]))

    def transcription_for(
        intervals: List[Tuple[float, float]], degrees: List[int]
    ) -> CasadiVariationalSolver:
        return compile_transcription(mesh_signature(intervals, degrees))

    initial_solver = transcription_for(initial_intervals, initial_degrees)
    if not options.refinement_enabled:
        return initial_solver

    def solve_adaptive(
        fixed_parameters: Dict[str, float],
    ) -> VariationalSolution:
        intervals = initial_intervals
        degrees = initial_degrees
        previous_solution = None
        for iteration in range(options.maximum_iterations + 1):
            current_solver = transcription_for(intervals, degrees)
            solution = current_solver._solve_once(
                previous_solution=previous_solution,
                **fixed_parameters,
            )
            errors = [
                factory.measure_interval_defect(poly, solution)
                for poly in solution.path.polys
            ]
            maximum_error = max(errors, default=0.0)
            if maximum_error <= options.mesh_tolerance:
                return replace(
                    solution,
                    adaptive_refinement_rounds=iteration,
                    adaptive_maximum_defect=maximum_error,
                )
            if iteration == options.maximum_iterations:
                raise RuntimeError(
                    "CasADi adaptive refinement reached maximum_iterations "
                    f"({options.maximum_iterations}) with defect "
                    f"{maximum_error:.3e}"
                )

            new_intervals = []
            new_degrees = []
            for (start, stop), degree, error in zip(
                intervals, degrees, errors
            ):
                if error <= options.mesh_tolerance:
                    new_intervals.append((start, stop))
                    new_degrees.append(degree)
                    continue
                predicted = _predict_refined_degree(
                    error, options.mesh_tolerance, degree
                )
                refined_intervals, refined_degrees = _split_refined_interval(
                    (start, stop),
                    predicted,
                    options.maximum_degree,
                    problem.transcription_options.minimum_degree,
                    options.minimum_interval_duration,
                )
                new_intervals.extend(refined_intervals)
                new_degrees.extend(refined_degrees)
            previous_solution = solution
            intervals, degrees = new_intervals, new_degrees
        raise RuntimeError(
            "CasADi adaptive refinement terminated unexpectedly"
        )

    initial_solver._adaptive_solve = solve_adaptive
    return initial_solver
