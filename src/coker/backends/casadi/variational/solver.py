from collections import OrderedDict
import math

from dataclasses import replace
from itertools import accumulate
from typing import Callable, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np
from coker.dynamics.transcription.collocation import (
    _create_reference_operator_cache,
    lgr_points,
)

from coker.backends.backend import VariationalSolver, get_backend_by_name
from coker.backends.casadi.lower import (
    lower as lower_casadi,
    substitute,
)
from coker.algebra.graph import Tracer
from coker.backends.casadi.variational.layout import DecisionLayout
from coker.backends.casadi.variational.options import CasadiVariationalOptions
from coker.dynamics import (
    BoundedVariable,
    ConstantControlVariable,
    ControlVariable,
    InterpolatingPoly,
    InterpolatingPolyCollection,
    ParameterVariable,
    PiecewiseConstantVariable,
    SpikeVariable,
    VariationalProblem,
    VariationalSolution,
    split_at_non_differentiable_points,
)
from coker.toolkits.codesign.optimisation import (
    SolveFailure,
    solve_info_from_casadi_stats,
)
from coker.backends.casadi.variational.variable_scaling import (
    _derive_variable_scaling,
)


def _resolve_options(problem: VariationalProblem) -> CasadiVariationalOptions:
    """Return validated CasADi options for a variational problem.

    Legacy ``TranscriptionOptions`` fields remain supported when no explicit
    backend policy is supplied. An explicit backend policy takes precedence.
    """
    options = problem.transcription_options.backend_options
    if options is None:
        transcription = problem.transcription_options
        return CasadiVariationalOptions(
            verbose=transcription.verbose,
            optimiser_options=transcription.optimiser_options,
            initialise_near_guess=transcription.initialise_near_guess,
            enable_scaling=transcription.enable_scaling,
            interation_callback=transcription.interation_callback,
        )
    if not isinstance(options, CasadiVariationalOptions):
        raise TypeError(
            "CasADi variational solving requires "
            "CasadiVariationalOptions as backend_options"
        )
    return options


def noop(*_args):
    return None


def _accepts_small_search_direction(
    solve_info,
    result: Dict[str, ca.DM],
    lower_bounds: ca.DM,
    upper_bounds: ca.DM,
    *,
    tolerance: float,
    min_tolerance: float = 1e-5,
) -> bool:
    if solve_info.return_status != "Search_Direction_Becomes_Too_Small":
        return False
    objective = float(result["f"])
    if not np.isfinite(objective):
        return False
    residual = result["g"]
    if residual.numel() == 0:
        return True
    violation = ca.fmax(lower_bounds - residual, residual - upper_bounds)
    max_violation = float(ca.mmax(ca.fmax(violation, 0)))
    return max_violation <= max(tolerance, min_tolerance)


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
        path_initial_guess=None,
        **fixed_parameters,
    ) -> VariationalSolution:
        """Solve one fixed transcription without adaptive mesh dispatch."""
        solver_arguments = self._map_arguments(
            fixed_parameters, path_initial_guess
        )
        x0 = solver_arguments["x0"]
        solver_kwargs = {
            "lbx": solver_arguments["lbx"],
            "ubx": solver_arguments["ubx"],
            "lbg": solver_arguments["lbg"],
            "ubg": solver_arguments["ubg"],
        }

        if (
            path_initial_guess is None
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
            if _accepts_small_search_direction(
                solve_info,
                result,
                solver_arguments["lbg"],
                solver_arguments["ubg"],
                tolerance=(
                    self.problem.transcription_options.absolute_tolerance
                ),
            ):
                solve_info = replace(solve_info, success=True)
            else:
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


_DEFECT_EVALUATOR_CACHE_SIZE = 16


class _TranscriptionFactory:
    """Mesh-independent CasADi bindings shared by all transcriptions."""

    def __init__(self, problem: VariationalProblem):
        self.problem = problem
        self.casadi = get_backend_by_name("casadi")
        self.options = _resolve_options(problem)
        x_dim, z_dim, q_dim = problem.system.get_state_dimensions()
        self.x_size = x_dim.flat()
        self.z_size = z_dim.flat() if z_dim else 0
        self.q_size = (q_dim.flat() if q_dim else 0) + len(problem.quadratures)
        self.has_state_quadrature = bool(q_dim)
        self.path_size = self.x_size + self.z_size + self.q_size
        self.tolerance = problem.transcription_options.absolute_tolerance
        self.free_horizon = problem.horizon_decision is not None

        self.proj_x = ca.hcat(
            [
                ca.MX.eye(self.x_size),
                ca.MX.zeros(self.x_size, self.z_size + self.q_size),
            ]
        )
        self.proj_z = ca.hcat(
            [
                ca.MX.zeros(self.z_size, self.x_size),
                ca.MX.eye(self.z_size),
                ca.MX.zeros(self.z_size, self.q_size),
            ]
        )
        self.proj_q = ca.hcat(
            [
                ca.MX.zeros(self.q_size, self.x_size + self.z_size),
                ca.MX.eye(self.q_size),
            ]
        )
        self._projectors = tuple(
            _to_output_projector(proj)
            for proj in (self.proj_x, self.proj_z, self.proj_q)
        )

        control_variables = problem.control or []
        self.control_factory = (
            ControlFactory(control_variables, 1.0)
            if control_variables
            else None
        )
        if self.control_factory is None:
            self.u_symbols = ca.MX.zeros(0, 1)
            self.u_lower = []
            self.u_upper = []
            self.u_guess = ca.DM.zeros(0, 1)
            self.control_eval = noop
            self.control_decoder = None
        else:
            self.u_symbols = self.control_factory.symbols()
            self.u_lower = self.control_factory.lower_bounds
            self.u_upper = self.control_factory.upper_bounds
            self.u_guess = self.control_factory.guess(0)
            self.control_eval = self.control_factory
            self.control_decoder = self.control_factory.to_output_array

        (
            self.p,
            self.p_symbols,
            self.p0_guess,
            (
                self.p_lower_base,
                self.p_guess_base,
                self.p_upper_base,
            ),
            self.p_output_map,
        ) = construct_parameters(problem.parameters)
        self.proj_p = (
            ca.DM(problem.system_parameter_map)
            if problem.system_parameter_map is not None
            else ca.DM.eye(self.p.shape[0])
        )
        self.horizon_symbol = (
            ca.MX.sym(problem.horizon_decision.name)
            if self.free_horizon
            else None
        )
        self.duration = (
            self.horizon_symbol
            if self.free_horizon
            else float(problem.t_final)
        )
        self.parameter_names = list(self.p_output_map.indices)
        self.parameter_indices = dict(self.p_output_map.indices)
        self.reference_operator_cache = _create_reference_operator_cache()

        self._defect_dynamics_maps: OrderedDict[int, ca.Function] = (
            OrderedDict()
        )
        self._defect_nodes: OrderedDict[int, tuple[float, ...]] = OrderedDict()
        self._defect_dynamics = self._build_defect_dynamics()

    def copy_solution_projectors(
        self,
    ) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """Return solution-owned projector arrays."""
        return tuple(
            projector.copy() if projector is not None else None
            for projector in self._projectors
        )

    def evaluate_dynamics(self, *args):
        return self.casadi.evaluate(self.problem.system.dxdt, args)

    def evaluate_quadrature(self, *args):
        if self.problem.system.dqdt is not None:
            return self.casadi.evaluate(self.problem.system.dqdt, args)
        return noop

    def evaluate_algebraic(self, *args):
        if self.problem.system.g:
            return self.casadi.evaluate(self.problem.system.g, args)
        return noop

    def evaluate_registered_quadratures(self, args):
        values = []
        for spec in self.problem.quadratures:
            workspace = dict(zip(spec.integrand.tape.input_indicies, args))
            _, outputs = lower_casadi(
                spec.integrand.tape, [spec.integrand], workspace
            )
            values.append(outputs[0])
        return values

    def _build_defect_dynamics(self) -> ca.Function:
        time = ca.MX.sym("defect_time")
        state = ca.MX.sym("defect_state", self.x_size)
        algebraic = ca.MX.sym("defect_algebraic", self.z_size)
        control = ca.MX.sym("defect_control", self.u_symbols.shape[0])
        parameters = ca.MX.sym("defect_parameters", self.proj_p.shape[0])
        (dynamics,) = self.evaluate_dynamics(
            time, state, algebraic, control, parameters
        )
        return ca.Function(
            "defect_dynamics",
            [time, state, algebraic, control, parameters],
            [dynamics],
        )

    def get_defect_nodes(self, degree: int) -> np.ndarray:
        """Return fresh elevated LGR defect nodes for one local degree."""
        try:
            nodes = self._defect_nodes.pop(degree)
        except KeyError:
            nodes = tuple(float(point) for point in lgr_points(degree + 2))
            if len(self._defect_nodes) == _DEFECT_EVALUATOR_CACHE_SIZE:
                self._defect_nodes.popitem(last=False)
        self._defect_nodes[degree] = nodes
        return np.asarray(nodes, dtype=float)

    def evaluate_defect_dynamics(
        self,
        times: np.ndarray,
        states: np.ndarray,
        algebraic: np.ndarray,
        controls: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        """Evaluate state dynamics over one interval in a CasADi batch."""
        count = int(times.shape[1])
        try:
            evaluator = self._defect_dynamics_maps.pop(count)
        except KeyError:
            evaluator = self._defect_dynamics.map(count)
            if len(self._defect_dynamics_maps) == _DEFECT_EVALUATOR_CACHE_SIZE:
                self._defect_dynamics_maps.popitem(last=False)
        self._defect_dynamics_maps[count] = evaluator
        return np.asarray(
            evaluator(
                ca.DM(times),
                ca.DM(states),
                ca.DM(algebraic),
                ca.DM(controls),
                ca.DM(parameters),
            ),
            dtype=float,
        )

    def measure_interval_defect(
        self, poly, solution: VariationalSolution
    ) -> float:
        """Return the maximum scaled state defect for one solved polynomial."""
        reference_nodes = self.get_defect_nodes(poly.degree)
        times = poly.interval[0] + (reference_nodes + 1.0) * poly.width
        path_values = np.asarray(poly.values, dtype=float).reshape(
            poly.dimension, -1
        )
        powers = np.power(
            reference_nodes,
            np.arange(poly.bases.shape[0], dtype=float)[:, np.newaxis],
        )
        derivative_powers = np.zeros_like(powers)
        derivative_powers[1:] = (
            np.arange(1, poly.bases.shape[0], dtype=float)[:, np.newaxis]
            * powers[:-1]
        )
        interpolation = path_values @ np.asarray(poly.bases).T
        states = interpolation @ powers
        derivatives = (
            interpolation @ derivative_powers / (poly.width * solution.t_final)
        )
        controls = (
            np.column_stack(
                [solution.control_law(float(time)) for time in times]
            )
            if self.control_factory is not None
            else np.zeros((0, len(times)))
        )
        parameters = np.repeat(
            np.asarray(solution.parameters, dtype=float).reshape((-1, 1)),
            len(times),
            axis=1,
        )
        model_dynamics = self.evaluate_defect_dynamics(
            (times * solution.t_final).reshape((1, -1)),
            states[: self.x_size],
            states[self.x_size : self.x_size + self.z_size],
            controls,
            parameters,
        )
        state_derivatives = derivatives[: self.x_size]
        if state_derivatives.size == 0:
            return 0.0
        scale = np.maximum(1.0, np.abs(model_dynamics))
        return float(
            np.max(np.abs(state_derivatives - model_dynamics) / scale)
        )


def _create_solver(
    factory: _TranscriptionFactory,
    intervals: List[Tuple[float, float]],
    degrees: List[int],
) -> CasadiVariationalSolver:
    """Compile one mesh-specific NLP from shared factory state."""
    problem = factory.problem
    duration = factory.duration
    projectors = factory.copy_solution_projectors()

    poly_collection = SymbolicPolyCollection(
        name="x",
        dimension=factory.path_size,
        intervals=intervals,
        degrees=list(degrees),
        factory=factory,
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

    equalities = []
    (t0, x0_symbol), *x_start = list(poly_collection.interval_starts())
    x_end = list(poly_collection.interval_ends())[:-1]

    x0_guess, z0_guess = factory.casadi.evaluate(
        problem.system.x0,
        [t0, factory.u_guess, factory.proj_p @ factory.p0_guess],
    )
    x0_val, z0_val = factory.casadi.evaluate(
        problem.system.x0,
        [t0, factory.control_eval, factory.proj_p @ factory.p],
    )

    equalities.append(factory.proj_x @ x0_symbol - x0_val)
    if factory.z_size > 0:
        equalities.append(factory.proj_z @ z0_val)

    equalities.extend(
        xs_i - xe_i for ((_, xs_i), (_, xe_i)) in zip(x_start, x_end)
    )

    def append_constraint(constraint, args):
        (value,) = factory.casadi.evaluate(constraint.residual, args)
        g_constraints.append(value)
        g_constraint_lowers.append(
            factory.casadi.to_backend_array(constraint.lower_bound)
        )
        g_constraint_uppers.append(
            factory.casadi.to_backend_array(constraint.upper_bound)
        )

    g_constraints = []
    g_constraint_lowers = []
    g_constraint_uppers = []
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
        append_constraint(constraint, initial_args)

    for poly in poly_collection.polys:
        interval_dynamics = []
        interval_quadratures = []
        for t, v, dv in poly.knot_points():
            physical_t = duration * t
            x = factory.proj_x @ v
            z = factory.proj_z @ v
            if factory.z_size > 0:
                z += z0_val
            dx = factory.proj_x @ dv
            (dynamics_ij,) = factory.evaluate_dynamics(
                physical_t,
                x,
                z,
                factory.control_eval(t),
                factory.proj_p @ factory.p,
            )
            scale = duration
            interval_dynamics.append(scale * dynamics_ij)
            equalities.append(dx - scale * dynamics_ij)

            if factory.q_size > 0:
                dq = factory.proj_q @ dv
                quadrature_values = []
                if factory.has_state_quadrature:
                    (base_quadrature,) = factory.evaluate_quadrature(
                        physical_t,
                        x,
                        z,
                        factory.control_eval(t),
                        factory.proj_p @ factory.p,
                    )
                    quadrature_values.append(base_quadrature)
                quadrature_values.extend(
                    factory.evaluate_registered_quadratures(
                        (
                            physical_t,
                            x,
                            z,
                            factory.control_eval(t),
                            factory.proj_p @ factory.p,
                            factory.proj_q @ v,
                        )
                    )
                )
                quadrature_ij = scale * ca.vertcat(*quadrature_values)
                equalities.append(dq - quadrature_ij)
                interval_quadratures.append(quadrature_ij)

            if factory.z_size > 0:
                (alg,) = factory.evaluate_algebraic(
                    physical_t,
                    x,
                    z,
                    factory.control_eval(t),
                    factory.proj_p @ factory.p,
                )
                equalities.append(alg)

        _, v_start = poly.start_point()
        _, v_end = poly.end_point()
        d_x = ca.hcat(interval_dynamics)
        weights = ca.DM(poly.weights[0, :-1])
        equalities.append(
            factory.proj_x @ v_end - factory.proj_x @ v_start - d_x @ weights
        )
        if factory.q_size > 0:
            d_q = ca.hcat(interval_quadratures)
            equalities.append(
                factory.proj_q @ v_end
                - factory.proj_q @ v_start
                - d_q @ weights
            )

    # Path constraints apply at interval endpoints and collocation knots.
    for poly in poly_collection.polys:
        for t, v in (poly.start_point(), poly.end_point()):
            physical_t = duration * t
            x = factory.proj_x @ v
            z = factory.proj_z @ v
            if factory.z_size > 0:
                z += z0_val
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
                append_constraint(constraint, args)

    path_lower_bound = -ca.DM.ones(poly_collection.size(), 1) * ca.inf
    path_upper_bound = ca.DM.ones(poly_collection.size(), 1) * ca.inf
    lower_bound_base = layout.bounds(
        path_lower_bound, factory.u_lower, factory.p_lower_base
    )
    upper_bound_base = layout.upper_bounds(
        path_upper_bound, factory.u_upper, factory.p_upper_base
    )

    def normalized_time(time):
        return time if factory.free_horizon else time / duration

    def state_proxy(time):
        return factory.proj_x @ poly_collection(normalized_time(time))

    def input_proxy(time):
        return factory.control_eval(normalized_time(time))

    def solution_proxy(*args):
        if len(args) == 1:
            time, control_val, p_val = (
                args[0],
                factory.control_eval,
                factory.p,
            )
        elif factory.control_factory is None:
            time, p_val = args
            control_val = factory.control_eval
        else:
            time, control_val, p_val = args
        tau = normalized_time(time)
        u_val = control_val(tau)
        inner = poly_collection(tau)
        x_tau = factory.proj_x @ inner
        z_tau = factory.proj_z @ inner
        q_tau = factory.proj_q @ inner
        (y_val,) = factory.casadi.evaluate(
            problem.system.y,
            [time, x_tau, z_tau, u_val, factory.proj_p @ p_val, q_tau],
        )
        return y_val

    if isinstance(problem.loss, Tracer):
        values = {
            "t": duration,
            "t_final": duration,
            "t_0": ca.DM.zeros(1, 1),
            "_state": state_proxy,
            "p": factory.p,
            "_output": solution_proxy,
        }
        if factory.control_factory is not None:
            values[problem.system.inputs.name] = input_proxy
        workspace = {
            index: values[name]
            for index, name in zip(
                problem.loss.tape.input_indicies,
                problem.loss.tape.input_names,
            )
            if name in values
        }
        (cost,) = substitute([problem.loss], workspace)
    elif factory.control_factory is None:
        (cost,) = factory.casadi.evaluate(
            problem.loss, [solution_proxy, factory.p]
        )
    else:
        (cost,) = factory.casadi.evaluate(
            problem.loss,
            [solution_proxy, factory.control_factory, factory.p],
        )

    equality_values = [e for e in equalities if e is not None]
    g = ca.vertcat(*equality_values, *g_constraints)
    equality_size = sum(e.shape[0] for e in equality_values)
    ubg = ca.vertcat(
        factory.tolerance * ca.DM.ones(equality_size, 1),
        *g_constraint_uppers,
    )
    lbg = ca.vertcat(
        -factory.tolerance * ca.DM.ones(equality_size, 1),
        *g_constraint_lowers,
    )

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
        (g_inner,) = factory.casadi.evaluate(constraint.residual, end_args)
        g_lower = factory.casadi.to_backend_array(constraint.lower_bound)
        g_upper = factory.casadi.to_backend_array(constraint.upper_bound)
        assert g_lower.shape == g_inner.shape == g_upper.shape
        g = ca.vertcat(g, g_inner)
        lbg = ca.vertcat(lbg, g_lower)
        ubg = ca.vertcat(ubg, g_upper)

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

    state_guess = ca.vertcat(
        x0_guess,
        (z0_guess if z0_guess is not None else ca.DM.zeros(factory.z_size, 1)),
        ca.DM.zeros(factory.q_size, 1),
    )
    n_reps = int(path_symbols.shape[0] / state_guess.shape[0])
    path_guess = ca.repmat(state_guess, n_reps)
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
        lower_bound_base, upper_bound_base = variable_scaling.encode_bounds(
            physical_lower_bound_base, physical_upper_bound_base
        )

    def unscale_objective(value: float) -> float:
        return value * objective_scale

    f_out = ca.Function(
        "Output",
        [decision_variables],
        [
            ca.substitute(
                path_symbols, raw_decision_variables, physical_variables
            ),
            ca.substitute(
                factory.u_symbols,
                raw_decision_variables,
                physical_variables,
            ),
            ca.substitute(
                factory.p, raw_decision_variables, physical_variables
            ),
            ca.substitute(
                factory.p_symbols,
                raw_decision_variables,
                physical_variables,
            ),
            (
                ca.substitute(
                    factory.horizon_symbol,
                    raw_decision_variables,
                    physical_variables,
                )
                if factory.free_horizon
                else ca.DM(problem.t_final)
            ),
        ],
        {},
    )
    assemble_solution = CasadiSolutionAssembler(
        problem=problem,
        output_function=f_out,
        poly_collection=poly_collection,
        projectors=projectors,
        proj_p=factory.proj_p,
        parameter_solution_map=factory.p_output_map,
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
            nx=decision_variables.shape[0],
            ng=normalized_g.shape[0],
            assemble_solution=assemble_solution,
            unscale_objective=unscale_objective,
        )
        nlp_solver_options["iteration_callback"] = callback_wrapper
    init_solver = None
    if factory.options.initialise_near_guess:
        init_spec = {
            "f": normalized_cost,
            "x": decision_variables,
            "g": ca.vertcat(
                normalized_g,
                ca.substitute(
                    factory.p_symbols,
                    raw_decision_variables,
                    physical_variables,
                ),
                ca.substitute(
                    factory.u_symbols,
                    raw_decision_variables,
                    physical_variables,
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
        "f": normalized_cost,
        "x": decision_variables,
        "g": normalized_g,
    }
    nlp_solver = ca.nlpsol("solver", "ipopt", nlp_spec, nlp_solver_options)

    parameter_offset = layout.parameter_slice.start
    path_offset = layout.horizon_size
    path_slice = slice(path_offset, path_offset + layout.path_size)

    def interpolate_path_guess(previous_path) -> ca.DM:
        knot_values = [
            ca.DM(previous_path(float(t))).reshape((factory.path_size, 1))
            for poly in poly_collection.polys
            for t in poly.knot_times()
        ]
        return ca.vertcat(*knot_values)

    def map_arguments(
        fixed_parameters: Dict[str, ParameterVariable],
        path_initial_guess=None,
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

        if path_initial_guess is not None:
            physical_x0[path_slice] = interpolate_path_guess(
                path_initial_guess
            )

        if variable_scaling is None:
            x0 = physical_x0
            lbx = physical_lbx
            ubx = physical_ubx
        else:
            x0 = ca.DM(variable_scaling.encode(physical_x0))
            lbx, ubx = variable_scaling.encode_bounds(
                physical_lbx, physical_ubx
            )
        init_lbg = ca.vertcat(
            lbg, p_guess, ca.DM.zeros(factory.u_symbols.shape)
        )
        init_ubg = ca.vertcat(
            ubg, p_guess, ca.DM.zeros(factory.u_symbols.shape)
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

    factory = _TranscriptionFactory(problem)
    initial_intervals = split_at_non_differentiable_points(
        problem.control or [], 1.0, problem.transcription_options
    )
    initial_degrees = [problem.transcription_options.minimum_degree] * len(
        initial_intervals
    )
    initial_solver = _create_solver(
        factory, initial_intervals, initial_degrees
    )
    if not options.refinement_enabled:
        return initial_solver

    cache_capacity = 8

    def mesh_signature(
        intervals: List[Tuple[float, float]], degrees: List[int]
    ) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[int, ...]]:
        return (
            tuple((float(start), float(stop)) for start, stop in intervals),
            tuple(int(degree) for degree in degrees),
        )

    compiled_transcriptions: OrderedDict[
        Tuple[Tuple[Tuple[float, float], ...], Tuple[int, ...]],
        CasadiVariationalSolver,
    ] = OrderedDict()
    initial_signature = mesh_signature(initial_intervals, initial_degrees)
    compiled_transcriptions[initial_signature] = initial_solver

    def transcription_for(
        intervals: List[Tuple[float, float]], degrees: List[int]
    ) -> CasadiVariationalSolver:
        signature = mesh_signature(intervals, degrees)
        try:
            solver = compiled_transcriptions.pop(signature)
        except KeyError:
            solver = _create_solver(
                factory, list(signature[0]), list(signature[1])
            )
            if len(compiled_transcriptions) == cache_capacity:
                compiled_transcriptions.popitem(last=False)
        compiled_transcriptions[signature] = solver
        return solver

    def solve_adaptive(
        fixed_parameters: Dict[str, float],
    ) -> VariationalSolution:
        intervals = initial_intervals
        degrees = initial_degrees
        previous_path = None
        for iteration in range(options.maximum_iterations + 1):
            current_solver = transcription_for(intervals, degrees)
            solution = current_solver._solve_once(
                path_initial_guess=previous_path,
                **fixed_parameters,
            )
            errors = [
                factory.measure_interval_defect(poly, solution)
                for poly in solution.path.polys
            ]
            maximum_error = max(errors, default=0.0)
            if maximum_error <= options.mesh_tolerance:
                return solution
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
                predicted = degree + max(
                    1,
                    int(
                        math.ceil(
                            math.log(
                                max(error, 1e-300) / options.mesh_tolerance
                            )
                        )
                    ),
                )
                if predicted <= options.maximum_degree:
                    new_intervals.append((start, stop))
                    new_degrees.append(predicted)
                    continue
                midpoint = (start + stop) / 2.0
                if stop - start <= 2.0 * options.minimum_interval_duration:
                    raise RuntimeError(
                        "CasADi adaptive refinement cannot split interval "
                        f"[{start}, {stop}] below minimum_interval_duration"
                    )
                new_intervals.extend(((start, midpoint), (midpoint, stop)))
                new_degrees.extend(
                    (
                        problem.transcription_options.minimum_degree,
                        problem.transcription_options.minimum_degree,
                    )
                )
            previous_path = solution.path
            intervals, degrees = new_intervals, new_degrees
        raise RuntimeError(
            "CasADi adaptive refinement terminated unexpectedly"
        )

    initial_solver._adaptive_solve = solve_adaptive
    return initial_solver


class CasadiSolutionAssembler:
    def __init__(
        self,
        *,
        problem: VariationalProblem,
        output_function: ca.Function,
        poly_collection: "SymbolicPolyCollection",
        projectors: Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
        ],
        proj_p: ca.DM,
        parameter_solution_map: Callable[[ca.DM], Dict[str, float]],
        decode_controls: Optional[Callable[[ca.DM], list]],
    ):
        self.problem = problem
        self.output_function = output_function
        self.poly_collection = poly_collection
        self.projectors = projectors
        self.proj_p = proj_p
        self.parameter_solution_map = parameter_solution_map
        self.decode_controls = decode_controls

    def __call__(
        self,
        decision_variables: ca.DM,
        loss: float,
        solve_info=None,
    ) -> VariationalSolution:
        (
            path_coefficients,
            control_coefficients,
            parameters,
            free_parameters,
            horizon,
        ) = self.output_function(decision_variables)
        path = self.poly_collection.to_fixed(np.array(path_coefficients))
        parameter_vector = np.array(parameters, dtype=float).reshape((-1, 1))
        system_parameters = np.array(
            self.proj_p @ ca.DM(parameter_vector)
        ).reshape((-1,))
        control_solutions = (
            self.decode_controls(control_coefficients)
            if self.decode_controls is not None
            else None
        )
        return VariationalSolution(
            cost=loss,
            projectors=tuple(
                projector.copy() if projector is not None else None
                for projector in self.projectors
            ),
            parameter_solutions=self.parameter_solution_map(free_parameters),
            parameters=system_parameters,
            parameter_block_layouts=self.problem.system.parameter_blocks,
            path=path,
            control_solutions=control_solutions,
            output=self.problem.system.y,
            t_final=float(horizon),
            solve_info=solve_info,
            path_constraint_exprs=self.problem.path_constraints,
            terminal_constraint_exprs=self.problem.terminal_constraints,
        )


class SymbolicPoly(InterpolatingPoly):
    def __init__(
        self,
        name,
        dimension,
        interval,
        degree,
        factory: Optional[_TranscriptionFactory] = None,
    ):
        size = (degree + 1) * dimension
        values = ca.MX.sym(name, size)
        super().__init__(
            dimension,
            interval,
            degree,
            values,
            reference_operators=(
                factory.reference_operator_cache(degree)
                if factory is not None
                else None
            ),
        )

    def symbols(self):
        return self.values

    def __call__(self, t):
        s = self._map_to_reference_coordinate(t)
        if not isinstance(s, (ca.SX, ca.MX)):
            try:
                i = next(
                    i for i, s_i in enumerate(self.s) if abs(s_i - s) < 1e-9
                )
                return self.values[
                    i * self.dimension : (i + 1) * self.dimension
                ]
            except StopIteration:
                pass
        n = len(self.s)
        s_vector = ca.vertcat(*[s**i for i in range(n)])
        projection = s_vector.T @ ca.DM(self.bases)
        value = ca.reshape(self.values, (self.dimension, -1)) @ projection.T
        return ca.reshape(value, (self.dimension, 1))


class SymbolicPolyCollection(InterpolatingPolyCollection):
    def symbols(self):
        return ca.vertcat(*[p.values for p in self.polys])

    def __init__(
        self,
        name,
        dimension,
        intervals,
        degrees,
        factory: Optional[_TranscriptionFactory] = None,
    ):
        assert len(intervals) == len(degrees)
        polys = [
            SymbolicPoly(
                f"{name}_{i}",
                dimension,
                interval,
                degree,
                factory=factory,
            )
            for i, (interval, degree) in enumerate(zip(intervals, degrees))
        ]
        super().__init__(polys)

    def to_fixed(self, array):
        size = sum(p.size() for p in self.polys)
        np_array = np.array(array)
        assert np_array.shape == (size, 1)

        slices = []
        offset = 0
        for p in self.polys:
            slices.append(slice(offset, offset + p.size()))
            offset += p.size()

        polys = [
            InterpolatingPoly(
                p.dimension,
                p.interval,
                p.degree,
                np_array[slc],
                reference_operators=p._reference_operators,
            )
            for (p, slc) in zip(self.polys, slices)
        ]
        return InterpolatingPolyCollection(polys)

    def __call__(self, t):
        if isinstance(t, (ca.SX, ca.MX)):
            result = 0
            for i, (start, end) in enumerate(self.intervals):
                poly_eval = self.polys[i](t)
                factor_1 = ca.if_else(t > start, poly_eval, 0)
                factor_2 = ca.if_else(t < end, factor_1, 0)
                result += factor_2
            return result
        return super().__call__(t)


class ParameterOutputMap:
    def __init__(self, indices: Dict[str, int]):
        self.indices = indices

    def __call__(self, value: ca.DM):
        return {name: float(value[i, 0]) for name, i in self.indices.items()}


def construct_parameters(parameters: Optional[List[ParameterVariable]]):
    parameters = parameters or []

    params = []
    upper_bounds = []
    guess = []
    lower_bounds = []
    symbols = {}
    p0 = []
    output_map = {}
    for p in parameters:
        if isinstance(p, BoundedVariable):
            try:
                symbol = symbols[p.name]
                params.append(symbol)
                index = output_map[p.name]
                p0.append(p0[index])
                continue
            except KeyError:
                pass
            symbol = ca.MX.sym(f"{p.name}")
            output_map[p.name] = len(symbols)
            params.append(symbol)
            symbols[p.name] = symbol
            upper_bounds.append(
                p.upper_bound if p.upper_bound is not None else ca.inf
            )
            guess.append(p.guess)
            p0.append(p.guess)
            lower_bounds.append(
                p.lower_bound if p.lower_bound is not None else -ca.inf
            )
        elif isinstance(p, (float, int)):
            params.append(ca.MX(p))
            p0.append(p)
        else:
            raise ValueError(f"Parameter {p} is not a valid parameter")

    if params:
        parameter_vector = ca.vertcat(*params)
    else:
        parameter_vector = ca.MX.zeros(0, 1)

    if symbols:
        symbol_vector = ca.vertcat(*symbols.values())
        lower = ca.DM(lower_bounds)
        guess_vector = ca.DM(guess)
        upper = ca.DM(upper_bounds)
    else:
        symbol_vector = ca.MX.zeros(0, 1)
        lower = ca.DM.zeros(0, 1)
        guess_vector = ca.DM.zeros(0, 1)
        upper = ca.DM.zeros(0, 1)

    return (
        parameter_vector,
        symbol_vector,
        ca.DM(p0).reshape((-1, 1)) if p0 else ca.DM.zeros(0, 1),
        (lower, guess_vector, upper),
        ParameterOutputMap(output_map),
    )


class ControlFactory:
    def __init__(self, variables: List[ControlVariable], t_final: float):
        self.t_final = t_final
        self.variables = variables
        self._symbols = [
            ca.MX.sym(v.name, v.degrees_of_freedom(0, t_final))
            for v in variables
        ]
        self.upper_bounds = [
            ca.DM.ones(v.degrees_of_freedom(0, t_final))
            * (v.upper_bound if v.upper_bound != np.inf else ca.inf)
            for v in variables
        ]
        self.lower_bounds = [
            ca.DM.ones(v.degrees_of_freedom(0, t_final))
            * (v.lower_bound if v.lower_bound != -np.inf else -ca.inf)
            for v in variables
        ]
        self.sizes = [v.degrees_of_freedom(0, t_final) for v in variables]
        offsets = [0, *accumulate(self.sizes[:-1])]
        self.offsets = offsets

    def guess(self, _):
        return ca.DM.zeros(len(self.variables), 1)

    def symbols(self) -> ca.MX:
        return (
            ca.vertcat(*self._symbols) if self._symbols else ca.MX.zeros(0, 1)
        )

    def __call__(self, t):
        assert (
            0 <= t <= self.t_final
        ), f"Control variable is not defined at t = {t}"
        out = []
        for s, var in zip(self._symbols, self.variables):
            if isinstance(var, ConstantControlVariable):
                out.append(s)
            elif isinstance(var, SpikeVariable):
                out.append(s if abs(t - var.time) < 1e-9 else 0)
            elif isinstance(var, PiecewiseConstantVariable):
                index = int(t * var.sample_rate)
                out.append(s[index])
            else:
                raise ValueError(
                    f"Control variable {var} is not a valid control variable"
                )
        return ca.vertcat(*out) if out else ca.MX.zeros(0, 1)

    def to_output_array(self, solution: ca.DM):
        return [
            v.to_solution(solution[offset : offset + size])
            for v, offset, size in zip(
                self.variables, self.offsets, self.sizes
            )
        ]


def _to_output_projector(proj: ca.MX) -> Optional[np.ndarray]:
    if proj.shape[0] == 0:
        return None
    return np.array(proj.to_DM()).reshape(proj.shape)


class CallbackWrapper(ca.Callback):
    def __init__(
        self,
        name: str,
        callback,
        *,
        nx: int,
        ng: int,
        assemble_solution: Callable[
            [ca.DM, float, object], VariationalSolution
        ],
        unscale_objective: Callable[[float], float] = float,
        opts=None,
    ):
        ca.Callback.__init__(self)
        self.callback = callback
        self.nx = nx
        self.ng = ng
        self.assemble_solution = assemble_solution
        self.unscale_objective = unscale_objective
        self.construct(name, {} if opts is None else opts)
        self._iterate_count = 0

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_name_out(self, _i):
        return "ret"

    def get_sparsity_in(self, i):
        name = ca.nlpsol_out(i)
        if name == "f":
            return ca.Sparsity.scalar()
        if name in ("x", "lam_x"):
            return ca.Sparsity.dense(self.nx, 1)
        if name in ("g", "lam_g"):
            return ca.Sparsity.dense(self.ng, 1)
        return ca.Sparsity(0, 0)

    def eval(self, arg):
        darg = {name: value for name, value in zip(ca.nlpsol_out(), arg)}
        solution = self.assemble_solution(
            darg["x"],
            self.unscale_objective(float(darg["f"])),
            None,
        )
        should_continue = bool(self.callback(self._iterate_count, solution))
        self._iterate_count += 1
        return [0 if should_continue else 1]
