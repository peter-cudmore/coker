from typing import Dict

import casadi as ca
import numpy as np

from coker.algebra.kernel import Noop
from coker.backends.backend import get_backend_by_name
from coker.dynamics import (
    BoundedVariable,
    ParameterVariable,
    VariationalProblem,
    VariationalSolution,
)
from coker.dynamics.transcription_helpers import (
    InterpolatingPoly,
    split_at_non_differentiable_points,
)
from coker.dynamics.types import InterpolatingPolyCollection

from .variational_solver import (
    CallbackWrapper,
    CasadiVariationalSolver,
    ParameterOutputMap,
    construct_parameters,
)

VARIATIONAL_SOLVER_OPTION = "variational_solver"
COLLOCATION_SOLVER = "collocation"
DAE_OPTIMISER_SOLVER = "dae_optimiser"


def _as_column(value):
    if value is None:
        return ca.DM.zeros(0, 1)
    if isinstance(value, (float, int, np.floating, np.integer)):
        return ca.DM([float(value)])
    if isinstance(value, np.ndarray):
        return ca.reshape(ca.DM(value), (value.size, 1))
    return ca.reshape(value, (int(value.numel()), 1))


def _vector_size(value) -> int:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return 1
    return int(value.numel())


class CasadiDaeOptimiser(CasadiVariationalSolver):
    pass


class DaePointEvaluator:
    def __init__(
        self,
        *,
        problem: VariationalProblem,
        parameter_projector: ca.DM,
    ):
        self.problem = problem
        self._casadi = get_backend_by_name("casadi")
        self._parameter_projector = parameter_projector
        self._is_dae = problem.system.g is not Noop()
        self._has_quadrature = problem.system.dqdt is not Noop()
        self._projected_parameter_size = parameter_projector.shape[0]
        self._integrator = self._build_integrator()

    def project_parameters(self, full_parameters):
        return self._parameter_projector @ full_parameters

    def initial_conditions(self, projected_parameters):
        x0, z0 = self._casadi.evaluate(
            self.problem.system.x0,
            [0.0, None, projected_parameters],
        )
        return _as_column(x0), _as_column(z0)

    def state_at(self, time: float, projected_parameters):
        if time < 0 or time > self.problem.t_final:
            raise ValueError(
                f"Time {time} is outside [0, {self.problem.t_final}]"
            )

        x0, z0 = self.initial_conditions(projected_parameters)
        q0 = self._quadrature_initial_state()
        if time == 0:
            return x0, z0, q0

        runtime_parameters = ca.vertcat(projected_parameters, ca.DM([time]))
        result = self._integrator(
            x0=ca.vertcat(ca.DM([0.0]), x0, q0),
            p=runtime_parameters,
            z0=z0,
        )
        txq_final = result["xf"]
        x_size = _vector_size(x0)
        x_final = txq_final[1 : 1 + x_size]
        q_final = txq_final[1 + x_size :]
        z_final = result["zf"] if self._is_dae else ca.DM.zeros(0, 1)
        return x_final, z_final, q_final

    def output_at(self, time: float, full_parameters):
        projected_parameters = self.project_parameters(full_parameters)
        x_val, z_val, q_val = self.state_at(time, projected_parameters)
        (output,) = self._casadi.evaluate(
            self.problem.system.y,
            [time, x_val, z_val, None, projected_parameters, q_val],
        )
        return output

    def _quadrature_initial_state(self):
        if not self._has_quadrature:
            return ca.DM.zeros(0, 1)
        q_dim = int(self.problem.system.y.input_shape()[-1].flat())
        return ca.DM.zeros(q_dim, 1)

    def _build_integrator(self) -> ca.Function:
        projected_parameters = ca.MX.sym(
            "p",
            self._projected_parameter_size,
            1,
        )
        horizon = ca.MX.sym("horizon")
        x0_expr, z0_expr = self.initial_conditions(projected_parameters)
        x = ca.MX.sym("x", _vector_size(x0_expr), 1)
        t = ca.MX.sym("t")
        z = self._algebraic_symbol(z0_expr)
        q, dqdt = self._quadrature_dynamics(
            t, x, z, projected_parameters, horizon
        )

        (dxdt,) = self._casadi.evaluate(
            self.problem.system.dxdt,
            [t, x, z, None, projected_parameters],
        )
        dae = {
            "x": ca.vertcat(t, x, q),
            "p": ca.vertcat(projected_parameters, horizon),
            "ode": ca.vertcat(horizon, horizon * _as_column(dxdt), dqdt),
        }
        if self._is_dae:
            (alg,) = self._casadi.evaluate(
                self.problem.system.g,
                [t, x, z, None, projected_parameters],
            )
            dae["z"] = z
            dae["alg"] = _as_column(alg)

        return ca.integrator("dae_optimiser", "idas", dae, 0.0, 1.0, {})

    def _algebraic_symbol(self, z0_expr):
        if not self._is_dae:
            return ca.MX.zeros(0, 1)
        return ca.MX.sym("z", _vector_size(z0_expr), 1)

    def _quadrature_dynamics(self, time, x, z, projected_parameters, horizon):
        if not self._has_quadrature:
            return ca.MX.zeros(0, 1), ca.MX.zeros(0, 1)

        q_dim = int(self.problem.system.y.input_shape()[-1].flat())
        q = ca.MX.sym("q", q_dim, 1)
        (dqdt,) = self._casadi.evaluate(
            self.problem.system.dqdt,
            [time, x, z, None, projected_parameters],
        )
        return q, horizon * _as_column(dqdt)


class IntegratedPath:
    def __init__(
        self,
        *,
        problem: VariationalProblem,
        point_evaluator: DaePointEvaluator,
        parameters: ca.DM,
        x_size: int,
        z_size: int,
        q_size: int,
    ):
        self.problem = problem
        self.point_evaluator = point_evaluator
        self.parameters = parameters
        self.x_size = x_size
        self.z_size = z_size
        self.q_size = q_size
        self.total_size = x_size + z_size + q_size
        self.degree = problem.transcription_options.minimum_degree
        self.intervals = split_at_non_differentiable_points(
            [],
            problem.t_final,
            problem.transcription_options,
        )

    def __call__(self, time):
        x_val, z_val, q_val = self.point_evaluator.state_at(
            time, self.parameters
        )
        point = [np.array(x_val, dtype=float).reshape((-1,))]
        if self.z_size > 0:
            point.append(np.array(z_val, dtype=float).reshape((-1,)))
        if self.q_size > 0:
            point.append(np.array(q_val, dtype=float).reshape((-1,)))
        return np.concatenate(point) if point else np.zeros((0,))

    def knot_points(self):
        for poly in self._sampled_polys().polys:
            yield from poly.knot_points()

    def map(self, func):
        return self._sampled_polys().map(func)

    def _sampled_polys(self) -> InterpolatingPolyCollection:
        polys = []
        for interval in self.intervals:
            seed = InterpolatingPoly(
                self.total_size,
                interval,
                self.degree,
                np.zeros(((self.degree + 1) * self.total_size, 1)),
            )
            values = [
                self(time).reshape((-1, 1)) for time in seed.knot_times()
            ]
            polys.append(
                InterpolatingPoly(
                    self.total_size,
                    interval,
                    self.degree,
                    np.vstack(values),
                )
            )
        return InterpolatingPolyCollection(polys)


class DaeSolutionAssembler:
    def __init__(
        self,
        *,
        problem: VariationalProblem,
        output_function: ca.Function,
        point_evaluator: DaePointEvaluator,
        parameter_solution_map: ParameterOutputMap,
    ):
        self.problem = problem
        self.output_function = output_function
        self.point_evaluator = point_evaluator
        self.parameter_solution_map = parameter_solution_map
        x_dim, z_dim, q_dim = problem.system.get_state_dimensions()
        self._x_size = int(x_dim.flat())
        self._z_size = int(z_dim.flat()) if z_dim else 0
        self._q_size = int(q_dim.flat()) if q_dim else 0
        self._projectors = self._build_projectors()

    def __call__(
        self,
        decision_variables: ca.DM,
        loss: float,
        solve_info=None,
    ) -> VariationalSolution:
        parameters, free_parameters = self.output_function(decision_variables)
        parameter_vector = ca.DM(parameters)
        path = IntegratedPath(
            problem=self.problem,
            point_evaluator=self.point_evaluator,
            parameters=parameter_vector,
            x_size=self._x_size,
            z_size=self._z_size,
            q_size=self._q_size,
        )
        projected_parameters = self.point_evaluator.project_parameters(
            parameter_vector
        )
        return VariationalSolution(
            cost=loss,
            projectors=self._projectors,
            parameter_solutions=self.parameter_solution_map(free_parameters),
            parameters=np.array(projected_parameters, dtype=float).reshape(
                (-1,)
            ),
            path=path,
            control_solutions=[],
            output=self.problem.system.y,
            t_final=self.problem.t_final,
            solve_info=solve_info,
            path_constraint_exprs=self.problem.path_constraints,
            terminal_constraint_exprs=self.problem.terminal_constraints,
        )

    def _build_projectors(self):
        proj_x = np.hstack(
            [
                np.eye(self._x_size),
                np.zeros((self._x_size, self._z_size + self._q_size)),
            ]
        )
        proj_z = (
            np.hstack(
                [
                    np.zeros((self._z_size, self._x_size)),
                    np.eye(self._z_size),
                    np.zeros((self._z_size, self._q_size)),
                ]
            )
            if self._z_size > 0
            else None
        )
        proj_q = (
            np.hstack(
                [
                    np.zeros((self._q_size, self._x_size + self._z_size)),
                    np.eye(self._q_size),
                ]
            )
            if self._q_size > 0
            else None
        )
        return proj_x, proj_z, proj_q


def _supported_solver_mode(problem: VariationalProblem) -> str:
    return problem.transcription_options.optimiser_options.get(
        VARIATIONAL_SOLVER_OPTION,
        COLLOCATION_SOLVER,
    )


def supports_dae_optimiser(problem: VariationalProblem) -> bool:
    if problem.control:
        return False
    if problem.path_constraints or problem.terminal_constraints:
        return False
    return True


def uses_dae_optimiser(problem: VariationalProblem) -> bool:
    mode = _supported_solver_mode(problem)
    if mode == COLLOCATION_SOLVER:
        return False
    if mode != DAE_OPTIMISER_SOLVER:
        raise ValueError(
            f"Unknown variational solver mode: {mode}. Expected one of "
            f"{DAE_OPTIMISER_SOLVER}, {COLLOCATION_SOLVER}."
        )
    if not supports_dae_optimiser(problem):
        raise ValueError(
            "The CasADi DAE optimiser only supports control-free "
            "variational problems with parameter bounds or fixed parameters."
        )
    return True


def create_dae_optimiser(problem: VariationalProblem) -> CasadiDaeOptimiser:
    casadi = get_backend_by_name("casadi")
    (
        p,
        p_symbols,
        _,
        (p_lower_base, p_guess_base, p_upper_base),
        p_output_map,
    ) = construct_parameters(problem.parameters)
    parameter_names = list(p_output_map.indices)
    parameter_indices = dict(p_output_map.indices)
    parameter_projector = _parameter_projector(problem, p.shape[0])

    point_evaluator = DaePointEvaluator(
        problem=problem,
        parameter_projector=parameter_projector,
    )

    output_function = ca.Function(
        "dae_optimiser_output",
        [p_symbols],
        [p, p_symbols],
        {},
    )
    assemble_solution = DaeSolutionAssembler(
        problem=problem,
        output_function=output_function,
        point_evaluator=point_evaluator,
        parameter_solution_map=p_output_map,
    )

    symbolic_output_cache: Dict[float, ca.MX] = {}

    def solution_proxy(time, p_val):
        if isinstance(time, (ca.MX, ca.SX)):
            raise TypeError(
                "The CasADi DAE optimiser requires concrete evaluation "
                "times in the loss function"
            )
        scalar_time = float(time)
        cached_output = symbolic_output_cache.get(scalar_time)
        if cached_output is not None:
            return cached_output
        value = point_evaluator.output_at(scalar_time, p_val)
        symbolic_output_cache[scalar_time] = value
        return value

    (cost,) = casadi.evaluate(problem.loss, [solution_proxy, p])
    lbg = ca.DM.zeros(0, 1)
    ubg = ca.DM.zeros(0, 1)

    solver_options = dict(problem.transcription_options.optimiser_options)
    warm_start = bool(solver_options.pop("warm_start", False))
    solver_options.pop(VARIATIONAL_SOLVER_OPTION, None)
    if not problem.transcription_options.verbose:
        solver_options.update(
            {
                "ipopt.print_level": 0,
                "print_time": False,
                "ipopt.sb": "yes",
            }
        )

    callback_wrapper = None
    nlp_solver_options = dict(solver_options)
    if warm_start:
        nlp_solver_options["ipopt.warm_start_init_point"] = "yes"
    if problem.transcription_options.interation_callback is not None:
        callback_wrapper = CallbackWrapper.new(
            "dae_optimiser_iteration_callback",
            problem.transcription_options.interation_callback,
            nx=p_symbols.shape[0],
            ng=0,
            assemble_solution=assemble_solution,
        )
        nlp_solver_options["iteration_callback"] = callback_wrapper

    nlp_spec = {"f": cost, "x": p_symbols, "g": ca.MX.zeros(0, 1)}
    init_solver = None
    if problem.transcription_options.initialise_near_guess:
        init_solver = ca.nlpsol(
            "dae_optimiser_initialiser",
            "ipopt",
            nlp_spec,
            dict(solver_options),
        )
    nlp_solver = ca.nlpsol(
        "dae_optimiser",
        "ipopt",
        nlp_spec,
        nlp_solver_options,
    )

    def map_arguments(
        fixed_parameters: Dict[str, ParameterVariable],
    ) -> Dict[str, ca.DM]:
        unknown = sorted(set(fixed_parameters) - set(parameter_indices))
        if unknown:
            raise KeyError(f"Unknown solver parameters: {', '.join(unknown)}")

        x0 = ca.DM(p_guess_base)
        lbx = ca.DM(p_lower_base)
        ubx = ca.DM(p_upper_base)

        for name, value in fixed_parameters.items():
            index = parameter_indices[name]
            if isinstance(value, BoundedVariable):
                x0[index] = value.guess
                lbx[index] = value.lower_bound
                ubx[index] = value.upper_bound
            else:
                scalar = float(value)
                x0[index] = scalar
                lbx[index] = scalar
                ubx[index] = scalar

        x0 = ca.fmin(ca.fmax(x0, lbx), ubx)
        return {
            "x0": x0,
            "lbx": lbx,
            "ubx": ubx,
            "lbg": lbg,
            "ubg": ubg,
            "init_lbg": lbg,
            "init_ubg": ubg,
        }

    solver = CasadiDaeOptimiser(
        problem=problem,
        parameters=parameter_names,
        map_arguments=map_arguments,
        solver=nlp_solver,
        assemble_solution=assemble_solution,
        initialiser=init_solver,
        warm_start=warm_start,
    )
    solver._callback_wrapper = callback_wrapper
    return solver


def _parameter_projector(
    problem: VariationalProblem, parameter_size: int
) -> ca.DM:
    if problem.system_parameter_map is None:
        return ca.DM.eye(parameter_size)
    return ca.DM(problem.system_parameter_map)
