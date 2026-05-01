from typing import Callable, List, Dict, Optional, Tuple

import casadi as ca
import numpy as np
from itertools import accumulate

from coker.backends.backend import get_backend_by_name


from coker.dynamics import (
    VariationalProblem,
    split_at_non_differentiable_points,
    ControlVariable,
    ParameterVariable,
    ConstantControlVariable,
    SpikeVariable,
    BoundedVariable,
    PiecewiseConstantVariable,
    InterpolatingPoly,
    InterpolatingPolyCollection,
    VariationalSolution,
)


def noop(*args):
    return None


def create_variational_solver(problem: VariationalProblem):

    # At each point t_i in segment j
    # ([D_{ij}(t_i)] * 1_{n})  X_j - (dt) f(t_i, X_{ij}, Z_{ij}, U(t_i), p) = 0
    #                                     g(t_i, X_{ij}, Z_{ij}, U(t_i), p) = 0
    # ([D_{ij}(t_i)] * 1_{n})  Q_j - (dt) h(t_i, X_{ij}, Z_{ij}, U(t_i), p) = 0

    # If U(t) is discontinuous, it is evaluated as if from below at t_i = 1
    # and from above at t_i = -1

    # At the end of each segment, we also have the constraint
    # [X, Z, Q]_{j - 1}(1) = [X, Z, Q]_{j}(-1),

    # with the [X]_{-1} = [X0(0, U(0), p)]
    #                 0 = g(0, X0(0,u,p), Z, U(0))
    #            Q_{-1} = 0

    # cost = cost(y(t, x, z, q, u, p), u, p)
    # calling cost symbolically, we should know the set of $t_i$ that are
    # required

    casadi = get_backend_by_name("casadi")

    x_dim, z_dim, q_dim = problem.system.get_state_dimensions()
    x_size = x_dim.flat()
    z_size = z_dim.flat() if z_dim else 0
    q_size = q_dim.flat() if q_dim else 0

    tolerance = problem.transcription_options.absolute_tolerance

    # initial intervals
    intervals = split_at_non_differentiable_points(
        problem.control if problem.control else [],
        problem.t_final,
        problem.transcription_options,
    )

    colocation_points = [problem.transcription_options.minimum_degree] * len(
        intervals
    )
    poly_collection = SymbolicPolyCollection(
        name="x",
        dimension=x_size + z_size + q_size,
        intervals=intervals,
        degrees=colocation_points,
    )
    time_points, _, _ = zip(*list(poly_collection.knot_points()))

    proj_x = ca.hcat([ca.MX.eye(x_size), ca.MX.zeros(q_size, z_size)])
    proj_z = ca.hcat(
        [
            ca.MX.zeros(z_size, x_size),
            ca.MX.eye(z_size),
            ca.MX.zeros(z_size, q_size),
        ]
    )
    proj_q = ca.hcat(
        [
            ca.MX.zeros(q_size, x_size),
            ca.MX.zeros(q_size, z_size),
            ca.MX.eye(q_size),
        ]
    )

    # Set up functions
    equalities = []

    def dynamics(*args):
        return casadi.evaluate(problem.system.dxdt, args)

    def algebraic(*args):
        if problem.system.g:
            return casadi.evaluate(problem.system.g, args)
        return noop

    def quadrature(*args):
        if problem.system.dqdt:
            return casadi.evaluate(problem.system.dqdt, args)
        return noop

    # parameters & control - need to know how big they are
    # to construct our decision variables
    control_variables, parameters = problem.control, problem.parameters

    p, p_symbols, p0_guess, (p_lower, p_guess, p_upper), p_output_map = (
        construct_parameters(parameters)
    )
    if problem.system_parameter_map is not None:
        proj_p = ca.DM(problem.system_parameter_map)
    else:
        proj_p = ca.DM.eye(p.shape[0])

    control_factory = (
        ControlFactory(control_variables, problem.t_final)
        if control_variables
        else noop
    )

    if problem.control:
        u_symbols = control_factory.symbols()
        u_lower, u_upper = (
            control_factory.lower_bounds,
            control_factory.upper_bounds,
        )
        u_guess = control_factory.guess
    else:
        u_symbols = ca.MX.zeros(0)
        u_lower, u_upper = [], []
        u_guess = noop

    (t0, x0_symbol), *x_start = list(poly_collection.interval_starts())
    x_end = list(poly_collection.interval_ends())[:-1]

    x0_guess, z0_guess = casadi.evaluate(
        problem.system.x0, [0, u_guess, proj_p @ p0_guess]
    )
    x0_val, z0_val = casadi.evaluate(
        problem.system.x0, [0, control_factory, proj_p @ p]
    )

    equalities += [
        proj_x @ x0_symbol - x0_val,
    ]
    if z_size > 0:
        equalities.append(proj_z @ z0_val)

    equalities += [
        xs_i - xe_i for ((_, xs_i), (_, xe_i)) in zip(x_start, x_end)
    ]
    for poly in poly_collection.polys:
        interval_dynamics = []
        interval_quadratures = []
        for t, v, dv in poly.knot_points():
            x = proj_x @ v
            z = proj_z @ v
            if z_size > 0:
                z += z0_val

            dx = proj_x @ dv

            (dynamics_ij,) = dynamics(t, x, z, control_factory, proj_p @ p)
            interval_dynamics.append(dynamics_ij)
            equalities.append(dx - dynamics_ij)

            if q_size > 0:
                dq = proj_q @ dv
                (quadrature_ij,) = quadrature(
                    t, x, z, control_factory, proj_p @ p
                )
                equalities.append(dq - quadrature_ij)
                interval_quadratures.append(quadrature_ij)

            if z_size > 0:
                (alg,) = algebraic(t, x, z, control_factory, proj_p @ p)
                equalities.append(alg)

            # xend = x_start + int_tstart^t_end f(x)dt
            # ->  0 = xend - xstart - [dx_0 | dx_1 | dx_2 | ...] @ w

        _, vstart = poly.start_point()
        _, vend = poly.end_point()

        dX = ca.hcat(interval_dynamics)
        assert poly.weights[0, poly.degree] == 0
        w = ca.DM(poly.weights[0, :-1])

        equalities.append(proj_x @ vend - proj_x @ vstart - (dX @ w))
        if q_size > 0:
            dQ = ca.hcat(interval_quadratures)
            equalities.append(proj_q @ vend - proj_q @ vstart - dQ @ w)

    path_symbols = poly_collection.symbols()

    decision_variables = ca.vertcat(path_symbols, u_symbols, p_symbols)

    lower_bound = ca.vertcat(
        -ca.DM.ones(poly_collection.size()) * ca.inf, *u_lower, p_lower
    )

    upper_bound = ca.vertcat(
        ca.DM.ones(poly_collection.size()) * ca.inf, *u_upper, p_upper
    )

    # cost
    #
    def solution_proxy(*args):
        if problem.control:
            tau, u_val, p_val = args
        else:
            tau, p_val = args
            u_val = ca.MX.zeros(u_symbols.shape)
        inner = poly_collection(tau)
        x_tau = proj_x @ inner
        z_tau = proj_z @ inner
        q_tau = proj_q @ inner
        (y_val,) = casadi.evaluate(
            problem.system.y, [tau, x_tau, z_tau, u_val, proj_p @ p_val, q_tau]
        )
        return y_val

    if problem.control:
        (cost,) = casadi.evaluate(
            problem.loss, [solution_proxy, control_factory, p]
        )
    else:
        (cost,) = casadi.evaluate(problem.loss, [solution_proxy, p])

    g = ca.vertcat(*[e for e in equalities if e is not None])
    ubg = tolerance * ca.DM.ones(g.shape)
    lbg = -tolerance * ca.DM.ones(g.shape)

    t_end, v_end = poly_collection.polys[-1].end_point()
    x_end = proj_x @ v_end
    z_end = proj_z @ v_end
    q_end = proj_q @ v_end
    u_end = control_factory(t_end, x_end, z_end)
    end_args = (t_end, x_end, z_end, u_end, p, q_end)

    for constraint in problem.terminal_constraints:
        (g_inner,) = casadi.evaluate(constraint.value, end_args)
        g_lower = casadi.to_backend_array(constraint.lower)
        g_upper = casadi.to_backend_array(constraint.upper)
        assert g_lower.shape == g_inner.shape == g_inner.shape
        g = ca.vertcat(g, g_inner)
        lbg = ca.vertcat(lbg, g_lower)
        ubg = ca.vertcat(ubg, g_upper)

    projectors = tuple(
        _to_output_projector(proj) for proj in (proj_x, proj_z, proj_q)
    )

    solver_options = dict(problem.transcription_options.optimiser_options)
    if not problem.transcription_options.verbose:
        solver_options.update(
            {
                "ipopt.print_level": 0,
                "print_time": False,
                "ipopt.sb": "yes",
            }
        )

    init_solver_options = dict(solver_options)
    nlp_solver_options = dict(solver_options)

    f_out = ca.Function(
        "Output",
        [decision_variables],
        [path_symbols, u_symbols, p],
        {},
    )

    callback_wrapper = None
    if problem.transcription_options.interation_callback is not None:
        callback_wrapper = CallbackWrapper.new(
            "variational_iteration_callback",
            problem.transcription_options.interation_callback,
            nx=decision_variables.shape[0],
            ng=g.shape[0],
            f_out=f_out,
            poly_collection=poly_collection,
            projectors=projectors,
            proj_p=proj_p,
            output_function=problem.system.y,
            path_constraints=problem.path_constraints,
            terminal_constraints=problem.terminal_constraints,
            t_final=problem.t_final,
            decode_controls=(
                control_factory.to_output_array if problem.control else None
            ),
        )
        nlp_solver_options["iteration_callback"] = callback_wrapper

    u0_guess = ca.DM.zeros(u_symbols.shape)

    state_guess = ca.vertcat(
        x0_guess,
        z0_guess if z0_guess is not None else ca.DM.zeros(z_size),
        ca.DM.zeros(q_size),
    )

    n_reps = int(path_symbols.shape[0] / state_guess.shape[0])

    decision_variables_0 = ca.vertcat(
        ca.repmat(state_guess, n_reps), u0_guess, p_guess
    )

    if problem.transcription_options.initialise_near_guess:

        init_spec = {
            "f": cost,
            "x": decision_variables,
            "g": ca.vertcat(g, p_symbols, u_symbols),
        }
        init_solver = ca.nlpsol(
            "initialiser", "ipopt", init_spec, init_solver_options
        )

        init_soln = init_solver(
            x0=decision_variables_0,
            lbx=lower_bound,
            ubx=upper_bound,
            lbg=ca.vertcat(lbg, p_guess, ca.DM.zeros(u_symbols.shape)),
            ubg=ca.vertcat(ubg, p_guess, ca.DM.zeros(u_symbols.shape)),
        )
        decision_variables_0 = init_soln["x"]
        initial_cost = float(init_soln["f"])
        assert (
            initial_cost <= ca.inf
        ), f"Cost at guess {initial_cost} is not finite"

    nlp_spec = {"f": cost, "x": decision_variables, "g": g}
    nlp_solver = ca.nlpsol("solver", "ipopt", nlp_spec, nlp_solver_options)
    soln = nlp_solver(
        x0=decision_variables_0,
        lbx=lower_bound,
        ubx=upper_bound,
        lbg=lbg,
        ubg=ubg,
    )

    min_loss = float(soln["f"])
    min_args = soln["x"]

    x_out, u_out, p_out = f_out(min_args)

    path = poly_collection.to_fixed(np.array(x_out))
    p_out = np.array(p_out)
    parameter_out = p_output_map(p_out)

    control_out = (
        control_factory.to_output_array(u_out) if problem.control else None
    )

    solution = VariationalSolution(
        cost=min_loss,
        projectors=projectors,
        parameter_solutions=parameter_out,
        parameters=p_out.flatten(),
        path=path,
        control_solutions=control_out,
        output=problem.system.y,
    )

    return solution


class SymbolicPoly(InterpolatingPoly):
    def __init__(self, name, dimension, interval, degree):
        size = (degree + 1) * dimension
        values = ca.MX.sym(name, size)
        super().__init__(dimension, interval, degree, values)

    def symbols(self):
        return self.values

    def __call__(self, t):
        s = self._interval_to_s(t)
        try:
            i = next(i for i, s_i in enumerate(self.s) if abs(s_i - s) < 1e-9)
            return self.values[i * self.dimension : (i + 1) * self.dimension]
        except (StopIteration, TypeError):
            pass
        n = len(self.s)
        s_vector = ca.vertcat(*[s**i for i in range(n)])

        projection = s_vector.T @ ca.DM(self.bases)

        # Casadi's reshape follows fortran convention (unlike numpy),
        # so we need to transpose the operation
        value = ca.reshape(self.values, (self.dimension, -1)) @ projection.T

        return ca.reshape(value, (self.dimension, 1))


class SymbolicPolyCollection(InterpolatingPolyCollection):
    def symbols(self):
        return ca.vertcat(*[p.values for p in self.polys])

    def __init__(self, name, dimension, intervals, degrees):
        assert len(intervals) == len(degrees)
        polys = [
            SymbolicPoly(f"{name}_{i}", dimension, interval, degree)
            for i, (interval, degree) in enumerate(zip(intervals, degrees))
        ]
        super().__init__(polys)

    def to_fixed(self, array):
        size = sum(p.size() for p in self.polys)
        np_array = np.array(array)
        assert np_array.shape == (size, 1)
        np_array.reshape((size,))

        slices = []
        offset = 0
        for p in self.polys:
            slices.append(slice(offset, offset + p.size()))
            offset += p.size()

        polys = [
            InterpolatingPoly(p.dimension, p.interval, p.degree, np_array[slc])
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


def construct_parameters(parameters: List[ParameterVariable]):

    params = []
    upper_bounds = []
    guess = []
    lower_bounds = []
    symbols = {}
    p0 = []
    output_map = {}
    for i, p in enumerate(parameters):
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
            output_map[p.name] = len(params)
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

    params = ca.vertcat(*params)

    symbols = ca.vertcat(*symbols.values())
    return (
        params,  # actual parameter vector
        symbols,  # symbols
        ca.DM(p0),  # actual parameter vector guess
        (
            ca.DM(lower_bounds),
            ca.DM(guess),
            ca.DM(upper_bounds),
        ),  # symbols bounds and guess
        ParameterOutputMap(output_map),  # map to symbols
    )


class ControlPath:
    def __init__(self, vector: ca.MX, rate: float):
        self.vector = vector
        self.rate = rate

    def __call__(self, t) -> ca.MX:
        index = int(t * self.rate)
        return self.vector[index]


class ControlFactory:
    def __init__(self, variables: List[ControlVariable], t_final: float):
        self.t_final = t_final
        self.variables = variables
        self._symbols = [
            ca.MX.sym(v.name, v.degrees_of_freedom(0, t_final))
            for v in variables
        ]
        self.values = []
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
        self.offsets = accumulate(self.sizes)

    def guess(self, _):
        return ca.DM.zeros(len(self.variables))

    def symbols(self) -> ca.MX:
        return ca.vertcat(*self._symbols)

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
        return ca.vertcat(*out)

    def to_output_array(self, solution: ca.DM):
        return [
            v.to_solution(solution[offset : offset + size])
            for v, offset, size in zip(
                self.variables, self.offsets, self.sizes
            )
        ]


def _to_output_projector(proj: ca.MX) -> Optional[np.ndarray]:
    if proj.shape == (0, 1):
        return None
    return np.array(proj.to_DM()).reshape(proj.shape)


def _to_flat_array(value) -> np.ndarray:
    if value is None:
        return np.zeros((0,))

    if isinstance(value, (tuple, list)):
        assert len(value) == 1, "Expected a single output value"
        (value,) = value

    array = np.array(value, dtype=float)
    return array.reshape((-1,))


def _evaluate_violation(raw_value, lower, upper) -> np.ndarray:
    values = _to_flat_array(raw_value)
    lower_bounds = _to_flat_array(lower)
    upper_bounds = _to_flat_array(upper)
    assert (
        values.shape == lower_bounds.shape == upper_bounds.shape
    ), "Constraint bounds do not match constraint values"

    violations = []
    for value_i, lower_i, upper_i in zip(values, lower_bounds, upper_bounds):
        has_lower = np.isfinite(lower_i)
        has_upper = np.isfinite(upper_i)
        if has_lower and has_upper:
            raise ValueError(
                "Variational iteration callbacks only support half-space "
                "constraints per component"
            )
        if has_lower:
            violations.append(max(lower_i - value_i, 0.0))
        elif has_upper:
            violations.append(max(value_i - upper_i, 0.0))

    if not violations:
        return np.zeros((0,))

    return np.array(violations, dtype=float)


class CallbackControlProxy:
    def __init__(self, control_solutions):
        self.control_solutions = control_solutions

    def __call__(self, t) -> np.ndarray:
        return np.array([control(t) for control in self.control_solutions])


class CallbackTrajectoryProxy:
    def __init__(
        self,
        path: InterpolatingPolyCollection,
        evaluator: Callable[[float], np.ndarray],
    ):
        self.path = path
        self.intervals = path.intervals
        self._evaluator = evaluator

    def __call__(self, t) -> np.ndarray:
        return _to_flat_array(self._evaluator(t))

    def interval_starts(self):
        for t, _ in self.path.interval_starts():
            yield t, self(t)

    def interval_ends(self):
        for t, _ in self.path.interval_ends():
            yield t, self(t)

    def knot_points(self):
        for t, _, _ in self.path.knot_points():
            yield t, self(t), None


class _CallbackIterate:
    def __init__(
        self,
        *,
        path: InterpolatingPolyCollection,
        projectors: Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
        ],
        control_solutions,
        parameters: np.ndarray,
        system_parameters: np.ndarray,
        output_function,
        path_constraints,
        terminal_constraints,
    ):
        self.path = path
        self.projectors = projectors
        self.control_solutions = control_solutions
        self.parameters = parameters
        self.system_parameters = system_parameters
        self.output_function = output_function
        self.path_constraints = path_constraints
        self.terminal_constraints = terminal_constraints

    def state(self, t) -> np.ndarray:
        projector = self.projectors[0]
        assert projector is not None
        return projector @ self.path(t)

    def algebraic(self, t) -> Optional[np.ndarray]:
        projector = self.projectors[1]
        if projector is None:
            return None
        return projector @ self.path(t)

    def quadratures(self, t) -> Optional[np.ndarray]:
        projector = self.projectors[2]
        if projector is None:
            return None
        return projector @ self.path(t)

    def control_law(self, t) -> Optional[np.ndarray]:
        if self.control_solutions is None:
            return None
        return np.array([control(t) for control in self.control_solutions])

    def _args_at(self, t):
        return (
            t,
            self.state(t),
            self.algebraic(t),
            self.control_law(t),
            self.system_parameters,
            self.quadratures(t),
        )

    def output(self, t) -> np.ndarray:
        return _to_flat_array(self.output_function(*self._args_at(t)))

    def constraint_violations(self, constraints, t) -> np.ndarray:
        if not constraints:
            return np.zeros((0,))

        args = self._args_at(t)
        violations = [
            _evaluate_violation(
                constraint.value(*args), constraint.lower, constraint.upper
            )
            for constraint in constraints
        ]
        return np.concatenate(violations) if violations else np.zeros((0,))

    def path_constraint_trajectory(self) -> CallbackTrajectoryProxy:
        return CallbackTrajectoryProxy(
            self.path,
            lambda t: self.constraint_violations(self.path_constraints, t),
        )

    def terminal_constraint_vector(self, t_final: float) -> np.ndarray:
        return self.constraint_violations(self.terminal_constraints, t_final)


class CallbackWrapper(ca.Callback):
    """Wrap a CasADi NLP iteration callback into the variational
    callback API."""

    def __init__(
        self,
        name: str,
        callback,
        *,
        nx: int,
        ng: int,
        f_out: ca.Function,
        poly_collection: SymbolicPolyCollection,
        projectors: Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
        ],
        proj_p: ca.DM,
        output_function,
        path_constraints,
        terminal_constraints,
        t_final: float,
        decode_controls: Optional[Callable[[ca.DM], list]],
        opts=None,
    ):
        ca.Callback.__init__(self)
        self.callback = callback
        self.nx = nx
        self.ng = ng
        self.f_out = f_out
        self.poly_collection = poly_collection
        self.projectors = projectors
        self.proj_p = proj_p
        self.output_function = output_function
        self.path_constraints = path_constraints
        self.terminal_constraints = terminal_constraints
        self.t_final = t_final
        self.decode_controls = decode_controls
        self.construct(name, {} if opts is None else opts)

    @staticmethod
    def new(*args, **kwargs) -> "CallbackWrapper":
        return CallbackWrapper(*args, **kwargs)

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
        decision_variables = darg["x"]
        loss = float(darg["f"])

        path_coefficients, control_coefficients, parameters = self.f_out(
            decision_variables
        )
        path = self.poly_collection.to_fixed(np.array(path_coefficients))
        parameter_vector = np.array(parameters, dtype=float).reshape((-1,))
        system_parameters = np.array(
            self.proj_p @ ca.DM(parameter_vector.reshape((-1, 1)))
        ).reshape((-1,))
        control_solutions = (
            self.decode_controls(control_coefficients)
            if self.decode_controls is not None
            else None
        )

        iterate = _CallbackIterate(
            path=path,
            projectors=self.projectors,
            control_solutions=control_solutions,
            parameters=parameter_vector,
            system_parameters=system_parameters,
            output_function=self.output_function,
            path_constraints=self.path_constraints,
            terminal_constraints=self.terminal_constraints,
        )

        x = CallbackTrajectoryProxy(path, iterate.state)
        z = (
            CallbackTrajectoryProxy(path, iterate.algebraic)
            if self.projectors[1] is not None
            else None
        )
        q = (
            CallbackTrajectoryProxy(path, iterate.quadratures)
            if self.projectors[2] is not None
            else None
        )
        u = (
            CallbackControlProxy(control_solutions)
            if control_solutions is not None
            else None
        )
        y = CallbackTrajectoryProxy(path, iterate.output)
        c_path = iterate.path_constraint_trajectory()
        c_terminal = iterate.terminal_constraint_vector(self.t_final)

        should_continue = bool(
            self.callback(
                x, z, q, parameter_vector, u, y, loss, c_path, c_terminal
            )
        )
        return [0 if should_continue else 1]
