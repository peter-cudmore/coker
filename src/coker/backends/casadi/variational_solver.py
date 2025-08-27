from typing import List, Dict

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

    constraints = []
    equalities = []

    dynamics = lambda *args: casadi.evaluate(problem.system.dxdt, args)
    algebraic = lambda *args: (
        casadi.evaluate(problem.system.g, args) if problem.system.g else noop
    )
    quadrature = lambda *args: (
        casadi.evaluate(problem.system.dqdt, args)
        if problem.system.dqdt
        else noop
    )

    # parameters & control - need to know how big they are
    # to construct our decision variables
    control_variables, parameters = problem.control, problem.parameters

    p, p_symbols, p0_guess, (p_lower, p_guess, p_upper), p_output_map = (
        construct_parameters(parameters)
    )

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
        problem.system.x0, [0, u_guess, p0_guess]
    )
    x0_val, z0_val = casadi.evaluate(
        problem.system.x0, [0, control_factory, p]
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

            (dynamics_ij,) = dynamics(t, x, z, control_factory, p)
            interval_dynamics.append(dynamics_ij)
            equalities.append(dx - dynamics_ij)

            if q_size > 0:
                dq = proj_q @ dv
                (quadrature_ij,) = quadrature(t, x, z, control_factory, p)
                equalities.append(dq - quadrature_ij)
                interval_quadratures.append(quadrature_ij)

            if z_size > 0:
                (alg,) = algebraic(t, x, z, control_factory, p)
                equalities.append(alg)

            # xend = x_start + int_tstart^t_end f(x)dt
            # ->  0 = xend - xstart - [dx_0 | dx_1 | dx_2 | ...] @ w

        _, xstart = poly.start_point()
        _, xend = poly.end_point()

        dX = ca.hcat(interval_dynamics)
        assert poly.weights[0, poly.degree] == 0
        w = ca.DM(poly.weights[0, :-1])

        equalities.append(proj_x @ xend - proj_x @ xstart - (dX @ w))
        if q_size > 0:
            dQ = ca.hcat(interval_quadratures)
            equalities.append(proj_q @ xend - proj_q @ xstart - dQ @ w)

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
            problem.system.y, [tau, x_tau, z_tau, u_val, p_val, q_tau]
        )
        return y_val

    if problem.control:
        cost = problem.loss(solution_proxy, control_factory, p)
    else:
        cost = problem.loss(solution_proxy, p)

    g = ca.vertcat(*[e for e in equalities if e is not None])
    ubg = tolerance
    lbg = -tolerance
    solver_options = problem.transcription_options.optimiser_options

    if not problem.transcription_options.verbose:
        solver_options.update(
            {
                "ipopt.print_level": 0,
                "print_time": False,
                "ipopt.sb": "yes",
            }
        )
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
            "g": ca.vertcat(g, p_symbols - p_guess, u_symbols),
        }
        init_solver = ca.nlpsol(
            "initialiser", "ipopt", init_spec, solver_options
        )
        init_soln = init_solver(
            x0=decision_variables_0,
            lbx=lower_bound,
            ubx=upper_bound,
            lbg=lbg,
            ubg=ubg,
        )
        decision_variables_0 = init_soln["x"]
        initial_cost = float(init_soln["f"])
        assert (
            initial_cost <= ca.inf
        ), f"Cost at guess {initial_cost} is not finite"
    else:
        initial_cost = ca.inf

    nlp_spec = {"f": cost, "x": decision_variables, "g": g}
    nlp_solver = ca.nlpsol("solver", "ipopt", nlp_spec, solver_options)
    soln = nlp_solver(
        x0=decision_variables_0,
        lbx=lower_bound,
        ubx=upper_bound,
        lbg=lbg,
        ubg=ubg,
    )

    min_loss = float(soln["f"])
    assert min_loss <= initial_cost
    min_args = soln["x"]

    f_out = ca.Function(
        "Output",
        [decision_variables],
        [path_symbols, u_symbols, p],
        {},
    )
    x_out, u_out, p_out = f_out(min_args)

    path = poly_collection.to_fixed(np.array(x_out))
    projectors = tuple(
        (
            np.array(proj.to_DM()).reshape(proj.shape)
            if proj.shape != (0, 1)
            else None
        )
        for proj in (proj_x, proj_z, proj_q)
    )
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
        params,
        symbols,
        ca.DM(p0),
        (ca.DM(lower_bounds), ca.DM(guess), ca.DM(upper_bounds)),
        ParameterOutputMap(output_map),
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
