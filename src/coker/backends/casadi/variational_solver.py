from typing import List

import casadi as ca
import numpy as np
from itertools import accumulate


from coker.backends.backend import get_backend_by_name
from coker.backends.casadi import substitute

from coker.dynamics import (
    VariationalProblem,
    split_at_non_differentiable_points,
    ControlVariable,
    ParameterVariable,
    ConstantControlVariable,
    SpikeVariable,
    ParameterMixin,
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

    # parameters & control - need to know how big they are
    # to construct our decision variables
    control_variables, parameters = problem.control, problem.parameters

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

    p, p_symbols, p0, (p_lower, p_guess, p_upper), p_output_map = (
        construct_parameters(parameters)
    )

    control_factory = (
        ControlFactory(control_variables, problem.t_final)
        if control_variables
        else noop
    )

    (t0, x0_symbol), *x_start = list(poly_collection.interval_starts())

    x_end = list(poly_collection.interval_ends())[:-1]

    x0_val, z0_val = casadi.evaluate(
        problem.system.x0, [0, control_factory, p]
    )

    equalities += [
        proj_x @ x0_symbol,
    ]
    if z_size > 0:
        equalities.append(proj_z @ z0_val)

    equalities += [
        xs_i - xe_i for ((_, xs_i), (_, xe_i)) in zip(x_start, x_end)
    ]

    for t, v, dv in poly_collection.knot_points():
        x = proj_x @ v + x0_val
        z = proj_z @ v
        if z_size > 0:
            z += z0_val

        dx = proj_x @ dv
        (dynamics_ij,) = dynamics(t, x, z, control_factory, p)
        equalities.append(dx - dynamics_ij)

        if q_size > 0:
            dq = proj_q @ dv
            (quadrature_ij,) = quadrature(t, x, z, control_factory, p)
            equalities.append(dq - quadrature_ij)

        if z_size > 0:
            (alg,) = algebraic(t, x, z, control_factory, p)
            equalities.append(alg)

    path_symbols = poly_collection.symbols()

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
        x_tau = proj_x @ inner + x0_val
        if z_size > 0:
            z_tau = proj_z @ inner + z0_val
        else:
            z_tau = proj_z @ inner
        q_tau = proj_q @ inner

        return casadi.evaluate(
            problem.system.y, [tau, x_tau, z_tau, u_val, p_val, q_tau]
        )[0]

    if problem.control:
        cost = problem.loss(solution_proxy, control_factory, p)
    else:
        cost = problem.loss(solution_proxy, p)
    g = ca.vertcat(*[e for e in equalities if e is not None])
    equality_bounds = ca.DM.ones(g.shape)
    ubg = 0 * equality_bounds
    lbg = 0 * equality_bounds

    solver_options = {}
    nlp_spec = {"f": cost, "x": decision_variables, "g": g}
    nlp_solver = ca.nlpsol("solver", "ipopt", nlp_spec)

    u0_guess = ca.DM.zeros(u_symbols.shape)

    x0_guess, z0_guess = casadi.evaluate(problem.system.x0, [0, u_guess, p0])
    state_guess = ca.vertcat(
        x0_guess,
        z0_guess if z0_guess is not None else ca.DM.zeros(z_size),
        ca.DM.zeros(q_size),
    )
    n_reps = int(path_symbols.shape[0] / state_guess.shape[0])

    decision_variables_0 = ca.vertcat(
        ca.repmat(state_guess, n_reps), u0_guess, p_guess
    )

    soln = nlp_solver(
        x0=decision_variables_0,
        lbx=lower_bound,
        ubx=upper_bound,
        lbg=lbg,
        ubg=ubg,
    )

    min_loss = float(soln["f"])
    min_args = soln["x"]
    offset = ca.vertcat(
        ca.repmat(
            ca.vertcat(
                x0_val,
                z0_val if z_size else ca.MX.zeros(z_size),
                ca.MX.zeros(q_size),
            ),
            n_reps,
        )
    )

    f_out = ca.Function(
        "Output",
        [decision_variables],
        [path_symbols + offset, u_symbols, p_symbols],
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
    parameter_out = p_output_map(np.array(p_out))
    control_out = (
        control_factory.to_output_array(u_out) if problem.control else None
    )

    solution = VariationalSolution(
        cost=min_loss,
        projectors=projectors,
        parameters=parameter_out,
        path=path,
        control_solutions=control_out,
        output=lambda t, x, z, u, q: problem.system.y(t, x, z, u, p_out, q),
    )

    return solution


class SymbolicPoly(InterpolatingPoly):
    def __init__(self, name, dimension, interval, degree):
        size = (degree + 1) * dimension
        values = ca.MX.sym(name, size)
        super().__init__(dimension, interval, degree, values)

    def __call__(self, t):
        assert (
            self.interval[0] <= t <= self.interval[1]
        ), f"Value {t} is not in interval {self.interval}"
        s = self._interval_to_s(t)
        try:
            i = next(i for i, s_i in enumerate(self.s) if abs(s_i - s) < 1e-9)
            return self.values[i * self.dimension : (i + 1) * self.dimension]
        except StopIteration:
            pass
        n = len(self.s)
        s_vector = ca.vertcat(*[s**i for i in range(n)]).reshape((1, n))

        projection = s_vector @ ca.DM(self.bases)

        value = ca.repmat(projection, 1, self.dimension) @ self.values
        return value

    def knot_points(self):
        # we skip the end point
        t_i = self.knot_times()[:-1]
        n = len(t_i)
        x_i = [
            self.values[i * self.dimension : (i + 1) * self.dimension]
            for i in range(n)
        ]
        x_mat = ca.horzcat(*x_i).T
        dx_i = [(ca.DM(dbasis) @ x_mat).T for dbasis in self.derivatives]

        return t_i, x_i, dx_i


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


class ParameterOutputMap:
    def __init__(self, names):
        self.names = names

    def __call__(self, value: ca.DM):
        return {name: float(value[i, 0]) for i, name in enumerate(self.names)}


def construct_parameters(parameters: List[ParameterVariable]):

    params = []
    upper_bounds = []
    guess = []
    lower_bounds = []
    symbols = {}
    p0 = []
    for i, p in enumerate(parameters):
        if isinstance(p, BoundedVariable):
            symbol = ca.MX.sym(f"{p.name}")
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
    output_map = ParameterOutputMap(list(symbols.keys()))

    symbols = ca.vertcat(*symbols.values())
    return (
        params,
        symbols,
        ca.DM(p0),
        (ca.DM(lower_bounds), ca.DM(guess), ca.DM(upper_bounds)),
        output_map,
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
