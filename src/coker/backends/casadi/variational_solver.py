from itertools import accumulate
from typing import Tuple, List

import casadi as ca

from coker.backends.backend import get_backend_by_name



from coker.dynamics import (
    VariationalProblem,
    split_at_non_differentiable_points,
    generate_discritisation_operators,
    ControlVariable, ParameterVariable, ConstantControlVariable, SpikeVariable, ParameterMixin, BoundedVariable,
    PiecewiseConstantVariable
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
        problem.arguments[0], problem.t_final, problem.transcription_options
    )
    colocation_points = [problem.transcription_options.minimum_degree] * len(
        intervals
    )
    poly_collection = InterpolatingPolyCollection(
        name="x",
        dimension=x_size + z_size + q_size,
        intervals=intervals,
        degrees=colocation_points
    )

    proj_x = ca.hcat([ca.MX.eye(x_size), ca.MX.zeros(q_size, z_size)])
    proj_z = ca.hcat([ca.MX.zeros(z_size, x_size), ca.MX.eye(z_size), ca.MX.zeros(z_size, q_size)])
    proj_q = ca.hcat([ca.MX.zeros(q_size, x_size), ca.MX.zeros(q_size, z_size), ca.MX.eye(q_size)])

    # parameters & control - need to know how big they are
    # to construct our decision variables
    control_variables, parameters = problem.arguments

    constraints = []
    equalities = []

    dynamics = lambda *args: casadi.evaluate(problem.system.dxdt, args)
    algebraic = lambda *args: casadi.evaluate(problem.system.g, args) if problem.system.g else noop
    quadrature = lambda *args: casadi.evaluate(problem.system.dqdt, args) if problem.system.dqdt else noop

    p, p_symbols, (p_lower, p_guess, p_upper), p_output_map = construct_parameters(parameters)
    u, u_symbols, (u_lower, u_guess, u_upper), u_output_map = construct_controller(control_variables, problem.t_final)

    (t0, x0_symbol), *x_start = list(poly_collection.interval_starts())
    
    x_end = list(poly_collection.interval_ends())[:-1]
    z0_symbol = proj_z @ x0_symbol if z_size > 0 else 0

    x0_val, = casadi.evaluate(problem.system.x0, [0, z0_symbol, u, p])

    equalities +=[
        proj_x @ x0_symbol - x0_val,
        # z_0 is free
        proj_q @ x0_symbol
    ]
    
    equalities += [
        xs_i - xe_i for ((_, xs_i),(_, xe_i)) in zip(x_start, x_end)
    ]
    
    for t, v, dv in poly_collection.knot_points():
        x = proj_x @ v
        z = proj_z @ v

        dx = proj_x @ dv
        dynamics_ij, = dynamics(t, x, z, u, p)
        equalities.append(dx - dynamics_ij)

        if q_size > 0:
            dq = proj_q @ dv
            quadrature_ij, = quadrature(t, x, z, u, p)
            equalities.append(dq - quadrature_ij)

        if z_size > 0:
            alg, = algebraic(t, x, z, u, p)
            equalities.append(alg)

    decision_variables = ca.vertcat(poly_collection.symbols(), u_symbols, p_symbols)

    lower_bound = ca.vertcat(
        -ca.DM.ones(poly_collection.size())  *ca.inf,
        u_lower,
        p_lower
    )

    upper_bound = ca.vertcat(
        ca.DM.ones(poly_collection.size()) * ca.inf,
        u_upper,
        p_upper
    )
    # cost
    #
    def solution_proxy(tau, u_val, p_val):
        inner = poly_collection(tau)
        x_tau = proj_x @ inner
        z_tau = proj_z @ inner
        q_tau = proj_q @ inner

        return casadi.evaluate(
            problem.system.y, [tau, x_tau, z_tau, u_val, p_val, q_tau]
        )

    cost = problem.loss(solution_proxy, u, p)
    g = ca.vertcat(*[e for e in equalities if e  is not None])

    gub = ca.DM.zeros(g.shape)
    glb = gub

    nlp_spec = {
        'f': cost,
        'x': decision_variables,
        'g': g
    }
    nlp_solver = ca.nlpsol("solver", "ipopt", nlp_spec)



    x0_guess = casadi.evaluate(problem.system.x0, [0, z0_symbol, u_guess, p_guess])




class InterpolatingPoly:
    def __init__(self, name, dimension, interval, degree):
        op_values = generate_discritisation_operators(interval, degree)
        self.s, self.s_to_interval, bases, self.derivatives, self.integrals =  op_values
        self.symbols = ca.MX.sym(name, len(self.s) * dimension)
        self.interval = interval
        self.dim = dimension
        self.bases = ca.vertcat(
            *[ca.reshape(ca.DM(base), (1,5 )) for base in bases]
        )


    def _interval_to_s(self ,t):
        width = (self.interval[1] - self.interval[0]) / 2
        mean = (self.interval[1] + self.interval[0]) / 2
        return (t - mean) / width

    def __call__(self, t):
        assert self.interval[0] <= t <= self.interval[1], f"Value {t} is not in interval {self.interval}"
        s = self._interval_to_s(t)
        try:
            i = next(i for i, s_i in enumerate(self.s) if abs(s_i - s) < 1e-9)
            return self.symbols[i  * self.dim: (i + 1) * self.dim]
        except StopIteration:
            pass
        n = len(self.s)
        s_vector = ca.vertcat(*[s ** i for i in range(n)]).reshape((1, n))

        projection = s_vector @ self.bases

        value = ca.repmat(projection, 1, self.dim) @ self.symbols
        return value


    def start_point(self):
        return self.s_to_interval(self.s[0]), self.symbols[:self.dim]

    def end_point(self):
        return self.s_to_interval(self.s[-1]), self.symbols[-self.dim:]

    def knot_points(self):
        # we skip the end point
        n = len(self.s) - 1

        t_i = [self.s_to_interval(s_i) for s_i in self.s[:n]]
        x_i = [
            self.symbols[i * self.dim: (i + 1) * self.dim]
            for i in range(n)
        ]
        dx_i = [
            ca.repmat(d_basis, 1, self.dim) @ x_ij
            for d_basis, x_ij in zip(self.derivatives, x_i)
        ]
        return t_i, x_i, dx_i


class InterpolatingPolyCollection:
    def __init__(self, name, dimension, intervals, degrees):
        assert len(intervals) == len(degrees)
        self._size = sum(degrees) * dimension
        self.polys = [
            InterpolatingPoly(f"{name}_{i}", dimension, interval, degree)
            for i, (interval, degree) in enumerate(zip(intervals, degrees))
        ]
        self.intervals = intervals

    def size(self):
        return self._size

    def __call__(self, t):
        for i, (start, end) in enumerate(self.intervals):
            if start <= t <= end:
                return self.polys[i](t)
        raise ValueError(f"Value {t} is not in any interval")


    def interval_starts(self):
         for p in self.polys:
             yield p.start_point()

    def interval_ends(self):
        for p in self.polys:
            yield p.end_point()

    def knot_points(self):
        for p in self.polys:
            t, x, dx = p.knot_points()
            for t_i, x_i, dx_i in zip(t, x, dx):
                yield t_i, x_i, dx_i

    def symbols(self):
        return ca.vertcat(*[p.symbols for p in self.polys])




class ParameterOutputMap:
    def __init__(self, names):
        self.names = names

    def __call__(self, value: ca.DM):
        return {
            name: value[i, 0] for i, name in enumerate(self.names)
        }




def construct_parameters(parameters: List[ParameterVariable]):

    params = []
    upper_bounds = []
    guess = []
    lower_bounds = []
    symbols = {}

    for i ,p in enumerate(parameters):
        if isinstance(p, BoundedVariable):
            symbol = ca.MX.sym(f"{p.name}")
            params.append(symbol)
            symbols[p.name] = symbol
            upper_bounds.append(p.upper_bound if p.upper_bound is not None else ca.inf)
            guess.append(0)
            lower_bounds.append(p.lower_bound if p.lower_bound is not None else -ca.inf)
        elif isinstance(p, (float, int)):
            params.append(ca.MX(p))
        else:
            raise ValueError(f"Parameter {p} is not a valid parameter")

    params = ca.vertcat(*params)
    output_map = ParameterOutputMap(list(symbols.keys()))

    symbols = ca.vertcat(*symbols.values())
    return params, symbols, (lower_bounds, guess, upper_bounds), output_map


class ControlPath:
    def __init__(self, vector:ca.MX, rate: float):
        self.vector = vector
        self.rate = rate

    def __call__(self, t) -> ca.MX:
        index = int(t * self.rate)
        return self.vector[index]


class ControlFactory:
    def __init__(self, t_final: float):
        self.t_final = t_final
        self.symbols = []
        self.upper_bounds = []
        self.lower_bounds = []
        self.generators = []
        self.guess = []

    def __call__(self, t):
        assert 0 <= t <= self.t_final, f"Control variable is not defined at t = {t}"
        return ca.vertcat(*[
            generator(t) for generator in self.generators
        ])

    def add_control_variable(self, control_variable: ControlVariable):

        if isinstance(control_variable, (float, int)):
            self.generators.append(lambda t: control_variable)

        elif isinstance(control_variable, ConstantControlVariable):
            symbol = ca.MX.sym(f"{control_variable.name}")
            self.symbols.append(symbol)
            self.upper_bounds.append(control_variable.upper_bound)
            self.guess.append(0)
            self.lower_bounds.append(control_variable.lower_bound)
            self.generators.append(lambda t: symbol)

        elif isinstance(control_variable, SpikeVariable):
            symbol = ca.MX.sym(f"{control_variable.name}")
            self.symbols.append(symbol)
            self.guess.append(0)
            self.upper_bounds.append(control_variable.upper_bound)
            self.lower_bounds.append(control_variable.lower_bound)
            self.generators.append(lambda t: ca.if_else(t == control_variable.time, symbol, 0))

        elif isinstance(control_variable, PiecewiseConstantVariable):
            dimension = control_variable.degrees_of_freedom(0, self.t_final)
            symbol = ca.MX.sym(f"{control_variable.name}", dimension)
            self.symbols.append(symbol)
            self.guess += [0] * dimension
            self.upper_bounds += [control_variable.upper_bound] * dimension
            self.lower_bounds += [control_variable.lower_bound] * dimension
            self.generators.append(
                ControlPath(symbol, control_variable.sample_rate)
           )
        else:
            raise ValueError(f"Control variable {control_variable} is not a valid control variable")


    def construct_decision_variables(self):
        symbols = ca.vertcat(*self.symbols)
        lower_bounds = ca.vertcat(*self.lower_bounds)
        upper_bounds = ca.vertcat(*self.upper_bounds)
        guess = ca.vertcat(*self.guess)
        return symbols, (lower_bounds, guess, upper_bounds), self.symbols


def construct_controller(control_variables: List[ControlVariable], t_final: float):

    helper = ControlFactory(t_final)

    for c in control_variables:
        helper.add_control_variable(c)
    s, bounds, output = helper.construct_decision_variables()
    return helper, s, bounds, output
