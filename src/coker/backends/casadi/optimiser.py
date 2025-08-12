from coker.backends.casadi.casadi import substitute, to_casadi, lower
from typing import List
from coker import Tracer

import casadi as ca
import numpy as np


def build_optimisation_problem(
    cost, constraints, parameters: List[Tracer], outputs, initial_conditions
):

    # p = P(parameters)
    # x = P(inputs ~ parameter)
    tape = cost.tape
    workspace = {}
    parameter_indicies = {p.index for p in parameters}
    inputs = [i for i in tape.input_indicies if i not in parameter_indicies]

    parameter_dim = sum(p.dim.flat() for p in parameters)
    input_dim = sum(tape.dim[i].flat() for i in inputs)

    x = ca.MX.sym("x", input_dim)
    x0 = ca.DM(input_dim, 1)
    input_offset = 0
    parameter_offset = 0

    for i in inputs:
        n_i = tape.dim[i].flat()
        projection = ca.MX(n_i, input_dim)
        dm_projection = ca.DM(n_i, input_dim)
        for j in range(n_i):
            projection[j, j + input_offset] = 1
            dm_projection[j, j + input_offset] = 1

        workspace[i] = projection @ x
        x0_i = initial_conditions[i]
        x0 += dm_projection.T @ ca.DM(x0_i)

        input_offset += n_i

    p = ca.MX.sym("p", parameter_dim)

    for i in parameter_indicies:
        n_i = tape.dim[i].flat()
        projection = ca.MX(n_i, parameter_dim)
        for j in range(n_i):
            projection[j, parameter_offset + j] = 1

        workspace[i] = projection @ p
        parameter_offset += n_i

    (cost_fn,) = substitute([cost], workspace)
    output_map = ca.Function("y", *lower(tape, outputs, workspace))

    n_constraints = len(constraints)

    #    constraint_function = np.zeros((n_constraints,))
    #    lower_bound = ca.DM(n_constraints, 1)
    #    upper_bound = ca.DM(n_constraints, 1)
    cs = []
    lbs = []
    ubs = []

    for i, constraint in enumerate(constraints):
        c, lb, ub = constraint.as_halfplane_bound()
        (c_i,) = substitute([c], workspace)

        lbs.append(to_casadi(lb) * ca.DM.ones(*c_i.shape))
        ubs.append(to_casadi(ub) * ca.DM.ones(*c_i.shape))
        cs.append(c_i)

    upper_bound = ca.vertcat(*ubs)
    lower_bound = ca.vertcat(*lbs)
    g = ca.vertcat(*cs)

    spec = {"x": x, "p": p, "f": cost_fn, "g": g}

    solver_inner = ca.nlpsol("solver", "ipopt", spec)

    return CasadiSolver(
        solver_inner, p, None, (lower_bound, upper_bound), output_map, x0
    )


class CasadiSolver:
    def __init__(self, solver_inner, p, x_bounds, g_bounds, output_map, x0):
        self.solver_inner = solver_inner
        self.x_bounds = x_bounds
        self.g_bounds = g_bounds
        self.output_map = output_map
        self.x0 = x0
        self.p: ca.MX = p

    def __call__(self, *args):
        if not self.p.is_empty():
            raise NotImplementedError

        spec = {
            "x0": self.x0,
            "lbg": self.g_bounds[0],
            "ubg": self.g_bounds[1],
        }
        soln = self.solver_inner(**spec)

        result = self.output_map(soln["x"])

        return [r.full() for r in result]
