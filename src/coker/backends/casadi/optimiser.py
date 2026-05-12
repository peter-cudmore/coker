from coker.backends.casadi.casadi import substitute, to_casadi, lower
from typing import List

import casadi as ca
import numpy as np

from coker.algebra.kernel import Tracer
from coker.optimisation import SolveFailure, solve_info_from_casadi_stats


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
        self.last_solve_info = None

    def __call__(self, *args):
        spec = {
            "x0": self.x0,
            "lbg": self.g_bounds[0],
            "ubg": self.g_bounds[1],
        }
        output_args = []
        if not self.p.is_empty():
            parameter_values = [
                np.asarray(arg, dtype=float).reshape((-1, 1)) for arg in args
            ]
            p_value = ca.vertcat(*[ca.DM(value) for value in parameter_values])
            if p_value.shape != self.p.shape:
                raise ValueError(
                    f"Expected parameter vector with shape "
                    f"{self.p.shape}, got {p_value.shape}"
                )
            spec["p"] = p_value
            output_args.append(p_value)
        elif args:
            raise ValueError(
                "This optimisation problem does not accept runtime parameters"
            )

        soln = self.solver_inner(**spec)
        self.last_solve_info = solve_info_from_casadi_stats(
            self.solver_inner.stats()
        )
        if not self.last_solve_info.success:
            raise SolveFailure(
                "CasADi optimisation solve failed with status "
                f"{self.last_solve_info.return_status}",
                self.last_solve_info,
            )

        result = self.output_map(soln["x"], *output_args)
        if self.output_map.n_out() == 1:
            result = [result]

        return [np.asarray(r.full()) for r in result]
