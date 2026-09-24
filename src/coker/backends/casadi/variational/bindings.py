"""CasADi control and parameter transcription bindings."""

from itertools import accumulate

from typing import List, Optional

import casadi as ca
import numpy as np

from coker.dynamics import (
    ConstantControlVariable,
    ControlVariable,
    PiecewiseConstantVariable,
    SpikeVariable,
)
from coker.parameters import (
    BoundedVariable,
    ParameterVariable,
    UnboundedVariable,
)


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
        if isinstance(p, (BoundedVariable, UnboundedVariable)):
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
            upper_bounds.append(p.upper_bound)
            guess.append(p.guess)
            p0.append(p.guess)
            lower_bounds.append(p.lower_bound)
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
        output_map,
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
        return ca.DM.zeros(sum(self.sizes), 1)

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


def _to_output_projector(proj: ca.DM) -> Optional[np.ndarray]:
    if proj.shape[0] == 0:
        return None
    return np.asarray(proj, dtype=float).reshape(proj.shape)
