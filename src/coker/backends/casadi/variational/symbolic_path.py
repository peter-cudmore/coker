"""Symbolic path, controls, and solution assembly for CasADi."""

from itertools import accumulate
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

from coker.dynamics import (
    BoundedVariable,
    ConstantControlVariable,
    ControlVariable,
    InterpolatingPoly,
    InterpolatingPolyCollection,
    ParameterVariable,
    PiecewiseConstantVariable,
    SpikeVariable,
    UnboundedVariable,
    VariationalProblem,
    VariationalSolution,
)

if TYPE_CHECKING:
    from .factory import _TranscriptionFactory


class CasadiSolutionAssembler:
    def __init__(
        self,
        *,
        problem: VariationalProblem,
        factory: "_TranscriptionFactory",
        output_function: ca.Function,
        poly_collection: "SymbolicPolyCollection",
        projectors: Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
        ],
        proj_p: ca.DM,
        parameter_indices: Dict[str, int],
        decode_controls: Optional[Callable[[ca.DM], list]],
    ):
        self.problem = problem
        self.factory = factory
        self.output_function = output_function
        self.poly_collection = poly_collection
        self.projectors = projectors
        self.proj_p = proj_p
        self.parameter_indices = parameter_indices
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
        solver_parameter_vector = np.asarray(
            free_parameters, dtype=float
        ).reshape((-1,))
        public_parameters = (
            self.problem.parameter_layout.reconstruct(system_parameters)
            if self.problem.parameter_layout is not None
            else {
                name: float(free_parameters[index, 0])
                for name, index in self.parameter_indices.items()
            }
        )
        control_solutions = (
            self.decode_controls(control_coefficients)
            if self.decode_controls is not None
            else None
        )
        solution = VariationalSolution.from_solver(
            cost=loss,
            projectors=tuple(
                projector.copy() if projector is not None else None
                for projector in self.projectors
            ),
            parameters=public_parameters,
            parameter_vector=system_parameters,
            solver_parameter_vector=solver_parameter_vector,
            path=path,
            control_solutions=control_solutions,
            output=self.problem.system.y,
            t_final=float(horizon),
            solve_info=solve_info,
            path_constraint_exprs=self.problem.path_constraints,
            terminal_constraint_exprs=self.problem.terminal_constraints,
        )
        solution.segment_defects = tuple(
            self.factory.measure_interval_segment_defect(poly, solution)
            for poly in solution.path.polys
        )
        return solution


class SymbolicPoly(InterpolatingPoly):
    def __init__(
        self,
        name,
        dimension,
        interval,
        degree,
        factory: Optional["_TranscriptionFactory"] = None,
        values: Optional[ca.MX] = None,
        decision_values: Optional[ca.MX] = None,
    ):
        size = (degree + 1) * dimension
        values = ca.MX.sym(name, size) if values is None else values
        self._decision_values = (
            values if decision_values is None else decision_values
        )
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
        return self._decision_values

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
    def __init__(
        self,
        name,
        dimension,
        intervals,
        degrees,
        factory: Optional["_TranscriptionFactory"] = None,
        *,
        state_size: Optional[int] = None,
        algebraic_size: int = 0,
    ):
        assert len(intervals) == len(degrees)
        self._dimension = dimension
        self._state_size = dimension if state_size is None else state_size
        self._algebraic_size = algebraic_size
        if (
            self._state_size < 0
            or self._algebraic_size < 0
            or self._state_size + self._algebraic_size > dimension
        ):
            raise ValueError(
                "state_size and algebraic_size must fit path dimension"
            )
        polys = []
        for i, (interval, degree) in enumerate(zip(intervals, degrees)):
            if i == 0:
                values = ca.MX.sym(f"{name}_{i}", (degree + 1) * dimension)
                decision_values = values
            else:
                algebraic_start = (
                    ca.MX.sym(
                        f"{name}_{i}_algebraic_start",
                        self._algebraic_size,
                    )
                    if self._algebraic_size
                    else ca.MX.zeros(0, 1)
                )
                boundary_values = self._boundary_values(
                    polys[-1].end_point()[1],
                    algebraic_start,
                )
                tail_values = ca.MX.sym(f"{name}_{i}_tail", degree * dimension)
                values = ca.vertcat(boundary_values, tail_values)
                decision_values = ca.vertcat(algebraic_start, tail_values)
            polys.append(
                SymbolicPoly(
                    f"{name}_{i}",
                    dimension,
                    interval,
                    degree,
                    factory=factory,
                    values=values,
                    decision_values=decision_values,
                )
            )
        super().__init__(polys)
        self._symbols = ca.vertcat(*[poly.symbols() for poly in polys])
        self._symbol_size = int(self._symbols.shape[0])

    def symbols(self):
        return self._symbols

    def size(self):
        return self._symbol_size

    def _boundary_values(
        self, previous_end: ca.MX, algebraic_start: ca.MX
    ) -> ca.MX:
        algebraic_end = self._state_size + self._algebraic_size
        pieces = []
        if self._state_size:
            pieces.append(previous_end[: self._state_size])
        if self._algebraic_size:
            pieces.append(algebraic_start)
        if algebraic_end < self._dimension:
            pieces.append(previous_end[algebraic_end:])
        return ca.vertcat(*pieces) if pieces else ca.MX.zeros(0, 1)

    def constant_guess(self, value: ca.DM) -> ca.DM:
        """Repeat one path value in the compact decision storage."""
        pieces = [
            ca.repmat(value, self.polys[0].degree + 1),
        ]
        for poly in self.polys[1:]:
            if self._algebraic_size:
                pieces.append(
                    value[
                        self._state_size : self._state_size
                        + self._algebraic_size
                    ]
                )
            pieces.append(ca.repmat(value, poly.degree))
        return ca.vertcat(*pieces)

    def collect_guess(self, path: InterpolatingPolyCollection) -> ca.DM:
        """Sample a fixed path into the compact decision storage."""
        pieces = []
        for index, poly in enumerate(self.polys):
            values = [
                ca.DM(path(float(time))).reshape((poly.dimension, 1))
                for time in poly.knot_times()
            ]
            if index == 0:
                pieces.extend(values)
                continue
            if self._algebraic_size:
                pieces.append(
                    values[0][
                        self._state_size : self._state_size
                        + self._algebraic_size
                    ]
                )
            pieces.extend(values[1:])
        return ca.vertcat(*pieces)

    def to_fixed(self, array):
        np_array = np.array(array)
        assert np_array.shape == (self._symbol_size, 1)

        polys = []
        fixed_values = []
        offset = 0
        for index, poly in enumerate(self.polys):
            decision_size = int(poly.symbols().shape[0])
            decision_values = np_array[offset : offset + decision_size]
            offset += decision_size
            if index == 0:
                values = decision_values
            else:
                previous_end = fixed_values[-1][-poly.dimension :]
                algebraic_end = self._state_size + self._algebraic_size
                pieces = []
                if self._state_size:
                    pieces.append(previous_end[: self._state_size])
                if self._algebraic_size:
                    pieces.append(decision_values[: self._algebraic_size])
                if algebraic_end < self._dimension:
                    pieces.append(previous_end[algebraic_end:])
                start = np.vstack(pieces)
                values = np.vstack(
                    (start, decision_values[self._algebraic_size :])
                )
            assert values.shape == (poly.size(), 1)
            fixed_values.append(values)
            polys.append(
                InterpolatingPoly(
                    poly.dimension,
                    poly.interval,
                    poly.degree,
                    values,
                    reference_operators=poly._reference_operators,
                )
            )
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
