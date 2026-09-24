"""Mesh-independent CasADi variational transcription bindings."""

from collections import OrderedDict
from functools import lru_cache
from typing import Tuple

import casadi as ca
import numpy as np

from coker.algebra.ops import Noop
from coker.backends.backend import get_backend_by_name
from coker.backends.casadi.lower import lower as lower_casadi
from coker.backends.casadi.variational.options import CasadiVariationalOptions
from coker.dynamics import VariationalProblem, VariationalSolution
from coker.dynamics.transcription.collocation import (
    _build_reference_operators,
    lgr_points,
)
from coker.dynamics.variational.solution import SegmentDefectDiagnostic

from .symbolic_path import (
    ControlFactory,
    _to_output_projector,
    construct_parameters,
)

_REFERENCE_OPERATOR_CACHE_SIZE = 32
_DEFECT_EVALUATOR_CACHE_SIZE = 16


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
        self.segment_defect_tolerance = (
            problem.transcription_options.segment_defect_tolerance
            if problem.transcription_options.segment_defect_tolerance
            is not None
            else self.tolerance
        )
        self.derivative_defect_tolerance = (
            problem.transcription_options.derivative_defect_tolerance
            if problem.transcription_options.derivative_defect_tolerance
            is not None
            else self.tolerance
        )
        self.free_horizon = problem.horizon_decision is not None

        self.proj_x = ca.hcat(
            [
                ca.DM.eye(self.x_size),
                ca.DM.zeros(self.x_size, self.z_size + self.q_size),
            ]
        )
        self.proj_z = ca.hcat(
            [
                ca.DM.zeros(self.z_size, self.x_size),
                ca.DM.eye(self.z_size),
                ca.DM.zeros(self.z_size, self.q_size),
            ]
        )
        self.proj_q = ca.hcat(
            [
                ca.DM.zeros(self.q_size, self.x_size + self.z_size),
                ca.DM.eye(self.q_size),
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
            self.control_eval = Noop()
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
            parameter_indices,
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
        self.parameter_names = list(parameter_indices)
        self.parameter_indices = parameter_indices
        self.reference_operator_cache = lru_cache(
            maxsize=_REFERENCE_OPERATOR_CACHE_SIZE
        )(_build_reference_operators)

        self._defect_dynamics_maps: OrderedDict[int, ca.Function] = (
            OrderedDict()
        )
        self._defect_nodes: OrderedDict[int, tuple[float, ...]] = OrderedDict()
        self._defect_dynamics = self._build_defect_dynamics()

    def evaluate_dynamics(self, *args):
        return self.casadi.evaluate(self.problem.system.dxdt, args)

    def evaluate_quadrature(self, *args):
        if self.problem.system.dqdt is not None:
            return self.casadi.evaluate(self.problem.system.dqdt, args)
        return Noop()

    def evaluate_algebraic(self, *args):
        if self.problem.system.g:
            return self.casadi.evaluate(self.problem.system.g, args)
        return Noop()

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
        quadrature = ca.MX.sym("defect_quadrature", self.q_size)

        def control_law(_time):
            return control

        (dynamics,) = self.evaluate_dynamics(
            time,
            state,
            algebraic,
            control_law,
            parameters,
        )
        quadrature_rates = []
        if self.has_state_quadrature:
            (base_rate,) = self.evaluate_quadrature(
                time,
                state,
                algebraic,
                control_law,
                parameters,
            )
            quadrature_rates.append(base_rate)
        quadrature_rates.extend(
            self.evaluate_registered_quadratures(
                (time, state, algebraic, control, parameters, quadrature)
            )
        )
        return ca.Function(
            "defect_dynamics",
            [time, state, algebraic, control, parameters, quadrature],
            [
                dynamics,
                (
                    ca.vertcat(*quadrature_rates)
                    if quadrature_rates
                    else ca.MX.zeros(0, 1)
                ),
            ],
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

    def evaluate_defect_rates(
        self,
        times: np.ndarray,
        values: np.ndarray,
        solution: VariationalSolution,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate diagnostic state and quadrature rates in a CasADi batch."""
        times = np.asarray(times, dtype=float).reshape((1, -1))
        count = times.shape[1]
        values = np.asarray(values, dtype=float).reshape(
            (self.path_size, count)
        )
        controls = (
            np.column_stack(
                [
                    np.asarray(
                        solution.control_law(float(time)), dtype=float
                    ).reshape((-1,))
                    for time in times.flat
                ]
            )
            if self.control_factory is not None
            else np.zeros((0, count))
        )
        parameters = np.repeat(
            solution._parameter_vector.reshape((-1, 1)),
            count,
            axis=1,
        )
        try:
            evaluator = self._defect_dynamics_maps.pop(count)
        except KeyError:
            evaluator = self._defect_dynamics.map(count)
            if len(self._defect_dynamics_maps) == _DEFECT_EVALUATOR_CACHE_SIZE:
                self._defect_dynamics_maps.popitem(last=False)
        self._defect_dynamics_maps[count] = evaluator
        state_rates, quadrature_rates = evaluator(
            ca.DM(times * solution.t_final),
            ca.DM(values[: self.x_size]),
            ca.DM(values[self.x_size : self.x_size + self.z_size]),
            ca.DM(controls),
            ca.DM(parameters),
            ca.DM(values[self.x_size + self.z_size :]),
        )
        return (
            np.asarray(state_rates, dtype=float),
            np.asarray(quadrature_rates, dtype=float),
        )

    def measure_interval_defect(
        self, poly, solution: VariationalSolution
    ) -> float:
        """Return the maximum scaled state defect for one solved polynomial."""
        reference_nodes = self.get_defect_nodes(poly.degree)
        times = poly.interval[0] + (reference_nodes + 1.0) * poly.width
        path_values = np.asarray(poly.values, dtype=float).reshape(
            poly.dimension, -1, order="F"
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
        values = interpolation @ powers
        derivatives = (
            interpolation @ derivative_powers / (poly.width * solution.t_final)
        )
        model_dynamics, _ = self.evaluate_defect_rates(times, values, solution)
        state_derivatives = derivatives[: self.x_size]
        if state_derivatives.size == 0:
            return 0.0
        scale = np.maximum(1.0, np.abs(model_dynamics))
        return float(
            np.max(np.abs(state_derivatives - model_dynamics) / scale)
        )

    def measure_interval_segment_defect(
        self, poly, solution: VariationalSolution
    ) -> SegmentDefectDiagnostic:
        """Measure one collocation segment residual in physical units."""
        state_rates = []
        quadrature_rates = []
        for time, value, _derivative in poly.knot_points():
            dynamics, rates = self.evaluate_defect_rates(
                np.asarray([time], dtype=float),
                np.asarray(value, dtype=float).reshape((self.path_size, 1)),
                solution,
            )
            state_rates.append(dynamics.reshape((-1,)))
            if self.q_size > 0:
                quadrature_rates.append(rates.reshape((-1,)))

        weights = np.asarray(poly.weights, dtype=float).reshape((-1,))[:-1]
        _, start_value = poly.start_point()
        _, end_value = poly.end_point()
        state_residual = (
            np.asarray(end_value[: self.x_size], dtype=float).reshape((-1,))
            - np.asarray(start_value[: self.x_size], dtype=float).reshape(
                (-1,)
            )
            - solution.t_final * np.column_stack(state_rates) @ weights
        )
        if self.q_size == 0:
            quadrature_residual = np.zeros((0,), dtype=float)
        else:
            quadrature_residual = (
                np.asarray(
                    end_value[self.x_size + self.z_size :], dtype=float
                ).reshape((-1,))
                - np.asarray(
                    start_value[self.x_size + self.z_size :], dtype=float
                ).reshape((-1,))
                - solution.t_final
                * np.column_stack(quadrature_rates)
                @ weights
            )
        normalized_interval = tuple(float(t) for t in poly.interval)
        return SegmentDefectDiagnostic(
            normalized_interval=normalized_interval,
            physical_interval=tuple(
                solution.t_final * time for time in normalized_interval
            ),
            degree=poly.degree,
            tolerance=self.segment_defect_tolerance,
            state_residual=state_residual,
            quadrature_residual=quadrature_residual,
        )
