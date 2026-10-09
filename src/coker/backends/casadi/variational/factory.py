"""Mesh-independent CasADi variational transcription bindings."""

from collections import OrderedDict
from functools import lru_cache

import casadi as ca
import numpy as np

from coker.algebra.ops import Noop
from coker.backends.backend import get_backend_by_name
from coker.backends.casadi.lower import lower as lower_casadi
from coker.backends.casadi.variational.options import CasadiVariationalOptions
from coker.backends.casadi.residual_support import lower_residual
from coker.dynamics import VariationalProblem, VariationalSolution
from coker.dynamics.residual import (
    ResidualDynamicalSystem,
    to_residual_dynamical_system,
)
from coker.dynamics.transcription.collocation import (
    _build_reference_operators,
    lgr_points,
)
from coker.dynamics.variational.solution import SegmentDefectDiagnostic

from .bindings import (
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
        self.residual = (
            problem.system
            if isinstance(problem.system, ResidualDynamicalSystem)
            else to_residual_dynamical_system(problem.system)
        )
        x_dim, z_dim, q_dim = self.residual.get_state_dimensions()
        self.x_size = x_dim.flat()
        self.z_size = z_dim.flat() if z_dim else 0
        self.system_quadrature_size = q_dim.flat() if q_dim else 0
        self.q_size = self.system_quadrature_size + len(problem.quadratures)
        self.path_size = self.x_size + self.z_size + self.q_size
        self.tolerance = problem.transcription_options.absolute_tolerance
        self.segment_defect_tolerance = (
            problem.transcription_options.segment_defect_tolerance
            if problem.transcription_options.segment_defect_tolerance is not None
            else self.tolerance
        )
        self.derivative_defect_tolerance = (
            problem.transcription_options.derivative_defect_tolerance
            if problem.transcription_options.derivative_defect_tolerance is not None
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
            ControlFactory(control_variables, 1.0) if control_variables else None
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
            ca.MX.sym(problem.horizon_decision.name) if self.free_horizon else None
        )
        self.duration = (
            self.horizon_symbol if self.free_horizon else float(problem.t_final)
        )
        self.parameter_names = list(parameter_indices)
        self.parameter_indices = parameter_indices
        self.reference_operator_cache = lru_cache(
            maxsize=_REFERENCE_OPERATOR_CACHE_SIZE
        )(_build_reference_operators)

        self._defect_residual_maps: OrderedDict[int, ca.Function] = OrderedDict()
        self._defect_nodes: OrderedDict[int, tuple[float, ...]] = OrderedDict()
        self._defect_residual = self._build_defect_residual()

    def _residual_state(self, state, quadrature):
        if self.system_quadrature_size == 0:
            return state
        return ca.vertcat(state, quadrature[: self.system_quadrature_size])

    def _residual_rate(self, state_rate, quadrature_rate):
        if self.system_quadrature_size == 0:
            return state_rate
        return ca.vertcat(state_rate, quadrature_rate[: self.system_quadrature_size])

    def evaluate_symbolic_residual(
        self,
        *,
        time,
        state,
        state_rate,
        algebraic,
        quadrature,
        quadrature_rate,
        control,
        parameters,
    ):
        return lower_residual(
            self.casadi,
            self.residual,
            time,
            self._residual_state(state, quadrature),
            self._residual_rate(state_rate, quadrature_rate),
            algebraic,
            (control, parameters),
        )

    def evaluate_registered_quadratures(
        self,
        *,
        time,
        state,
        algebraic,
        control,
        parameters,
        quadrature,
    ):
        arguments = (time, state, algebraic, control, parameters, quadrature)
        values = []
        for spec in self.problem.quadratures:
            workspace = dict(zip(spec.integrand.tape.input_indicies, arguments))
            _, outputs = lower_casadi(spec.integrand.tape, [spec.integrand], workspace)
            values.append(outputs[0])
        return values

    def _build_defect_residual(self) -> ca.Function:
        time = ca.MX.sym("defect_time")
        state = ca.MX.sym("defect_state", self.x_size)
        state_rate = ca.MX.sym("defect_state_rate", self.x_size)
        algebraic = ca.MX.sym("defect_algebraic", self.z_size)
        control = ca.MX.sym("defect_control", self.u_symbols.shape[0])
        parameters = ca.MX.sym("defect_parameters", self.proj_p.shape[0])
        quadrature = ca.MX.sym("defect_quadrature", self.q_size)
        quadrature_rate = ca.MX.sym("defect_quadrature_rate", self.q_size)
        residual = self.evaluate_symbolic_residual(
            time=time,
            state=state,
            state_rate=state_rate,
            algebraic=algebraic,
            quadrature=quadrature,
            quadrature_rate=quadrature_rate,
            control=control,
            parameters=parameters,
        )
        return ca.Function(
            "defect_residual",
            [
                time,
                state,
                state_rate,
                algebraic,
                control,
                parameters,
                quadrature,
                quadrature_rate,
            ],
            [residual],
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

    def evaluate_numeric_defect_residuals(
        self,
        times: np.ndarray,
        values: np.ndarray,
        derivatives: np.ndarray,
        solution: VariationalSolution,
    ) -> np.ndarray:
        """Evaluate the coupled model residual over a batch of path nodes."""
        times = np.asarray(times, dtype=float).reshape((1, -1))
        count = times.shape[1]
        values = np.asarray(values, dtype=float).reshape((self.path_size, count))
        derivatives = np.asarray(derivatives, dtype=float).reshape(
            (self.path_size, count)
        )
        if self.control_factory is None:
            controls = np.zeros((0, count))
        else:
            control_samples = [
                np.asarray(solution.control_law(float(time)), dtype=float).reshape(
                    (-1,)
                )
                for time in times.flat
            ]
            controls = np.column_stack(control_samples)
        parameter_vector = solution._parameter_vector.reshape((-1, 1))
        parameters = np.repeat(parameter_vector, count, axis=1)
        try:
            evaluator = self._defect_residual_maps.pop(count)
        except KeyError:
            evaluator = self._defect_residual.map(count)
            if len(self._defect_residual_maps) == _DEFECT_EVALUATOR_CACHE_SIZE:
                self._defect_residual_maps.popitem(last=False)
        self._defect_residual_maps[count] = evaluator
        physical_times = times * solution.t_final
        state_values = values[: self.x_size]
        state_derivatives = derivatives[: self.x_size]
        algebraic_values = values[self.x_size : self.x_size + self.z_size]
        quadrature_values = values[self.x_size + self.z_size :]
        quadrature_derivatives = derivatives[self.x_size + self.z_size :]
        residuals = evaluator(
            ca.DM(physical_times),
            ca.DM(state_values),
            ca.DM(state_derivatives),
            ca.DM(algebraic_values),
            ca.DM(controls),
            ca.DM(parameters),
            ca.DM(quadrature_values),
            ca.DM(quadrature_derivatives),
        )
        return np.asarray(residuals, dtype=float)

    def measure_interval_defect(self, poly, solution: VariationalSolution) -> float:
        """Return the maximum absolute coupled residual over one interval."""
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
            np.arange(1, poly.bases.shape[0], dtype=float)[:, np.newaxis] * powers[:-1]
        )
        interpolation = path_values @ np.asarray(poly.bases).T
        values = interpolation @ powers
        physical_width = poly.width * solution.t_final
        derivatives = interpolation @ derivative_powers / physical_width
        residuals = self.evaluate_numeric_defect_residuals(
            times, values, derivatives, solution
        )
        return float(np.max(np.abs(residuals))) if residuals.size else 0.0

    def measure_interval_segment_defect(
        self, poly, solution: VariationalSolution
    ) -> SegmentDefectDiagnostic:
        """Measure one collocation segment's coupled residual."""
        weights = np.asarray(poly.weights, dtype=float).reshape((-1,))[:-1]
        full_residual = None
        for weight, (time, value, derivative) in zip(weights, poly.knot_points()):
            node_time = np.asarray([time], dtype=float)
            node_values = np.asarray(value, dtype=float).reshape((self.path_size, 1))
            node_derivatives = (
                np.asarray(derivative, dtype=float).reshape((self.path_size, 1))
                / solution.t_final
            )
            residual = self.evaluate_numeric_defect_residuals(
                node_time,
                node_values,
                node_derivatives,
                solution,
            ).reshape((-1,))
            residual *= weight
            if full_residual is None:
                full_residual = residual
            else:
                full_residual += residual

        assert full_residual is not None
        full_residual *= solution.t_final
        normalized_interval = tuple(float(t) for t in poly.interval)
        physical_interval = tuple(
            solution.t_final * time for time in normalized_interval
        )
        return SegmentDefectDiagnostic(
            normalized_interval=normalized_interval,
            physical_interval=physical_interval,
            degree=poly.degree,
            tolerance=self.segment_defect_tolerance,
            full_residual=full_residual,
        )
