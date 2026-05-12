from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import scipy.optimize as optimize

from coker.algebra import Dimension, OP
from coker.algebra.kernel import Tape, Tracer
from coker.backends.evaluator import evaluate_inner
from coker.optimisation import SolveFailure, solve_info_from_scipy_result


@dataclass(frozen=True)
class InputBinding:
    index: int
    dim: Dimension
    start: int
    stop: int


@dataclass(frozen=True)
class ConstraintFactory:
    lower_bound: float
    upper_bound: float
    is_affine_in_decisions: bool
    decision_dimension: int
    output_dimension: int
    evaluate: Callable[[np.ndarray, tuple], np.ndarray]
    jacobian: Callable[[np.ndarray, tuple], np.ndarray]

    def build(
        self, runtime_args: tuple
    ) -> optimize.LinearConstraint | optimize.NonlinearConstraint:
        if self.is_affine_in_decisions:
            zero_decision = np.zeros(self.decision_dimension, dtype=float)
            affine_offset = self.evaluate(zero_decision, runtime_args)
            linear_term = self.jacobian(zero_decision, runtime_args)
            lower_bound = (
                _expand_bound(self.lower_bound, self.output_dimension)
                - affine_offset
            )
            upper_bound = (
                _expand_bound(self.upper_bound, self.output_dimension)
                - affine_offset
            )
            return optimize.LinearConstraint(
                linear_term, lower_bound, upper_bound
            )

        lower_bound = _expand_bound(self.lower_bound, self.output_dimension)
        upper_bound = _expand_bound(self.upper_bound, self.output_dimension)

        def nonlinear_constraint_value(
            decision_vector: np.ndarray,
        ) -> np.ndarray:
            return self.evaluate(decision_vector, runtime_args)

        def nonlinear_constraint_jacobian(
            decision_vector: np.ndarray,
        ) -> np.ndarray:
            return self.jacobian(decision_vector, runtime_args)

        return optimize.NonlinearConstraint(
            nonlinear_constraint_value,
            lower_bound,
            upper_bound,
            jac=nonlinear_constraint_jacobian,
            hess=optimize.BFGS(),
        )


class TrustConstrProblem:
    def __init__(
        self,
        *,
        tape: Tape,
        backend,
        decision_bindings: list[InputBinding],
        parameter_bindings: list[InputBinding],
        cost: Tracer,
        constraints: list[ConstraintFactory],
        outputs: list[Tracer],
        initial_guess: np.ndarray,
    ):
        self.tape = tape
        self.backend = backend
        self.decision_bindings = decision_bindings
        self.parameter_bindings = parameter_bindings
        self.cost = cost
        self.constraints = constraints
        self.outputs = outputs
        self.initial_guess = initial_guess
        self.last_solve_info = None

    def __call__(self, *runtime_args):
        runtime_args_tuple = self._normalise_runtime_args(runtime_args)
        scipy_constraints = [
            constraint.build(runtime_args_tuple)
            for constraint in self.constraints
        ]
        solution = optimize.minimize(
            self._evaluate_cost,
            self.initial_guess.copy(),
            args=runtime_args_tuple,
            method="trust-constr",
            jac=self._evaluate_cost_jacobian,
            hess=self._evaluate_cost_hessian,
            constraints=scipy_constraints,
            options={
                "gtol": 1e-10,
                "xtol": 1e-10,
                "barrier_tol": 1e-10,
                "maxiter": 1000,
            },
        )
        self.last_solve_info = solve_info_from_scipy_result(solution)
        if not self.last_solve_info.success:
            raise SolveFailure(
                "NumPy optimisation solve failed with status "
                f"{self.last_solve_info.return_status}",
                self.last_solve_info,
            )
        return self._evaluate_outputs(solution.x, runtime_args_tuple)

    def _normalise_runtime_args(
        self, runtime_args: Sequence[object]
    ) -> tuple[object, ...]:
        if len(runtime_args) != len(self.parameter_bindings):
            raise ValueError(
                "Expected "
                f"{len(self.parameter_bindings)} runtime arguments, got "
                f"{len(runtime_args)}"
            )
        return tuple(
            _normalise_value(value, binding.dim)
            for value, binding in zip(runtime_args, self.parameter_bindings)
        )

    def _materialise_inputs(
        self, decision_vector: np.ndarray, runtime_args: tuple[object, ...]
    ) -> list[object]:
        decision_values = {
            binding.index: _reshape_flat_slice(
                decision_vector[binding.start : binding.stop], binding.dim
            )
            for binding in self.decision_bindings
        }
        parameter_values = {
            binding.index: value
            for binding, value in zip(self.parameter_bindings, runtime_args)
        }
        tape_inputs = []
        for index in self.tape.input_indicies:
            if index in decision_values:
                tape_inputs.append(decision_values[index])
                continue
            if index in parameter_values:
                tape_inputs.append(parameter_values[index])
                continue
            raise ValueError(
                f"Missing optimisation input for tape index {index}"
            )
        return tape_inputs

    def _evaluate_tracers(
        self,
        tracers: Iterable[Tracer],
        decision_vector: np.ndarray,
        runtime_args: tuple[object, ...],
    ):
        tape_inputs = self._materialise_inputs(decision_vector, runtime_args)
        return evaluate_inner(
            self.tape, tape_inputs, list(tracers), self.backend, {}
        )

    def _evaluate_cost(
        self, decision_vector: np.ndarray, *runtime_args
    ) -> float:
        (cost_value,) = self._evaluate_tracers(
            [self.cost], decision_vector, tuple(runtime_args)
        )
        return _coerce_scalar(cost_value)

    def _evaluate_cost_jacobian(
        self, decision_vector: np.ndarray, *runtime_args
    ) -> np.ndarray:
        runtime_args_tuple = tuple(runtime_args)
        return _finite_difference_gradient(
            self._evaluate_cost, decision_vector, runtime_args_tuple
        )

    def _evaluate_cost_hessian(
        self, decision_vector: np.ndarray, *runtime_args
    ) -> np.ndarray:
        runtime_args_tuple = tuple(runtime_args)
        return _finite_difference_hessian(
            self._evaluate_cost_jacobian, decision_vector, runtime_args_tuple
        )

    def _evaluate_outputs(
        self, decision_vector: np.ndarray, runtime_args: tuple[object, ...]
    ) -> list[object]:
        return list(
            self._evaluate_tracers(self.outputs, decision_vector, runtime_args)
        )


def build_optimisation_problem(
    backend,
    cost: Tracer,
    constraints: list[Tracer],
    parameters: list[Tracer],
    outputs: list[Tracer],
    initial_conditions: dict[int, object],
):
    tape = cost.tape
    assert all(constraint.tape == tape for constraint in constraints)
    assert all(parameter.tape == tape for parameter in parameters)
    assert all(output.tape == tape for output in outputs)

    parameter_indices = {parameter.index for parameter in parameters}
    decision_indices = [
        index
        for index in tape.input_indicies
        if index not in parameter_indices
    ]
    parameter_bindings = _make_bindings(
        [parameter.index for parameter in parameters], tape
    )
    decision_bindings = _make_bindings(decision_indices, tape)
    initial_guess = _build_initial_guess(decision_bindings, initial_conditions)

    constraint_factories = []
    for constraint in constraints:
        residual, lower_bound, upper_bound = constraint.as_halfplane_bound()
        constraint_factories.append(
            _build_constraint_factory(
                backend,
                tape,
                decision_indices,
                parameter_bindings,
                decision_bindings,
                residual,
                lower_bound,
                upper_bound,
            )
        )

    return TrustConstrProblem(
        tape=tape,
        backend=backend,
        decision_bindings=decision_bindings,
        parameter_bindings=parameter_bindings,
        cost=cost,
        constraints=constraint_factories,
        outputs=outputs,
        initial_guess=initial_guess,
    )


def _make_bindings(indices: Iterable[int], tape: Tape) -> list[InputBinding]:
    bindings = []
    offset = 0
    for index in indices:
        dim = tape.dim[index]
        flat_size = dim.flat()
        bindings.append(
            InputBinding(
                index=index, dim=dim, start=offset, stop=offset + flat_size
            )
        )
        offset += flat_size
    return bindings


def _build_initial_guess(
    decision_bindings: list[InputBinding],
    initial_conditions: dict[int, object],
) -> np.ndarray:
    if not decision_bindings:
        return np.zeros(0, dtype=float)
    flat_slices = []
    for binding in decision_bindings:
        if binding.index not in initial_conditions:
            raise ValueError(
                "Missing initial condition for decision "
                f"variable {binding.index}"
            )
        flat_slices.append(
            _flatten_value(initial_conditions[binding.index], binding.dim)
        )
    return np.concatenate(flat_slices)


def _build_constraint_factory(
    backend,
    tape: Tape,
    decision_indices: list[int],
    parameter_bindings: list[InputBinding],
    decision_bindings: list[InputBinding],
    residual: Tracer,
    lower_bound: float,
    upper_bound: float,
) -> ConstraintFactory:
    decision_dimension = decision_bindings[-1].stop if decision_bindings else 0
    problem = TrustConstrProblem(
        tape=tape,
        backend=backend,
        decision_bindings=decision_bindings,
        parameter_bindings=parameter_bindings,
        cost=residual,
        constraints=[],
        outputs=[residual],
        initial_guess=np.zeros(decision_dimension, dtype=float),
    )

    def evaluate(
        decision_vector: np.ndarray, runtime_args: tuple
    ) -> np.ndarray:
        (residual_value,) = problem._evaluate_tracers(
            [residual], decision_vector, runtime_args
        )
        return _coerce_vector(residual_value)

    def jacobian(
        decision_vector: np.ndarray, runtime_args: tuple
    ) -> np.ndarray:
        return _finite_difference_jacobian(
            evaluate, decision_vector, runtime_args
        )

    return ConstraintFactory(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        is_affine_in_decisions=_is_affine_in_decisions(
            residual, tape, set(decision_indices)
        ),
        decision_dimension=decision_dimension,
        output_dimension=residual.dim.flat(),
        evaluate=evaluate,
        jacobian=jacobian,
    )


def _is_affine_in_decisions(
    tracer: Tracer | object, tape: Tape, decision_indices: set[int], memo=None
) -> bool:
    if memo is None:
        memo = {}
    return _decision_degree(tracer, tape, decision_indices, memo) <= 1


def _decision_degree(
    tracer: Tracer | object,
    tape: Tape,
    decision_indices: set[int],
    memo: dict[int, int],
) -> int:
    if not isinstance(tracer, Tracer) or tracer.tape is not tape:
        return 0
    if tracer.index in memo:
        return memo[tracer.index]

    node = tape.nodes[tracer.index]
    if isinstance(node, Tracer):
        degree = 1 if tracer.index in decision_indices else 0
        memo[tracer.index] = degree
        return degree

    op, *arguments = node
    if op == OP.VALUE:
        degree = _decision_degree(arguments[0], tape, decision_indices, memo)
    elif op.is_linear():
        degree = max(
            (
                _decision_degree(argument, tape, decision_indices, memo)
                for argument in arguments
            ),
            default=0,
        )
    elif op.is_bilinear():
        argument_degrees = [
            _decision_degree(argument, tape, decision_indices, memo)
            for argument in arguments
        ]
        if any(degree > 1 for degree in argument_degrees):
            degree = 2
        else:
            dependent_argument_count = sum(
                1 for degree in argument_degrees if degree > 0
            )
            degree = (
                max(argument_degrees, default=0)
                if dependent_argument_count <= 1
                else 2
            )
    else:
        argument_degrees = [
            _decision_degree(argument, tape, decision_indices, memo)
            for argument in arguments
        ]
        degree = 0 if all(degree == 0 for degree in argument_degrees) else 2

    memo[tracer.index] = degree
    return degree


def _finite_difference_gradient(
    function: Callable[..., float],
    decision_vector: np.ndarray,
    runtime_args: tuple,
) -> np.ndarray:
    gradient = np.zeros_like(decision_vector, dtype=float)
    if decision_vector.size == 0:
        return gradient

    step_sizes = _finite_difference_steps(decision_vector)
    for axis, step_size in enumerate(step_sizes):
        positive = decision_vector.copy()
        negative = decision_vector.copy()
        positive[axis] += step_size
        negative[axis] -= step_size
        gradient[axis] = (
            function(positive, *runtime_args)
            - function(negative, *runtime_args)
        ) / (2.0 * step_size)
    return gradient


def _finite_difference_jacobian(
    function: Callable[[np.ndarray, tuple], np.ndarray],
    decision_vector: np.ndarray,
    runtime_args: tuple,
) -> np.ndarray:
    base_value = function(decision_vector, runtime_args)
    jacobian = np.zeros((base_value.size, decision_vector.size), dtype=float)
    if decision_vector.size == 0:
        return jacobian

    step_sizes = _finite_difference_steps(decision_vector)
    for axis, step_size in enumerate(step_sizes):
        positive = decision_vector.copy()
        negative = decision_vector.copy()
        positive[axis] += step_size
        negative[axis] -= step_size
        positive_value = function(positive, runtime_args)
        negative_value = function(negative, runtime_args)
        jacobian[:, axis] = (positive_value - negative_value) / (
            2.0 * step_size
        )
    return jacobian


def _finite_difference_hessian(
    gradient_function: Callable[..., np.ndarray],
    decision_vector: np.ndarray,
    runtime_args: tuple,
) -> np.ndarray:
    hessian = np.zeros(
        (decision_vector.size, decision_vector.size), dtype=float
    )
    if decision_vector.size == 0:
        return hessian

    step_sizes = _finite_difference_steps(decision_vector)
    for axis, step_size in enumerate(step_sizes):
        positive = decision_vector.copy()
        negative = decision_vector.copy()
        positive[axis] += step_size
        negative[axis] -= step_size
        positive_gradient = gradient_function(positive, *runtime_args)
        negative_gradient = gradient_function(negative, *runtime_args)
        hessian[:, axis] = (positive_gradient - negative_gradient) / (
            2.0 * step_size
        )
    return 0.5 * (hessian + hessian.T)


def _finite_difference_steps(decision_vector: np.ndarray) -> np.ndarray:
    machine_step = np.sqrt(np.finfo(float).eps)
    return machine_step * np.maximum(1.0, np.abs(decision_vector))


def _coerce_scalar(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise TypeError(f"Expected scalar value, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _coerce_vector(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _expand_bound(bound: float, size: int) -> np.ndarray:
    return np.full(size, float(bound), dtype=float)


def _normalise_value(value: object, dim: Dimension) -> object:
    array = np.asarray(value, dtype=float)
    if dim.is_scalar():
        if array.size != 1:
            raise ValueError(
                f"Expected scalar value for {dim}, got shape {array.shape}"
            )
        return float(array.reshape(-1)[0])
    if array.shape != dim.dim:
        raise ValueError(
            f"Expected value with shape {dim.dim}, got {array.shape}"
        )
    return array


def _flatten_value(value: object, dim: Dimension) -> np.ndarray:
    normalised_value = _normalise_value(value, dim)
    if dim.is_scalar():
        return np.array([normalised_value], dtype=float)
    return np.asarray(normalised_value, dtype=float).reshape(-1)


def _reshape_flat_slice(value: np.ndarray, dim: Dimension) -> object:
    if dim.is_scalar():
        if value.size != 1:
            raise ValueError(f"Expected scalar slice, got shape {value.shape}")
        return float(value[0])
    return np.asarray(value, dtype=float).reshape(dim.dim)
