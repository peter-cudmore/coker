from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import scipy.optimize as optimize

from coker.algebra import Dimension
from coker.algebra.kernel import Tape, Tracer
from coker.backends.evaluator import evaluate_inner
from coker.backends.optimisation import (
    InputBinding,
    build_initial_guess,
    build_problem_bindings,
    coerce_scalar,
    coerce_vector,
    decision_degree,
    make_bindings as shared_make_bindings,
    flatten_value,
    is_affine_in_decisions,
    materialise_tape_inputs,
    normalise_runtime_args,
    normalise_value,
    reshape_flat_slice,
)
from coker.toolkits.codesign.optimisation import (
    SolveFailure,
    solve_info_from_scipy_result,
)


@dataclass(frozen=True)
class ConstraintFactory:
    lower_bound: object
    upper_bound: object
    is_affine_in_decisions: bool
    decision_dimension: int
    output_dimension: int
    evaluate: Callable[[np.ndarray, tuple], np.ndarray]
    jacobian: Callable[[np.ndarray, tuple], np.ndarray]
    evaluate_lower_bound: Callable[[tuple], np.ndarray]
    evaluate_upper_bound: Callable[[tuple], np.ndarray]

    def build(
        self, runtime_args: tuple
    ) -> optimize.LinearConstraint | optimize.NonlinearConstraint:
        lower_bound = self.evaluate_lower_bound(runtime_args)
        upper_bound = self.evaluate_upper_bound(runtime_args)
        if self.is_affine_in_decisions:
            zero_decision = np.zeros(self.decision_dimension, dtype=float)
            affine_offset = self.evaluate(zero_decision, runtime_args)
            return optimize.LinearConstraint(
                self.jacobian(zero_decision, runtime_args),
                lower_bound - affine_offset,
                upper_bound - affine_offset,
            )

        return optimize.NonlinearConstraint(
            lambda decision_vector: self.evaluate(
                decision_vector, runtime_args
            ),
            lower_bound,
            upper_bound,
            jac=lambda decision_vector: self.jacobian(
                decision_vector, runtime_args
            ),
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
        return normalise_runtime_args(runtime_args, self.parameter_bindings)

    def _materialise_inputs(
        self, decision_vector: np.ndarray, runtime_args: tuple[object, ...]
    ) -> list[object]:
        return materialise_tape_inputs(
            self.tape,
            self.decision_bindings,
            self.parameter_bindings,
            decision_vector,
            runtime_args,
        )

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

    bindings = build_problem_bindings(
        tape, [parameter.index for parameter in parameters]
    )
    decision_indices = bindings.decision_indices
    parameter_bindings = bindings.parameter_bindings
    decision_bindings = bindings.decision_bindings
    initial_guess = build_initial_guess(decision_bindings, initial_conditions)

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


def _make_bindings(indices, tape):
    return shared_make_bindings(indices, tape)


def _build_initial_guess(
    decision_bindings: list[InputBinding],
    initial_conditions: dict[int, object],
) -> np.ndarray:
    return build_initial_guess(decision_bindings, initial_conditions)


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

    def evaluate_bound(bound: object, runtime_args: tuple) -> np.ndarray:
        if isinstance(bound, Tracer):
            (value,) = problem._evaluate_tracers(
                [bound],
                np.zeros(decision_dimension, dtype=float),
                runtime_args,
            )
        else:
            value = bound
        return _expand_bound(value, residual.dim.flat())

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
        evaluate_lower_bound=lambda runtime_args: evaluate_bound(
            lower_bound, runtime_args
        ),
        evaluate_upper_bound=lambda runtime_args: evaluate_bound(
            upper_bound, runtime_args
        ),
    )


def _is_affine_in_decisions(
    tracer: Tracer | object, tape: Tape, decision_indices: set[int], memo=None
) -> bool:
    return is_affine_in_decisions(tracer, tape, decision_indices, memo)


def _decision_degree(
    tracer: Tracer | object,
    tape: Tape,
    decision_indices: set[int],
    memo: dict[int, int],
) -> int:
    return decision_degree(tracer, tape, decision_indices, memo)


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
    return coerce_scalar(value)


def _coerce_vector(value: object) -> np.ndarray:
    return coerce_vector(value)


def _expand_bound(bound: object, size: int) -> np.ndarray:
    values = np.asarray(bound, dtype=float)
    if values.ndim == 0:
        return np.full(size, float(values), dtype=float)
    return np.broadcast_to(values, (size,)).astype(float, copy=False)


def _normalise_value(value: object, dim: Dimension) -> object:
    return normalise_value(value, dim)


def _flatten_value(value: object, dim: Dimension) -> np.ndarray:
    return flatten_value(value, dim)


def _reshape_flat_slice(value: np.ndarray, dim: Dimension) -> object:
    return reshape_flat_slice(value, dim)
