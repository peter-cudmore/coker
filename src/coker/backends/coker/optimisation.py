from __future__ import annotations

from typing import Protocol

import numpy as np

from coker.algebra.kernel import Tape, Tracer
from coker.backends.evaluator import evaluate_inner
from coker.backends.optimisation import (
    InputBinding,
    build_initial_guess,
    build_problem_bindings,
    materialise_tape_inputs,
    normalise_runtime_args,
)


class RuntimeQpProgram(Protocol):
    def solve(
        self,
        runtime_args: tuple[object, ...],
        *,
        warm_start: np.ndarray | None,
    ) -> tuple[np.ndarray, object]: ...


class CokerSolver:
    def __init__(
        self,
        *,
        tape: Tape,
        backend,
        decision_bindings: list[InputBinding],
        parameter_bindings: list[InputBinding],
        outputs: list[Tracer],
        initial_guess: np.ndarray,
        runtime_qp: RuntimeQpProgram,
        warm_start: bool = True,
    ):
        self.tape = tape
        self.backend = backend
        self.decision_bindings = decision_bindings
        self.parameter_bindings = parameter_bindings
        self.outputs = outputs
        self.initial_guess = np.asarray(initial_guess, dtype=float)
        self.runtime_qp = runtime_qp
        self.warm_start = warm_start
        self.last_solve_info = None
        self._warm_start_vector = self.initial_guess.copy()

    def __call__(self, *runtime_args):
        runtime_args_tuple = normalise_runtime_args(
            runtime_args, self.parameter_bindings
        )
        warm_start = self._warm_start_vector if self.warm_start else None
        solution, solve_info = self.runtime_qp.solve(
            runtime_args_tuple, warm_start=warm_start
        )
        self.last_solve_info = solve_info
        self._warm_start_vector = np.asarray(solution, dtype=float).copy()
        return self._evaluate_outputs(solution, runtime_args_tuple)

    def _evaluate_outputs(
        self, decision_vector: np.ndarray, runtime_args: tuple[object, ...]
    ) -> list[object]:
        tape_inputs = materialise_tape_inputs(
            self.tape,
            self.decision_bindings,
            self.parameter_bindings,
            np.asarray(decision_vector, dtype=float),
            runtime_args,
        )
        return list(
            evaluate_inner(
                self.tape, tape_inputs, list(self.outputs), self.backend, {}
            )
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
    if tape is None:
        raise ValueError("Optimisation cost must belong to a tape")
    assert all(constraint.tape == tape for constraint in constraints)
    assert all(parameter.tape == tape for parameter in parameters)
    assert all(output.tape == tape for output in outputs)

    bindings = build_problem_bindings(
        tape, [parameter.index for parameter in parameters]
    )
    runtime_qp = compile_qp_problem(
        backend,
        cost,
        constraints,
        outputs,
        bindings.decision_indices,
        bindings.decision_bindings,
        bindings.parameter_bindings,
        initial_conditions,
    )
    return CokerSolver(
        tape=tape,
        backend=backend,
        decision_bindings=bindings.decision_bindings,
        parameter_bindings=bindings.parameter_bindings,
        outputs=outputs,
        initial_guess=build_initial_guess(
            bindings.decision_bindings, initial_conditions
        ),
        runtime_qp=runtime_qp,
    )


def compile_qp_problem(
    backend,
    cost: Tracer,
    constraints: list[Tracer],
    outputs: list[Tracer],
    decision_indices: list[int],
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
    initial_conditions: dict[int, object],
) -> RuntimeQpProgram:
    raise NotImplementedError("Coker QP compilation is not implemented yet")
