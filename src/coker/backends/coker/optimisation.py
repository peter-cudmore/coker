from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from coker.algebra.kernel import Function, Tape, Tracer, function
from coker.backends.evaluator import evaluate_inner
from coker.backends.optimisation import (
    InputBinding,
    build_initial_guess,
    build_problem_bindings,
    decision_degree,
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


@dataclass(frozen=True)
class SparseEntryPattern:
    row: int
    col: int

    def to_export_dict(self) -> dict[str, int]:
        return {"row": int(self.row), "col": int(self.col)}


@dataclass(frozen=True)
class OutputSlice:
    start: int
    length: int

    def to_export_dict(self) -> dict[str, int]:
        return {"start": int(self.start), "length": int(self.length)}


@dataclass(frozen=True)
class ExtractedQpProgram:
    n: int
    m: int
    parameter_bindings: list[InputBinding]
    decision_bindings: list[InputBinding]
    decision_indices: list[int]
    constraint_row_offsets: list[int]
    p_structure: list[SparseEntryPattern]
    a_structure: list[SparseEntryPattern]
    coefficient_payload: dict[str, object]
    coefficient_slices: dict[str, OutputSlice]
    warm_start: bool

    def export_payload(self) -> dict[str, object]:
        return {
            "program": {
                "n": self.n,
                "m": self.m,
                "parameter_inputs": [
                    {"length": binding.dim.flat()}
                    for binding in self.parameter_bindings
                ],
                "decision_input_indices": [
                    int(index) for index in self.decision_indices
                ],
                "constraint_row_offsets": [
                    int(offset) for offset in self.constraint_row_offsets
                ],
                "p_structure": [
                    entry.to_export_dict() for entry in self.p_structure
                ],
                "a_structure": [
                    entry.to_export_dict() for entry in self.a_structure
                ],
                "coefficient_outputs": {
                    name: output_slice.to_export_dict()
                    for name, output_slice in self.coefficient_slices.items()
                },
                "coefficient_evaluator": self.coefficient_payload,
                "warm_start": self.warm_start,
            }
        }


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


def validate_qp_problem(
    tape: Tape,
    cost: Tracer,
    constraints: list[Tracer],
    decision_indices: list[int],
) -> None:
    if not cost.dim.is_scalar():
        raise ValueError("Coker QP objective must be scalar")

    decision_index_set = set(decision_indices)
    if decision_degree(cost, tape, decision_index_set) > 2:
        raise ValueError("Coker QP objective must be at most quadratic")

    for input_index in tape.input_indicies:
        if tape.dim[input_index] is None:
            raise ValueError("Coker QP inputs must have known dimensions")

    for constraint in constraints:
        residual, _lower_bound, _upper_bound = constraint.as_halfplane_bound()
        if decision_degree(residual, tape, decision_index_set) > 1:
            raise ValueError(
                "Coker QP constraints must be affine in decision variables"
            )


def extract_qp_program(
    cost: Tracer,
    constraints: list[Tracer],
    outputs: list[Tracer],
    decision_indices: list[int],
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
    *,
    warm_start: bool = True,
) -> ExtractedQpProgram:
    tape = cost.tape
    if tape is None:
        raise ValueError("Optimisation cost must belong to a tape")

    validate_qp_problem(tape, cost, constraints, decision_indices)

    n = decision_bindings[-1].stop if decision_bindings else 0
    constraint_rows = [
        constraint.as_halfplane_bound()[0].dim.flat() for constraint in constraints
    ]
    row_offsets = [0]
    for row_count in constraint_rows:
        row_offsets.append(row_offsets[-1] + row_count)
    m = row_offsets[-1]

    coefficient_function, coefficient_slices = _build_coefficient_function(
        tape,
        cost,
        constraints,
        decision_bindings,
        parameter_bindings,
    )
    from coker.backends.coker.core import create_opgraph

    coefficient_graph = create_opgraph(coefficient_function)

    p_structure = [
        SparseEntryPattern(row=row, col=col)
        for col in range(n)
        for row in range(col + 1)
    ]
    a_structure = [
        SparseEntryPattern(row=row, col=col)
        for col in range(n)
        for row in range(m)
    ]

    return ExtractedQpProgram(
        n=n,
        m=m,
        parameter_bindings=parameter_bindings,
        decision_bindings=decision_bindings,
        decision_indices=decision_indices,
        constraint_row_offsets=row_offsets,
        p_structure=p_structure,
        a_structure=a_structure,
        coefficient_payload=coefficient_graph.export_payload(),
        coefficient_slices=coefficient_slices,
        warm_start=warm_start,
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
    _ = (
        backend,
        outputs,
        initial_conditions,
        extract_qp_program(
            cost,
            constraints,
            outputs,
            decision_indices,
            decision_bindings,
            parameter_bindings,
        ),
    )
    raise NotImplementedError("Coker QP runtime bindings are not implemented yet")


def _build_coefficient_function(
    tape: Tape,
    cost: Tracer,
    constraints: list[Tracer],
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
) -> tuple[Function, dict[str, OutputSlice]]:
    decision_dimension = decision_bindings[-1].stop if decision_bindings else 0
    zero_decision = np.zeros(decision_dimension, dtype=float)
    basis_vectors = np.eye(decision_dimension, dtype=float)

    cost_function = Function(tape, cost, backend="numpy")
    constraint_data = [
        constraint.as_halfplane_bound() for constraint in constraints
    ]
    constraint_functions = [
        Function(tape, residual, backend="numpy")
        for residual, _lower_bound, _upper_bound in constraint_data
    ]

    parameter_spaces = [
        binding.dim.to_space(tape.input_names[binding.index])
        for binding in parameter_bindings
    ]

    px_length = decision_dimension * (decision_dimension + 1) // 2
    q_length = decision_dimension
    ax_length = sum(
        constraint_function.output[0].dim.flat() * decision_dimension
        for constraint_function in constraint_functions
    )
    bound_length = sum(
        constraint_function.output[0].dim.flat()
        for constraint_function in constraint_functions
    )

    coefficient_slices = _build_output_slices(
        px_length=px_length,
        q_length=q_length,
        ax_length=ax_length,
        bound_length=bound_length,
    )

    def implementation(*runtime_args):
        f_zero = _evaluate_function(
            cost_function,
            tape,
            decision_bindings,
            parameter_bindings,
            zero_decision,
            runtime_args,
        )

        px_values: list[object] = []
        q_values: list[object] = []
        positive_costs = []
        for column in range(decision_dimension):
            basis = basis_vectors[column]
            positive = _evaluate_function(
                cost_function,
                tape,
                decision_bindings,
                parameter_bindings,
                basis,
                runtime_args,
            )
            negative = _evaluate_function(
                cost_function,
                tape,
                decision_bindings,
                parameter_bindings,
                -basis,
                runtime_args,
            )
            positive_costs.append(positive)
            px_values.append(positive + negative - (2.0 * f_zero))
            q_values.append(0.5 * (positive - negative))

            for row in range(column):
                pair_basis = basis + basis_vectors[row]
                pair_value = _evaluate_function(
                    cost_function,
                    tape,
                    decision_bindings,
                    parameter_bindings,
                    pair_basis,
                    runtime_args,
                )
                px_values.append(
                    pair_value - positive_costs[row] - positive + f_zero
                )

        ordered_px_values = []
        value_index = 0
        for column in range(decision_dimension):
            ordered_px_values.append(px_values[value_index])
            value_index += 1
            for row in range(column):
                ordered_px_values.append(px_values[value_index])
                value_index += 1

        ax_values: list[object] = []
        lower_values: list[object] = []
        upper_values: list[object] = []
        for (
            _residual,
            lower_bound,
            upper_bound,
        ), residual_function in zip(
            constraint_data, constraint_functions, strict=False
        ):
            residual_zero = _flatten_symbolic_vector(
                _evaluate_function(
                    residual_function,
                    tape,
                    decision_bindings,
                    parameter_bindings,
                    zero_decision,
                    runtime_args,
                )
            )
            row_count = residual_zero.size
            lower_values.extend(
                lower_bound - residual_zero[row] for row in range(row_count)
            )
            upper_values.extend(
                upper_bound - residual_zero[row] for row in range(row_count)
            )

            for column in range(decision_dimension):
                basis = basis_vectors[column]
                residual_basis = _flatten_symbolic_vector(
                    _evaluate_function(
                        residual_function,
                        tape,
                        decision_bindings,
                        parameter_bindings,
                        basis,
                        runtime_args,
                    )
                )
                ax_values.extend(
                    residual_basis[row] - residual_zero[row]
                    for row in range(row_count)
                )

        return [
            _symbolic_vector(ordered_px_values, f_zero),
            _symbolic_vector(q_values, f_zero),
            _symbolic_vector(ax_values, f_zero),
            _symbolic_vector(lower_values, f_zero),
            _symbolic_vector(upper_values, f_zero),
            f_zero,
        ]

    return (
        function(parameter_spaces, implementation, backend="coker"),
        coefficient_slices,
    )


def _build_output_slices(
    *, px_length: int, q_length: int, ax_length: int, bound_length: int
) -> dict[str, OutputSlice]:
    start = 0
    slices = {}
    for name, length in (
        ("px", px_length),
        ("q", q_length),
        ("ax", ax_length),
        ("l", bound_length),
        ("u", bound_length),
        ("r", 1),
    ):
        slices[name] = OutputSlice(start=start, length=length)
        start += length
    return slices


def _evaluate_function(
    compiled_function: Function,
    tape: Tape,
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
    decision_vector: np.ndarray,
    runtime_args: tuple[object, ...],
):
    return compiled_function(
        *materialise_tape_inputs(
            tape,
            decision_bindings,
            parameter_bindings,
            np.asarray(decision_vector, dtype=float),
            runtime_args,
        )
    )


def _flatten_symbolic_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=object).reshape(-1, order="C")


def _symbolic_vector(values: list[object], zero: object) -> np.ndarray:
    if not values:
        return np.asarray([0.0 * zero], dtype=object)
    return np.asarray(values, dtype=object)
