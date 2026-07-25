from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from coker.algebra.kernel import Function, Tape, Tracer, function
from coker.backends.backend import get_backend_by_name
from coker.backends.evaluator import evaluate_inner
from coker.backends.optimisation import (
    InputBinding,
    build_initial_guess,
    build_problem_bindings,
    decision_degree,
    materialise_tape_inputs,
    normalise_runtime_args,
)

OSQP_INFINITY = 1.0e30


class RuntimeQpProgram(Protocol):
    def solve(
        self,
        runtime_args: tuple[object, ...],
        *,
        warm_start: np.ndarray | None,
    ) -> tuple[np.ndarray, object]: ...

    def push_forward(
        self, *tangent_spaces: object
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class OutputSlice:
    start: int
    length: int

    def to_export_dict(self):
        return {"start": int(self.start), "length": int(self.length)}


QP_ARENA_REGION_NAMES = [
    "pdata_x",
    "pdata",
    "adata_x",
    "adata",
    "qdata",
    "ldata",
    "udata",
    "data",
    "settings",
    "xsolution",
    "ysolution",
    "solution",
    "info",
    "qdldl_l_x",
    "qdldl_l",
    "qdldl_kkt_x",
    "qdldl_kkt",
    "qdldl",
    "qdldl_dinv",
    "qdldl_bp",
    "qdldl_sol",
    "qdldl_rho_inv_vec",
    "qdldl_d",
    "qdldl_iwork",
    "qdldl_bwork",
    "qdldl_fwork",
    "work_rho_vec",
    "work_rho_inv_vec",
    "work_constr_type",
    "work_x",
    "work_y",
    "work_z",
    "work_xz_tilde",
    "work_x_prev",
    "work_z_prev",
    "work_ax",
    "work_px",
    "work_aty",
    "work_delta_y",
    "work_atdelta_y",
    "work_delta_x",
    "work_pdelta_x",
    "work_adelta_x",
    "workspace",
]


def _csc_pattern_dict(
    nrows: int, ncols: int, indptr: list[int], indices: list[int]
):
    return {
        "nrows": int(nrows),
        "ncols": int(ncols),
        "indptr": [int(value) for value in indptr],
        "indices": [int(value) for value in indices],
    }


def _arena_layout_dict():
    layout = {"total_bytes": len(QP_ARENA_REGION_NAMES), "arena_alignment": 1}
    for offset, name in enumerate(QP_ARENA_REGION_NAMES):
        layout[name] = {
            "byte_offset": int(offset),
            "byte_len": 1,
            "byte_alignment": 1,
        }
    return layout


def _dense_symbolic_l(kkt_size: int):
    indptr = [0]
    indices: list[int] = []
    lnz: list[int] = []
    for col in range(kkt_size):
        rows = list(range(col + 1, kkt_size))
        indices.extend(rows)
        indptr.append(len(indices))
        lnz.append(len(rows))
    etree = [col + 1 for col in range(kkt_size - 1)] + [2**32 - 1]
    return {
        "l_pattern": _csc_pattern_dict(kkt_size, kkt_size, indptr, indices),
        "etree": etree,
        "lnz": lnz,
    }


def _build_qdldl_plan_payload(
    n: int,
    m: int,
    p_indptr: list[int],
    p_indices: list[int],
    a_indptr: list[int],
    a_indices: list[int],
):
    p_column_rows: list[list[int]] = []
    p_diag_indices: list[int] = []
    for col in range(n):
        original_rows = p_indices[p_indptr[col] : p_indptr[col + 1]]
        rows = sorted(original_rows)
        if col not in rows:
            rows.append(col)
            rows.sort()
        p_column_rows.append(rows)
        for local_index, row in enumerate(original_rows):
            if row == col:
                p_diag_indices.append(p_indptr[col] + local_index)

    constraint_rows: list[list[int]] = [[] for _ in range(m)]
    a_entry_positions: list[tuple[int, int]] = []
    for col in range(n):
        rows = a_indices[a_indptr[col] : a_indptr[col + 1]]
        for row in rows:
            constraint_rows[row].append(col)
            a_entry_positions.append((row, col))

    kkt_columns = list(p_column_rows)
    for row in range(m):
        rows = sorted(constraint_rows[row])
        rows.append(n + row)
        kkt_columns.append(rows)

    indptr = [0]
    indices: list[int] = []
    for rows in kkt_columns:
        indices.extend(rows)
        indptr.append(len(indices))

    p_to_kkt: list[int] = []
    for col in range(n):
        rows = kkt_columns[col]
        base = indptr[col]
        for row in p_indices[p_indptr[col] : p_indptr[col + 1]]:
            p_to_kkt.append(base + rows.index(row))

    a_to_kkt: list[int] = []
    for row, col in a_entry_positions:
        base = indptr[n + row]
        a_to_kkt.append(base + kkt_columns[n + row].index(col))

    rho_to_kkt: list[int] = []
    for row in range(m):
        base = indptr[n + row]
        rho_to_kkt.append(base + kkt_columns[n + row].index(n + row))

    kkt_size = n + m
    return {
        "p_pattern": _csc_pattern_dict(n, n, p_indptr, p_indices),
        "a_pattern": _csc_pattern_dict(m, n, a_indptr, a_indices),
        "kkt_pattern": _csc_pattern_dict(kkt_size, kkt_size, indptr, indices),
        "p_diag_indices": p_diag_indices,
        "kkt_permutation": list(range(kkt_size)),
        "p_to_kkt": p_to_kkt,
        "a_to_kkt": a_to_kkt,
        "rho_to_kkt": rho_to_kkt,
        "symbolic_l": _dense_symbolic_l(kkt_size),
    }


@dataclass(frozen=True)
class ExtractedQpProgram:
    n: int
    m: int
    parameter_bindings: list[InputBinding]
    decision_bindings: list[InputBinding]
    decision_indices: list[int]
    constraint_row_offsets: list[int]
    p_indptr: list[int]
    p_indices: list[int]
    a_indptr: list[int]
    a_indices: list[int]
    coefficient_function_id: int
    coefficient_payload: dict[str, object]
    coefficient_slices: dict[str, OutputSlice]
    warm_start: bool

    def export_payload(self) -> dict[str, object]:
        coefficient_functions = list(self.coefficient_payload["functions"])
        coefficient_function_id = int(self.coefficient_function_id)
        qp_function_id = (
            max(function["function_id"] for function in coefficient_functions)
            + 1
        )
        output_spec = {"memory": {"location": 0, "count": self.n}}
        input_specs = [
            {
                "memory": {
                    "location": binding.start,
                    "count": binding.dim.flat(),
                }
            }
            for binding in self.parameter_bindings
        ]
        embedded_plan = {
            "abi_version": 1,
            "profile": "Osqp063Embedded2Qdldl",
            "version": 1,
            "settings": {
                "rho": 0.1,
                "sigma": 1.0e-6,
                "alpha": 1.6,
                "adaptive_rho": True,
                "adaptive_rho_interval": 50,
                "adaptive_rho_tolerance": 5.0,
                "max_iter": 4000,
                "eps_abs": 1.0e-3,
                "eps_rel": 1.0e-3,
                "eps_prim_inf": 1.0e-4,
                "eps_dual_inf": 1.0e-4,
                "scaling": 0,
                "scaled_termination": False,
                "check_termination": 25,
                "warm_start": bool(self.warm_start),
                "linsys_solver": "Qdldl",
            },
            "arena_layout": _arena_layout_dict(),
            "qdldl_plan": _build_qdldl_plan_payload(
                self.n,
                self.m,
                self.p_indptr,
                self.p_indices,
                self.a_indptr,
                self.a_indices,
            ),
        }
        return {
            "functions": coefficient_functions,
            "qp_programs": [
                {
                    "function_id": qp_function_id,
                    "coefficient_function_id": coefficient_function_id,
                    "required_primal_workspace_size": self.n,
                    "required_tangent_workspace_size": self.n,
                    "input_specs": input_specs,
                    "output_spec": output_spec,
                    "p_pattern": _csc_pattern_dict(
                        self.n, self.n, self.p_indptr, self.p_indices
                    ),
                    "a_pattern": _csc_pattern_dict(
                        self.m, self.n, self.a_indptr, self.a_indices
                    ),
                    "coefficient_outputs": {
                        name: output_slice.to_export_dict()
                        for (
                            name,
                            output_slice,
                        ) in self.coefficient_slices.items()
                    },
                    "embedded_plan": embedded_plan,
                }
            ],
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
        warm_start = self.initial_guess
        if self.warm_start:
            warm_start = self._warm_start_vector
        solution, solve_info = self.runtime_qp.solve(
            runtime_args_tuple, warm_start=warm_start
        )
        if self.warm_start and solve_info.success:
            refined_solution, refined_info = self.runtime_qp.solve(
                runtime_args_tuple, warm_start=solution
            )
            if refined_info.success:
                solution = refined_solution
                solve_info = refined_info
        self.last_solve_info = solve_info
        if not solve_info.success:
            from coker.optimisation import SolveFailure

            raise SolveFailure(
                f"OSQP solve failed: {solve_info.return_status}", solve_info
            )
        if self.warm_start:
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
                self.tape,
                tape_inputs,
                list(self.outputs),
                get_backend_by_name("numpy", set_current=False),
                {},
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
        constraint.as_halfplane_bound()[0].dim.flat()
        for constraint in constraints
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

    p_indptr, p_indices = _pack_csc_upper_triangular(n)
    a_indptr, a_indices = _pack_csc_rectangular(m, n)
    return ExtractedQpProgram(
        n=n,
        m=m,
        parameter_bindings=parameter_bindings,
        decision_bindings=decision_bindings,
        decision_indices=decision_indices,
        constraint_row_offsets=row_offsets,
        p_indptr=p_indptr,
        p_indices=p_indices,
        a_indptr=a_indptr,
        a_indices=a_indices,
        coefficient_function_id=coefficient_graph.function_id,
        coefficient_payload=coefficient_graph.export_payload(),
        coefficient_slices=coefficient_slices,
        warm_start=warm_start,
    )


def _pack_csc_upper_triangular(n: int) -> tuple[list[int], list[int]]:
    indptr = [0]
    indices: list[int] = []
    for col in range(n):
        for row in range(col + 1):
            indices.append(row)
        indptr.append(len(indices))
    return indptr, indices


def _pack_csc_rectangular(
    nrows: int, ncols: int
) -> tuple[list[int], list[int]]:
    indptr = [0]
    indices: list[int] = []
    for _col in range(ncols):
        for row in range(nrows):
            indices.append(row)
        indptr.append(len(indices))
    return indptr, indices


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
    _ = backend, initial_conditions
    extracted = extract_qp_program(
        cost,
        constraints,
        outputs,
        decision_indices,
        decision_bindings,
        parameter_bindings,
    )
    from coker.backends.coker.runtime import RuntimeQpProgram

    return RuntimeQpProgram.compile(extracted)


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

    constraint_data = [
        constraint.as_halfplane_bound() for constraint in constraints
    ]

    parameter_spaces = [
        binding.dim.to_space(tape.input_names[binding.index])
        for binding in parameter_bindings
    ]

    px_length = decision_dimension * (decision_dimension + 1) // 2
    q_length = decision_dimension
    ax_length = sum(
        residual.dim.flat() * decision_dimension
        for residual, _lower_bound, _upper_bound in constraint_data
    )
    bound_length = sum(
        residual.dim.flat()
        for residual, _lower_bound, _upper_bound in constraint_data
    )

    coefficient_slices = _build_output_slices(
        px_length=px_length,
        q_length=q_length,
        ax_length=ax_length,
        bound_length=bound_length,
    )

    def implementation(*runtime_args):
        f_zero = _evaluate_function(
            cost,
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
                cost,
                tape,
                decision_bindings,
                parameter_bindings,
                basis,
                runtime_args,
            )
            negative = _evaluate_function(
                cost,
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
                    cost,
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
        for column in range(decision_dimension):
            value_index = column * (column + 1) // 2
            for row in range(column):
                ordered_px_values.append(px_values[value_index + 1 + row])
            ordered_px_values.append(px_values[value_index])

        ax_values: list[object] = []
        lower_values: list[object] = []
        upper_values: list[object] = []
        for _residual, lower_bound, upper_bound in constraint_data:
            residual_zero = _flatten_symbolic_vector(
                _evaluate_function(
                    _residual,
                    tape,
                    decision_bindings,
                    parameter_bindings,
                    zero_decision,
                    runtime_args,
                )
            )
            row_count = residual_zero.size
            lower_bound = _osqp_bound(lower_bound)
            upper_bound = _osqp_bound(upper_bound)
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
                        _residual,
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

        if not parameter_bindings:
            return [
                np.asarray(ordered_px_values, dtype=float),
                np.asarray(q_values, dtype=float),
                np.asarray(ax_values, dtype=float),
                np.asarray(lower_values, dtype=float),
                np.asarray(upper_values, dtype=float),
                np.asarray([float(f_zero)], dtype=float),
            ]

        return [
            *ordered_px_values,
            *q_values,
            *ax_values,
            *lower_values,
            *upper_values,
            f_zero,
        ]

    return (
        function(parameter_spaces, implementation, backend="numpy"),
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


def _osqp_bound(value: float) -> float:
    if np.isposinf(value):
        return OSQP_INFINITY
    if np.isneginf(value):
        return -OSQP_INFINITY
    return float(value)


def _evaluate_function(
    tracer: Tracer,
    tape: Tape,
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
    decision_vector: np.ndarray,
    runtime_args: tuple[object, ...],
):
    return evaluate_inner(
        tape,
        materialise_tape_inputs(
            tape,
            decision_bindings,
            parameter_bindings,
            np.asarray(decision_vector, dtype=float),
            runtime_args,
        ),
        [tracer],
        get_backend_by_name("numpy", set_current=False),
        {},
    )[0]


def _flatten_symbolic_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=object).reshape(-1, order="C")


def _symbolic_vector(
    values: list[object], zero: object
) -> np.ndarray | list[object]:
    if not values:
        return [0.0 * zero]
    return values
