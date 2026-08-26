"""Coker QP runtime-facing value types and solver adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from coker.algebra.kernel import Tape, Tracer
from coker.backends.backend import get_backend_by_name
from coker.backends.evaluator import evaluate_inner
from coker.backends.optimisation import (
    InputBinding,
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
    "qdldl_l_p",
    "qdldl_l_i",
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
        "nnz": len(indices),
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


def _symbolic_l_from_kkt(kkt_size: int, indptr: list[int], indices: list[int]):
    """Build natural-order LDL fill from upper-triangular KKT CSC."""
    adjacency = [set() for _ in range(kkt_size)]
    for column in range(kkt_size):
        for row in indices[indptr[column] : indptr[column + 1]]:
            if row == column:
                continue
            adjacency[row].add(column)
            adjacency[column].add(row)

    l_indptr = [0]
    l_indices: list[int] = []
    etree: list[int] = []
    lnz: list[int] = []
    for column in range(kkt_size):
        later = sorted(row for row in adjacency[column] if row > column)
        l_indices.extend(later)
        l_indptr.append(len(l_indices))
        lnz.append(len(later))
        etree.append(later[0] if later else 2**32 - 1)
        for index, left in enumerate(later):
            adjacency[left].discard(column)
            adjacency[left].update(later[index + 1 :])

    return {
        "l_pattern": _csc_pattern_dict(
            kkt_size, kkt_size, l_indptr, l_indices
        ),
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
        "symbolic_l": _symbolic_l_from_kkt(kkt_size, indptr, indices),
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
            "abi_version": 3,
            "profile": "Osqp063Embedded2Qdldl",
            "version": 3,
            "settings": {
                "rho": 0.1,
                "sigma": 1.0e-6,
                "alpha": 1.6,
                "adaptive_rho": True,
                "adaptive_rho_interval": 50,
                "adaptive_rho_tolerance": 5.0,
                "max_iter": 4000,
                "eps_abs": 1.0e-6,
                "eps_rel": 1.0e-6,
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
        extracted_qp: ExtractedQpProgram,
        warm_start: bool = True,
    ):
        self.tape = tape
        self.backend = backend
        self.decision_bindings = decision_bindings
        self.parameter_bindings = parameter_bindings
        self.outputs = outputs
        self.initial_guess = np.asarray(initial_guess, dtype=float)
        self.runtime_qp = runtime_qp
        self._extracted_qp = extracted_qp
        self.warm_start = warm_start
        self.last_solve_info = None
        self._warm_start_vector = self.initial_guess.copy()

    def export_payload(self) -> dict[str, object]:
        """Return the deterministic QP payload used by the Coker compiler."""
        return self._extracted_qp.export_payload()

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
            for _ in range(8):
                refined_solution, refined_info = self.runtime_qp.solve(
                    runtime_args_tuple, warm_start=solution
                )
                if not refined_info.success:
                    break
                previous_solution = solution
                solution = refined_solution
                solve_info = refined_info
                if np.allclose(
                    solution, previous_solution, atol=1.0e-6, rtol=0.0
                ):
                    break
        self.last_solve_info = solve_info
        if not solve_info.success:
            from coker.toolkits.codesign.optimisation import SolveFailure

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
