from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from coker.algebra.kernel import Function, Tape, Tracer, function
from coker.algebra.sparse import SparseMatrixPattern
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
    extracted = extract_qp_program(
        cost,
        constraints,
        outputs,
        bindings.decision_indices,
        bindings.decision_bindings,
        bindings.parameter_bindings,
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
        extracted_qp=extracted,
    )


def validate_qp_problem(
    tape: Tape,
    cost: Tracer,
    constraints: list[Tracer],
    decision_indices: list[int],
) -> None:
    weighted_norm = getattr(cost, "weighted_norm", None)
    if weighted_norm is not None and not isinstance(
        getattr(weighted_norm.weight, "sparse_matrix_pattern", None),
        SparseMatrixPattern,
    ):
        raise ValueError(
            "Coker weighted-norm QP objectives require a "
            "SparseMatrixBuilder weight"
        )
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


def _bilinear_coefficient_function(
    tape: Tape,
    cost: Tracer,
    constraints: list[Tracer],
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
    constraint_data,
):
    """Extract QP coefficients without materialising weighted products."""
    from coker.backends.coker.core import create_opgraph

    def graph_for(tracer):
        return create_opgraph(Function(tape, [tracer], backend="numpy"))

    weighted_norm = getattr(cost, "weighted_norm", None)
    weighted_pattern = None
    weighted_residual_weights = None
    weighted_data_weights = None
    if weighted_norm is None:
        cost_graph = graph_for(cost)
        cost_weights = cost_graph.output_weights[0]
    else:
        weighted_pattern = getattr(
            weighted_norm.weight, "sparse_matrix_pattern", None
        )
        if not isinstance(weighted_pattern, SparseMatrixPattern):
            raise ValueError(
                "Coker weighted-norm QP objectives require a "
                "SparseMatrixBuilder weight"
            )
        if weighted_pattern.shape[1] != weighted_norm.residual.dim.flat():
            raise ValueError(
                "Coker weighted-norm weight and residual dimensions differ"
            )
        residual_graph = graph_for(weighted_norm.residual)
        data_graph = graph_for(weighted_pattern.data)
        weighted_residual_weights = residual_graph.output_weights[0]
        weighted_data_weights = data_graph.output_weights[0]
        if (
            not getattr(residual_graph, "output_weights_are_raw", False)
            or not getattr(data_graph, "output_weights_are_raw", False)
            or weighted_residual_weights is None
            or weighted_data_weights is None
        ):
            raise ValueError(
                "Coker weighted-norm QP inputs must have raw "
                "bilinear provenance"
            )
        cost_graph = None
        cost_weights = None

    residual_weights = []
    residual_graphs = []
    for residual, _lower, _upper in constraint_data:
        residual_graph = graph_for(residual)
        residual_graphs.append(residual_graph)
        residual_weights.append(residual_graph.output_weights[0])
    input_memory_count = sum(
        tape.dim[index].flat() for index in tape.input_indicies
    )
    if (
        (
            cost_graph is not None
            and not getattr(cost_graph, "output_weights_are_raw", False)
        )
        or any(
            not getattr(graph, "output_weights_are_raw", False)
            for graph in residual_graphs
        )
        or (cost_weights is None and weighted_norm is None)
        or any(weight is None for weight in residual_weights)
        or (
            cost_weights is not None
            and cost_weights.memory.count != input_memory_count
        )
        or (
            weighted_residual_weights is not None
            and weighted_residual_weights.memory.count != input_memory_count
        )
        or (
            weighted_data_weights is not None
            and weighted_data_weights.memory.count != input_memory_count
        )
        or any(
            weight.memory.count != input_memory_count
            for weight in residual_weights
        )
    ):
        # Workspace relocation or an opaque lowered operation changes the
        # coordinate system.  Never interpret those coordinates as tape inputs.
        return None
    bound_weights = {}
    for _residual, lower_bound, upper_bound in constraint_data:
        for bound in (lower_bound, upper_bound):
            if not isinstance(bound, Tracer):
                continue
            bound_graph = graph_for(bound)
            if not getattr(bound_graph, "output_weights_are_raw", False):
                return None
            weight = bound_graph.output_weights[0]
            if weight is None or weight.memory.count != input_memory_count:
                return None
            bound_weights[id(bound)] = weight
    memory_offsets = {}
    offset = 0
    for index in tape.input_indicies:
        size = tape.dim[index].flat()
        memory_offsets[index] = list(range(offset, offset + size))
        offset += size
    decision_memory = [
        coordinate
        for binding in decision_bindings
        for coordinate in memory_offsets[binding.index]
    ]
    parameter_memory = [
        coordinate
        for binding in parameter_bindings
        for coordinate in memory_offsets[binding.index]
    ]
    parameter_spaces = [
        binding.dim.to_space(tape.input_names[binding.index])
        for binding in parameter_bindings
    ]

    if weighted_data_weights is not None:
        data_indices = {
            index
            for table in (
                weighted_data_weights.linear.keys,
                weighted_data_weights.quadratic.keys,
            )
            for key in table
            for index in key[1:]
        }
        if data_indices.intersection(
            decision_memory
        ) or not data_indices.issubset(parameter_memory):
            raise ValueError(
                "Coker weighted-norm weights must depend only on QP parameters"
            )

    def lookup(table, key):
        return table.keys.get(key, 0.0)

    def expression(weights, row, params, decision=None, other=None):
        result = lookup(weights.constant, (row,))
        if decision is None:
            for (out_row, index), value in weights.linear.keys.items():
                if out_row == row:
                    result += value * params.get(index, 0.0)
            for (
                out_row,
                left,
                right,
            ), value in weights.quadratic.keys.items():
                if out_row == row:
                    result += (
                        value * params.get(left, 0.0) * params.get(right, 0.0)
                    )
            return result
        result = lookup(weights.linear, (row, decision))
        if other is not None:
            return lookup(weights.quadratic, (row, decision, other)) + lookup(
                weights.quadratic, (row, other, decision)
            )
        for (out_row, left, right), value in weights.quadratic.keys.items():
            if out_row != row:
                continue
            if left == decision:
                result += value * params.get(right, 0.0)
            elif right == decision:
                result += value * params.get(left, 0.0)
        return result

    residual_rows = [
        residual.dim.flat() for residual, _l, _u in constraint_data
    ]
    pairs = [
        (column, row)
        for column in range(len(decision_memory))
        for row in range(column + 1)
    ]
    if weighted_pattern is not None:
        weighted_rows = [[] for _ in range(weighted_pattern.shape[0])]
        for column in range(weighted_pattern.shape[1]):
            for data_index in range(
                weighted_pattern.indptr[column],
                weighted_pattern.indptr[column + 1],
            ):
                weighted_rows[weighted_pattern.indices[data_index]].append(
                    (column, data_index)
                )

    def implementation(*runtime_params):
        params = {
            coordinate: item
            for coordinate, item in zip(
                parameter_memory,
                (
                    item
                    for value in runtime_params
                    for item in _flatten_symbolic_vector(value)
                ),
            )
        }
        if weighted_pattern is None:
            px = [
                expression(
                    cost_weights,
                    0,
                    params,
                    decision=decision_memory[column],
                    other=decision_memory[row],
                )
                for column, row in pairs
            ]
            q = [
                expression(cost_weights, 0, params, decision=decision)
                for decision in decision_memory
            ]
            objective_offset = expression(cost_weights, 0, params)
        else:

            def weighted_component(row, decision=None):
                return sum(
                    expression(weighted_data_weights, data_index, params)
                    * expression(
                        weighted_residual_weights,
                        column,
                        params,
                        decision=decision,
                    )
                    for column, data_index in weighted_rows[row]
                )

            px = [
                2.0
                * sum(
                    weighted_component(row, decision_memory[column])
                    * weighted_component(row, decision_memory[inner_row])
                    for row in range(weighted_pattern.shape[0])
                )
                for column, inner_row in pairs
            ]
            q = [
                2.0
                * sum(
                    weighted_component(row, decision) * weighted_component(row)
                    for row in range(weighted_pattern.shape[0])
                )
                for decision in decision_memory
            ]
            objective_offset = sum(
                weighted_component(row) ** 2
                for row in range(weighted_pattern.shape[0])
            )
        ax = [
            expression(weights, row, params, decision=decision)
            for decision in decision_memory
            for weights, row_count in zip(
                residual_weights, residual_rows, strict=True
            )
            for row in range(row_count)
        ]

        def bound_values(bound, weight, row_count):
            if weight is not None:
                return [
                    expression(weight, row, params) for row in range(row_count)
                ]

            values = np.asarray(_osqp_bound(bound)).reshape(-1)
            if values.size == 1:
                return [values[0]] * row_count
            if values.size == row_count:
                return values.tolist()
            return None

        lower = []
        upper = []
        for (residual, lower_bound, upper_bound), weights in zip(
            constraint_data, residual_weights, strict=True
        ):
            row_count = residual.dim.flat()
            lower_values = bound_values(
                lower_bound, bound_weights.get(id(lower_bound)), row_count
            )
            upper_values = bound_values(
                upper_bound, bound_weights.get(id(upper_bound)), row_count
            )
            if lower_values is None or upper_values is None:
                return None
            for row, (lower_value, upper_value) in enumerate(
                zip(lower_values, upper_values, strict=True)
            ):
                residual_offset = expression(weights, row, params)
                lower.append(lower_value - residual_offset)
                upper.append(upper_value - residual_offset)
        return [*px, *q, *ax, *lower, *upper, objective_offset]

    function_outputs = function(
        parameter_spaces, implementation, backend="numpy"
    )
    return function_outputs, _build_output_slices(
        px_length=len(decision_memory) * (len(decision_memory) + 1) // 2,
        q_length=len(decision_memory),
        ax_length=sum(rows * len(decision_memory) for rows in residual_rows),
        bound_length=sum(rows for rows in residual_rows),
    )


def _build_coefficient_function(
    tape: Tape,
    cost: Tracer,
    constraints: list[Tracer],
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
) -> tuple[Function, dict[str, OutputSlice]]:
    constraint_data = [
        constraint.as_halfplane_bound() for constraint in constraints
    ]
    direct = _bilinear_coefficient_function(
        tape,
        cost,
        constraints,
        decision_bindings,
        parameter_bindings,
        constraint_data,
    )
    if direct is None:
        raise ValueError(
            "Coker QP coefficients must be exactly representable "
            "as raw bilinear weights"
        )
    return direct


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


def _osqp_bound(value):
    if isinstance(value, Tracer):
        return value
    if np.isposinf(value):
        return OSQP_INFINITY
    if np.isneginf(value):
        return -OSQP_INFINITY
    return float(value)


def _flatten_symbolic_vector(value) -> np.ndarray:
    if isinstance(value, Tracer) and not value.dim.is_scalar():
        return np.asarray(
            [value[index] for index in range(value.dim.flat())],
            dtype=object,
        )
    return np.asarray(value, dtype=object).reshape(-1, order="C")


def _symbolic_vector(
    values: list[object], zero: object
) -> np.ndarray | list[object]:
    if not values:
        return [0.0 * zero]
    return values
