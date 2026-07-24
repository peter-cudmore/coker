from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from coker.algebra import Dimension, OP
from coker.algebra.kernel import Tape, Tracer


@dataclass(frozen=True)
class InputBinding:
    index: int
    dim: Dimension
    start: int
    stop: int


@dataclass(frozen=True)
class ProblemBindings:
    tape: Tape
    decision_indices: list[int]
    decision_bindings: list[InputBinding]
    parameter_bindings: list[InputBinding]

    @property
    def decision_dimension(self) -> int:
        if not self.decision_bindings:
            return 0
        return self.decision_bindings[-1].stop

    @property
    def parameter_spaces(self):
        return [
            binding.dim.to_space(self.tape.input_names[binding.index])
            for binding in self.parameter_bindings
        ]


def build_problem_bindings(
    tape: Tape, parameter_indices: Iterable[int]
) -> ProblemBindings:
    parameter_index_list = list(parameter_indices)
    parameter_index_set = set(parameter_index_list)
    decision_indices = [
        index for index in tape.input_indicies if index not in parameter_index_set
    ]
    return ProblemBindings(
        tape=tape,
        decision_indices=decision_indices,
        decision_bindings=make_bindings(decision_indices, tape),
        parameter_bindings=make_bindings(parameter_index_list, tape),
    )


def make_bindings(indices: Iterable[int], tape: Tape) -> list[InputBinding]:
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


def build_initial_guess(
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
        flat_slices.append(flatten_value(initial_conditions[binding.index], binding.dim))
    return np.concatenate(flat_slices)


def normalise_runtime_args(
    runtime_args: Sequence[object], parameter_bindings: list[InputBinding]
) -> tuple[object, ...]:
    if len(runtime_args) != len(parameter_bindings):
        raise ValueError(
            "Expected "
            f"{len(parameter_bindings)} runtime arguments, got "
            f"{len(runtime_args)}"
        )
    return tuple(
        normalise_value(value, binding.dim)
        for value, binding in zip(runtime_args, parameter_bindings)
    )


def materialise_tape_inputs(
    tape: Tape,
    decision_bindings: list[InputBinding],
    parameter_bindings: list[InputBinding],
    decision_vector: np.ndarray,
    runtime_args: tuple[object, ...],
) -> list[object]:
    decision_values = {
        binding.index: reshape_flat_slice(
            decision_vector[binding.start : binding.stop], binding.dim
        )
        for binding in decision_bindings
    }
    parameter_values = {
        binding.index: value
        for binding, value in zip(parameter_bindings, runtime_args)
    }

    tape_inputs = []
    for index in tape.input_indicies:
        if index in decision_values:
            tape_inputs.append(decision_values[index])
            continue
        if index in parameter_values:
            tape_inputs.append(parameter_values[index])
            continue
        raise ValueError(f"Missing optimisation input for tape index {index}")
    return tape_inputs


def decision_degree(
    tracer: Tracer | object,
    tape: Tape,
    decision_indices: set[int],
    memo: dict[int, int] | None = None,
) -> int:
    if memo is None:
        memo = {}
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
        degree = decision_degree(arguments[0], tape, decision_indices, memo)
    elif op.is_linear():
        degree = max(
            (
                decision_degree(argument, tape, decision_indices, memo)
                for argument in arguments
            ),
            default=0,
        )
    elif op.is_bilinear():
        degree = sum(
            decision_degree(argument, tape, decision_indices, memo)
            for argument in arguments
        )
    else:
        argument_degrees = [
            decision_degree(argument, tape, decision_indices, memo)
            for argument in arguments
        ]
        degree = 0 if all(degree == 0 for degree in argument_degrees) else 3

    memo[tracer.index] = degree
    return degree


def is_affine_in_decisions(
    tracer: Tracer | object,
    tape: Tape,
    decision_indices: set[int],
    memo: dict[int, int] | None = None,
) -> bool:
    return decision_degree(tracer, tape, decision_indices, memo) <= 1


def coerce_scalar(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise TypeError(f"Expected scalar value, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def coerce_vector(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def normalise_value(value: object, dim: Dimension) -> object:
    array = np.asarray(value, dtype=float)
    if dim.is_scalar():
        if array.size != 1:
            raise ValueError(
                f"Expected scalar value for {dim}, got shape {array.shape}"
            )
        return float(array.reshape(-1)[0])
    if array.shape != dim.dim:
        raise ValueError(f"Expected value with shape {dim.dim}, got {array.shape}")
    return array


def flatten_value(value: object, dim: Dimension) -> np.ndarray:
    normalised_value = normalise_value(value, dim)
    if dim.is_scalar():
        return np.array([normalised_value], dtype=float)
    return np.asarray(normalised_value, dtype=float).reshape(-1)


def reshape_flat_slice(value: np.ndarray, dim: Dimension) -> object:
    if dim.is_scalar():
        if value.size != 1:
            raise ValueError(f"Expected scalar slice, got shape {value.shape}")
        return float(value[0])
    return np.asarray(value, dtype=float).reshape(dim.dim)
