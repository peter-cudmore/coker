"""Coker tape-to-SparseNet lowering phases.

This module is the data boundary between the backend facade and graph consumers.
It accepts a :class:`Function` and produces a :class:`SparseNet`; callers use
the graph rather than reaching into another lowering phase.
"""

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse

from coker.algebra.kernel import Function, Tracer
from coker.algebra.ops import ConcatenateOP, OP, ReshapeOP
from coker.backends.backend import get_backend_by_name
from coker.backends.coker.ast_preprocessing import FunctionTable, SparseNet
from coker.backends.coker.layers import (
    IDENTITY_OP,
    UNUSED_REF,
    BilinearWorkspaceLayer,
    ConstantOperand,
    FunctionEvaluationLayer,
    GenericVectorLayer,
    InputLayer,
    OpaqueProgram,
    OutputLayer,
    WorkspaceOperand,
)
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.op_impl import ops
from coker.backends.coker.sparse_tensor import dok_ndarray
from coker.backends.coker.weights import BilinearWeights

def _node_shape(dimension):
    return dimension.shape if hasattr(dimension, "shape") else None


def _as_numpy_value(value):
    if scipy.sparse.issparse(value):
        value = value.toarray()
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    if isinstance(value, (float, int, bool, np.bool_)):
        return value
    return value


def _constant_array(value, shape: Tuple[int, ...]) -> np.ndarray:
    value = _as_numpy_value(value)
    if isinstance(value, np.ndarray):
        return np.reshape(value, shape, order="C")
    return np.array([value]).reshape(shape, order="C")


def _constant_to_bw(
    memory: MemorySpec, value, shape: Tuple[int, ...]
) -> BilinearWeights:
    if scipy.sparse.issparse(value):
        constant = dok_ndarray.from_scipy(value.reshape(shape))
    else:
        constant = dok_ndarray.fromarray(_constant_array(value, shape))
    return BilinearWeights(memory, shape, constant=constant)


def _flatten_constant_rows(value, shape: Tuple[int, ...]) -> List[float]:
    flat = _constant_array(value, shape).reshape(-1, order="C")
    return [float(item) for item in flat]


def _append_bilinear_value(
    constant: dok_ndarray,
    linear: dok_ndarray,
    quadratic: dok_ndarray,
    start: int,
    weights: BilinearWeights,
    output_shape: Tuple[int, ...],
):
    output_rank = len(output_shape)
    for key, value in weights.constant.keys.items():
        row = np.ravel_multi_index(key[:output_rank], output_shape, order="C")
        target = (start + row,)
        constant[target] = constant[target] + value
    for key, value in weights.linear.keys.items():
        row = np.ravel_multi_index(key[:output_rank], output_shape, order="C")
        target = (start + row, key[-1])
        linear[target] = linear[target] + value
    for key, value in weights.quadratic.keys.items():
        row = np.ravel_multi_index(key[:output_rank], output_shape, order="C")
        target = (start + row, key[-2], key[-1])
        quadratic[target] = quadratic[target] + value


def _constant_extension_weights(
    memory: MemorySpec, current_size: int, constant_values: List[float]
) -> BilinearWeights:
    next_size = current_size + len(constant_values)
    constant = dok_ndarray((next_size,))
    linear = dok_ndarray((next_size, memory.count))
    quadratic = dok_ndarray((next_size, memory.count, memory.count))
    for row in range(current_size):
        linear[(row, row)] = 1
    for offset, value in enumerate(constant_values):
        constant[(current_size + offset,)] = value
    return BilinearWeights(
        memory,
        (next_size,),
        constant=constant,
        linear=linear,
        quadratic=quadratic,
    )


def _concatenate_bilinear_operands(
    memory: MemorySpec, operands: List, axis: int
) -> BilinearWeights:
    bilinear_operands = []
    for operand in operands:
        if isinstance(operand, BilinearWeights):
            assert operand.memory == memory
            bilinear_operands.append(operand)
            continue
        operand_array = np.asarray(_as_numpy_value(operand))
        bilinear_operands.append(
            _constant_to_bw(memory, operand_array, operand_array.shape)
        )

    if not bilinear_operands:
        raise ValueError("cannot concatenate no bilinear operands")
    rank = len(bilinear_operands[0].shape)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("concatenation axis is outside the operand rank")
    output_shape = list(bilinear_operands[0].shape)
    output_shape[axis] = 0
    for operand in bilinear_operands:
        if len(operand.shape) != rank or any(
            left != right
            for index, (left, right) in enumerate(
                zip(output_shape, operand.shape, strict=True)
            )
            if index != axis
        ):
            raise ValueError("bilinear concatenation has incompatible shapes")
        output_shape[axis] += operand.shape[axis]

    constant = dok_ndarray(tuple(output_shape))
    linear = dok_ndarray((*output_shape, memory.count))
    quadratic = dok_ndarray((*output_shape, memory.count, memory.count))
    offset = 0
    for operand in bilinear_operands:
        for key, value in operand.constant.keys.items():
            output_key = list(key)
            output_key[axis] += offset
            constant[tuple(output_key)] = value
        for key, value in operand.linear.keys.items():
            output_key = list(key[:rank])
            output_key[axis] += offset
            linear[(*output_key, key[-1])] = value
        for key, value in operand.quadratic.keys.items():
            output_key = list(key[:rank])
            output_key[axis] += offset
            quadratic[(*output_key, key[-2], key[-1])] = value
        offset += operand.shape[axis]
    return BilinearWeights.from_trusted_dok(
        memory,
        tuple(output_shape),
        constant=constant,
        linear=linear,
        quadratic=quadratic,
    )


def _build_opaque_operand(
    value, spec: MemorySpec | None, shape: Tuple[int, ...]
):
    if spec is not None:
        return WorkspaceOperand(spec, shape)
    if isinstance(value, np.ndarray):
        stored = dok_ndarray.fromarray(np.reshape(value, shape, order="C"))
        return ConstantOperand(stored, shape)
    if scipy.sparse.issparse(value):
        stored = dok_ndarray.from_scipy(value.reshape(shape))
        return ConstantOperand(stored, shape)
    return ConstantOperand(value, shape)




def create_compact_bilinear_opgraph(function: Function):
    """Lower a vector-valued bilinear graph in one fused workspace layer.

    This avoids relocation of intermediate bilinear weights. Matrix outputs
    use ordinary lowering because their tangent layout is not fused here.
    """
    if any(
        not (output.dim.is_scalar() or output.dim.is_vector())
        for output in function.output
        if output is not None
    ):
        return None
    tape = function.tape
    if tape is None:
        return None
    input_layer = InputLayer()
    input_specs = {}
    for input_index in tape.input_indicies:
        if input_index < 0:
            return None
        position = input_layer.add_input(tape.dim[input_index])
        input_specs[input_index] = input_layer.input_specs[position]
    memory = MemorySpec(0, input_layer.dimension)
    values = {}
    for input_index, (spec, shape) in input_specs.items():
        values[input_index] = BilinearWeights.project(memory, spec, shape)

    for index, node in enumerate(tape.nodes):
        if index in values:
            continue
        operation, *arguments = node
        shape = _node_shape(tape.dim[index])
        if operation == OP.VALUE:
            values[index] = _constant_to_bw(memory, arguments[0], shape)
            continue
        operands = []
        for argument in arguments:
            if not isinstance(argument, Tracer):
                operands.append(argument)
                continue
            value = values.get(argument.index)
            if value is None:
                return None
            operands.append(value)
        try:
            value = ops[operation](*operands)
        except (KeyError, AssertionError, TypeError, NotImplementedError):
            return None
        if not isinstance(value, BilinearWeights):
            return None
        values[index] = value

    outputs = []
    for output in function.output:
        if output is None:
            return None
        value = values.get(output.index)
        if value is None or value.memory != memory:
            return None
        outputs.append(value)
    if not outputs:
        return None
    fused = _concatenate_bilinear_operands(memory, outputs, axis=0)
    output_memory = MemorySpec(0, int(np.prod(fused.shape)))
    output_layer = OutputLayer()
    offset = 0
    for output in function.output:
        count = tape.dim[output.index].flat()
        output_layer.add_output(
            MemorySpec(offset, count), tape.dim[output.index]
        )
        offset += count
    workspace_memory = MemorySpec(0, max(memory.count, output_memory.count))
    graph = SparseNet(
        workspace_memory,
        input_layer,
        output_layer,
        [BilinearWorkspaceLayer(memory, output_memory, fused)],
    )
    graph.output_weights = outputs
    graph.output_weights_are_raw = True
    return graph


def create_function_table(function: Function) -> FunctionTable:
    """Lower a function module whose table owns every nested SparseNet."""
    compact = create_compact_bilinear_opgraph(function)
    if compact is not None:
        return FunctionTable([compact])
    from coker.backends.coker.unfused_lowering import build_function_table

    return build_function_table(function)


def create_unfused_opgraph(function: Function) -> SparseNet:
    """Lower a graph without materialising a whole-function bilinear tensor."""
    from coker.backends.coker.unfused_lowering import build_unfused_opgraph

    return build_unfused_opgraph(function)


def create_opgraph(function: Function) -> SparseNet:
    """Return the entry program for compatibility host-graph callers."""
    return create_function_table(function).entry


def _generic_layer_weights(layer: GenericVectorLayer):
    """Represent an exactly bilinear generic layer as a workspace mapping."""
    if layer.opaque_programs:
        rows = []
        for program in layer.opaque_programs:
            if program.op != OP.MATMUL or len(program.operands) != 2:
                return None
            left_operand, right_operand = program.operands
            if isinstance(left_operand, ConstantOperand):
                left = left_operand.value
            elif isinstance(left_operand, WorkspaceOperand):
                left = BilinearWeights.project(
                    layer.memory_in, left_operand.spec, left_operand.shape
                )
            else:
                return None
            if isinstance(right_operand, ConstantOperand):
                right = right_operand.value
            elif isinstance(right_operand, WorkspaceOperand):
                right = BilinearWeights.project(
                    layer.memory_in, right_operand.spec, right_operand.shape
                )
            else:
                return None
            try:
                mapped = left @ right
            except (
                AssertionError,
                TypeError,
                ValueError,
                NotImplementedError,
            ):
                return None
            if not isinstance(mapped, BilinearWeights):
                return None
            rows.extend(
                mapped[index] for index in range(int(np.prod(mapped.shape)))
            )
        return _concatenate_bilinear_operands(layer.memory_in, rows, axis=0)

    def operand(ref):
        if ref == UNUSED_REF:
            return None
        return BilinearWeights.project(
            layer.memory_in, MemorySpec(ref, 1), (1,)
        )

    rows = []
    for operation, first, second, third in layer.ops:
        if operation == IDENTITY_OP:
            value = operand(first)
        elif operation in ops:
            arguments = [
                value
                for value in (operand(first), operand(second), operand(third))
                if value is not None
            ]
            try:
                value = ops[operation](*arguments)
            except (AssertionError, TypeError, NotImplementedError):
                return None
        else:
            return None
        if not isinstance(value, BilinearWeights):
            return None
        rows.append(value)
    return _concatenate_bilinear_operands(layer.memory_in, rows, axis=0)


def _raw_output_weights(graph: SparseNet, output_weights):
    """Compose graph outputs back through workspace relocation layers."""
    from coker.backends.coker.weights import compose_bilinear_weights

    weights = list(output_weights)
    for layer in reversed(graph.intermediate_layers):
        if isinstance(layer, BilinearWorkspaceLayer):
            composed = []
            for weight in weights:
                if weight is None:
                    composed.append(None)
                    continue
                try:
                    composed.append(
                        compose_bilinear_weights(weight, layer.weights)
                    )
                except AssertionError:
                    return None
            if any(weight is None for weight in composed):
                return None
            weights = composed
            continue
        if isinstance(layer, GenericVectorLayer):
            mapping = _generic_layer_weights(layer)
            if mapping is None:
                return None
            composed = []
            for weight in weights:
                if weight is None:
                    composed.append(None)
                    continue
                try:
                    composed.append(compose_bilinear_weights(weight, mapping))
                except AssertionError:
                    return None
            if any(weight is None for weight in composed):
                return None
            weights = composed
            continue
        # Function evaluation is an opaque computation from provenance's
        # perspective, even when its output happens to be numeric.
        return None
    return weights


