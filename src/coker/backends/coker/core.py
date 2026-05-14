from typing import Dict, List, Tuple, Type

import numpy as np
import scipy.sparse

from coker.algebra.dimensions import FunctionSpace
from coker.algebra.kernel import Function, Tracer
from coker.algebra.ops import ConcatenateOP, OP, ReshapeOP
from coker.backends.backend import ArrayLike, Backend, get_backend_by_name
from coker.backends.coker.ast_preprocessing import SparseNet
from coker.backends.coker.layers import (
    IDENTITY_OP,
    OPAQUE_OP,
    UNUSED_REF,
    BilinearWorkspaceLayer,
    ConstantOperand,
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


class CokerBackend(Backend):
    def __init__(self):
        pass

    def to_backend_array(self, array: ArrayLike):
        pass

    def to_numpy_array(self, array) -> ArrayLike:
        pass

    def native_types(self) -> Tuple[Type]:
        pass

    def call(self, op, *args) -> ArrayLike:
        pass

    def evaluate(self, function, inputs: ArrayLike):
        if all(o is None for o in function.output):
            return function.output

        if any(
            isinstance(function.tape.dim[i], FunctionSpace)
            for i in function.tape.input_indicies
        ):
            numpy_backend = get_backend_by_name("numpy", set_current=False)
            return numpy_backend.evaluate(function, inputs)

        g = create_opgraph(function)
        return [g(*inputs)]

    def reshape(self, array: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
        raise NotImplementedError

    def lower(self, function: Function):
        if any(
            isinstance(function.tape.dim[i], FunctionSpace)
            for i in function.tape.input_indicies
        ) or any(o is None for o in function.output):
            numpy_backend = get_backend_by_name("numpy", set_current=False)
            return numpy_backend.lower(function)

        g = create_opgraph(function)

        def compiled(inputs):
            return [g(*inputs)]

        return compiled

    def build_optimisation_problem(
        self,
        cost: Tracer,
        constraints: List[Tracer],
        parameters: List[Tracer],
        outputs: List[Tracer],
        initial_conditions: Dict[int, ArrayLike],
    ):
        raise NotImplementedError


def _node_shape(dim):
    return dim.shape if hasattr(dim, "shape") else None


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


def _constant_to_bw(memory: MemorySpec, value, shape: Tuple[int, ...]) -> BilinearWeights:
    array = _constant_array(value, shape)
    return BilinearWeights(memory, shape, constant=dok_ndarray.fromarray(array))


def _flatten_constant_refs(value, shape: Tuple[int, ...], add_constant):
    flat = _constant_array(value, shape).reshape(-1, order="C")
    if flat.size == 1:
        return add_constant(flat[0])
    return [add_constant(v) for v in flat]


def _append_bilinear_value(
    constant: dok_ndarray,
    linear: dok_ndarray,
    quadratic: dok_ndarray,
    start: int,
    bw: BilinearWeights,
):
    for key, value in bw.constant.keys.items():
        row = np.ravel_multi_index(key, bw.shape, order="C")
        target = (start + row,)
        constant[target] = constant[target] + value
    for key, value in bw.linear.keys.items():
        row = np.ravel_multi_index(key[:-1], bw.shape, order="C")
        target = (start + row, key[-1])
        linear[target] = linear[target] + value
    for key, value in bw.quadratic.keys.items():
        row = np.ravel_multi_index(key[:-2], bw.shape, order="C")
        target = (start + row, key[-2], key[-1])
        quadratic[target] = quadratic[target] + value


def _build_opaque_operand(value, spec: MemorySpec | None, shape: Tuple[int, ...]):
    if spec is not None:
        return WorkspaceOperand(spec, shape)
    if isinstance(value, np.ndarray):
        stored = dok_ndarray.fromarray(np.reshape(value, shape, order="C"))
        return ConstantOperand(stored, shape)
    if scipy.sparse.issparse(value):
        stored = dok_ndarray.fromarray(np.reshape(value.toarray(), shape, order="C"))
        return ConstantOperand(stored, shape)
    return ConstantOperand(value, shape)


def create_opgraph(function: Function):
    tape = function.tape
    numpy_backend = get_backend_by_name("numpy", set_current=False)

    input_layer = InputLayer()
    node_values = {}
    node_specs: Dict[int, MemorySpec] = {}

    for i in tape.input_indicies:
        input_idx = input_layer.add_input(tape.dim[i])
        spec, _ = input_layer.input_specs[input_idx]
        node_specs[i] = spec

    current_memory = MemorySpec(location=0, count=input_layer.dimension)
    for i in tape.input_indicies:
        node_values[i] = BilinearWeights.project(
            current_memory, node_specs[i], _node_shape(tape.dim[i])
        )

    layers = []
    current_size = input_layer.dimension
    pending_bilinear: List[int] = []

    def extend_node_values(new_memory: MemorySpec):
        for node_idx, value in list(node_values.items()):
            if isinstance(value, BilinearWeights) and value.memory != new_memory:
                node_values[node_idx] = value.extend_memory(new_memory)

    def queue_bilinear(idx: int, value: BilinearWeights):
        node_values[idx] = value
        if idx not in pending_bilinear and idx not in node_specs:
            pending_bilinear.append(idx)

    def flush_bilinear():
        nonlocal current_memory, current_size
        if not pending_bilinear:
            return

        old_memory = current_memory
        additions = []
        next_location = current_size
        for idx in pending_bilinear:
            bw = node_values[idx]
            shape = _node_shape(tape.dim[idx])
            spec = MemorySpec(next_location, tape.dim[idx].flat())
            additions.append((idx, bw, shape, spec))
            next_location += spec.count

        new_memory = MemorySpec(0, next_location)
        constant = dok_ndarray((next_location,))
        linear = dok_ndarray((next_location, old_memory.count))
        quadratic = dok_ndarray((next_location, old_memory.count, old_memory.count))
        for i in range(current_size):
            linear[(i, i)] = 1
        for _, bw, _, spec in additions:
            _append_bilinear_value(constant, linear, quadratic, spec.location, bw)

        weights = BilinearWeights(
            old_memory,
            (next_location,),
            constant=constant,
            linear=linear,
            quadratic=quadratic,
        )
        layers.append(BilinearWorkspaceLayer(old_memory, new_memory, weights))

        current_memory = new_memory
        current_size = next_location
        extend_node_values(new_memory)
        for idx, _, shape, spec in additions:
            node_specs[idx] = spec
            node_values[idx] = BilinearWeights.project(new_memory, spec, shape)
        pending_bilinear.clear()

    def lower_generic(idx: int, op, args):
        nonlocal current_memory, current_size
        flush_bilinear()

        output_shape = _node_shape(tape.dim[idx])
        output_count = tape.dim[idx].flat()
        output_spec = MemorySpec(current_size, output_count)
        next_memory = MemorySpec(0, current_size + output_count)

        layer_ops = [
            (IDENTITY_OP, i, UNUSED_REF, UNUSED_REF) for i in range(current_size)
        ]
        constants: Dict[int, float] = {}
        next_constant = -1

        def add_constant(value):
            nonlocal next_constant
            ref = next_constant
            constants[ref] = float(value)
            next_constant -= 1
            return ref

        def refs_for_arg(arg):
            arg_shape = _node_shape(tape.dim[arg.index])
            if arg.index in node_specs:
                spec = node_specs[arg.index]
                if spec.count == 1:
                    return spec.location
                return [spec.location + i for i in range(spec.count)]
            return _flatten_constant_refs(
                node_values[arg.index], arg_shape, add_constant
            )

        def row_ref(refs, row: int):
            return refs if isinstance(refs, int) else refs[row]

        scalar_lowered = False
        if isinstance(op, ReshapeOP):
            (arg,) = args
            refs = refs_for_arg(arg)
            layer_ops.extend(
                (IDENTITY_OP, row_ref(refs, row), UNUSED_REF, UNUSED_REF)
                for row in range(output_count)
            )
            scalar_lowered = True
        elif isinstance(op, ConcatenateOP) and op.axis == 0 and all(
            tape.dim[arg.index].is_scalar() or tape.dim[arg.index].is_vector()
            for arg in args
        ):
            concatenated_refs = []
            for arg in args:
                refs = refs_for_arg(arg)
                if isinstance(refs, int):
                    concatenated_refs.append(refs)
                else:
                    concatenated_refs.extend(refs)
            layer_ops.extend(
                (IDENTITY_OP, ref, UNUSED_REF, UNUSED_REF)
                for ref in concatenated_refs
            )
            scalar_lowered = True
        elif op in {
            OP.SIN,
            OP.COS,
            OP.TAN,
            OP.EXP,
            OP.SQRT,
            OP.LOG,
            OP.NEG,
            OP.ABS,
        }:
            (arg,) = args
            refs = refs_for_arg(arg)
            layer_ops.extend(
                (op, row_ref(refs, row), UNUSED_REF, UNUSED_REF)
                for row in range(output_count)
            )
            scalar_lowered = True
        elif op in {
            OP.ADD,
            OP.SUB,
            OP.MUL,
            OP.DIV,
            OP.PWR,
            OP.INT_PWR,
            OP.ARCTAN2,
            OP.EQUAL,
            OP.LESS_THAN,
            OP.LESS_EQUAL,
        }:
            lhs_refs = refs_for_arg(args[0])
            rhs_refs = refs_for_arg(args[1])
            layer_ops.extend(
                (
                    op,
                    row_ref(lhs_refs, row),
                    row_ref(rhs_refs, row),
                    UNUSED_REF,
                )
                for row in range(output_count)
            )
            scalar_lowered = True
        elif op == OP.CASE:
            cond_refs = refs_for_arg(args[0])
            true_refs = refs_for_arg(args[1])
            false_refs = refs_for_arg(args[2])
            layer_ops.extend(
                (
                    op,
                    row_ref(cond_refs, 0),
                    row_ref(true_refs, row),
                    row_ref(false_refs, row),
                )
                for row in range(output_count)
            )
            scalar_lowered = True

        opaque_programs: List[OpaqueProgram] = []
        if not scalar_lowered:
            operand_specs = []
            for arg in args:
                spec = node_specs.get(arg.index)
                operand_specs.append(
                    _build_opaque_operand(
                        node_values[arg.index], spec, _node_shape(tape.dim[arg.index])
                    )
                )
            opaque_programs.append(
                OpaqueProgram(
                    output_spec.location, output_shape, op, tuple(operand_specs)
                )
            )
            layer_ops.extend(
                (OPAQUE_OP, 0, row, UNUSED_REF) for row in range(output_count)
            )

        layer = GenericVectorLayer(
            current_memory,
            next_memory,
            layer_ops,
            constants=constants,
            opaque_programs=opaque_programs,
        )
        layers.append(layer)
        current_memory = next_memory
        current_size = next_memory.count
        extend_node_values(next_memory)
        node_specs[idx] = output_spec
        node_values[idx] = BilinearWeights.project(next_memory, output_spec, output_shape)

    for idx in range(len(tape)):
        if idx in tape.input_indicies:
            continue

        op, *args = tape.nodes[idx]
        if op == OP.VALUE:
            (constant_value,) = args
            node_values[idx] = _as_numpy_value(constant_value)
            continue

        operands = [node_values[a.index] for a in args]
        if all(not isinstance(operand, BilinearWeights) for operand in operands):
            node_values[idx] = numpy_backend.call(op, *operands)
            continue

        if op in ops:
            memories = {
                id(operand.memory)
                for operand in operands
                if isinstance(operand, BilinearWeights)
            }
            if len(memories) <= 1:
                try:
                    result = ops[op](*operands)
                    if isinstance(result, BilinearWeights):
                        queue_bilinear(idx, result)
                        continue
                except (AssertionError, TypeError, NotImplementedError):
                    pass

        if isinstance(op, ReshapeOP):
            (arg_value,) = operands
            if isinstance(arg_value, BilinearWeights):
                queue_bilinear(idx, arg_value.reshape(op.newshape, order=op.order))
                continue
            node_values[idx] = np.reshape(arg_value, op.newshape, order=op.order)
            continue

        lower_generic(idx, op, args)

    for output in function.output:
        if output is None:
            continue
        if output.index in node_specs:
            continue
        value = node_values[output.index]
        if isinstance(value, BilinearWeights):
            queue_bilinear(output.index, value)
        else:
            queue_bilinear(
                output.index,
                _constant_to_bw(current_memory, value, _node_shape(output.dim)),
            )

    flush_bilinear()

    output_layer = OutputLayer()
    for output in function.output:
        output_layer.add_output(node_specs[output.index], output.dim)

    return SparseNet(current_memory, input_layer, output_layer, layers)
