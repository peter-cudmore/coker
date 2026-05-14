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
        if all(output is None for output in function.output):
            return function.output

        if any(
            isinstance(function.tape.dim[input_index], FunctionSpace)
            for input_index in function.tape.input_indicies
        ):
            numpy_backend = get_backend_by_name("numpy", set_current=False)
            return numpy_backend.evaluate(function, inputs)

        graph = create_opgraph(function)
        return [graph(*inputs)]

    def reshape(self, array: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
        raise NotImplementedError

    def lower(self, function: Function):
        if any(
            isinstance(function.tape.dim[input_index], FunctionSpace)
            for input_index in function.tape.input_indicies
        ) or any(output is None for output in function.output):
            numpy_backend = get_backend_by_name("numpy", set_current=False)
            return numpy_backend.lower(function)

        graph = create_opgraph(function)

        def compiled(inputs):
            return [graph(*inputs)]

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
    array = _constant_array(value, shape)
    return BilinearWeights(
        memory, shape, constant=dok_ndarray.fromarray(array)
    )


def _flatten_constant_rows(value, shape: Tuple[int, ...]) -> List[float]:
    flat = _constant_array(value, shape).reshape(-1, order="C")
    return [float(item) for item in flat]


def _append_bilinear_value(
    constant: dok_ndarray,
    linear: dok_ndarray,
    quadratic: dok_ndarray,
    start: int,
    weights: BilinearWeights,
):
    for key, value in weights.constant.keys.items():
        row = np.ravel_multi_index(key, weights.shape, order="C")
        target = (start + row,)
        constant[target] = constant[target] + value
    for key, value in weights.linear.keys.items():
        row = np.ravel_multi_index(key[:-1], weights.shape, order="C")
        target = (start + row, key[-1])
        linear[target] = linear[target] + value
    for key, value in weights.quadratic.keys.items():
        row = np.ravel_multi_index(key[:-2], weights.shape, order="C")
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

    constant = dok_ndarray.fromarray(
        np.concatenate(
            [operand.constant.toarray() for operand in bilinear_operands],
            axis=axis,
        )
    )
    linear = dok_ndarray.fromarray(
        np.concatenate(
            [operand.linear.toarray() for operand in bilinear_operands],
            axis=axis,
        )
    )
    quadratic = dok_ndarray.fromarray(
        np.concatenate(
            [operand.quadratic.toarray() for operand in bilinear_operands],
            axis=axis,
        )
    )
    return BilinearWeights.from_trusted_dok(
        memory,
        constant.shape,
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
        stored = dok_ndarray.fromarray(
            np.reshape(value.toarray(), shape, order="C")
        )
        return ConstantOperand(stored, shape)
    return ConstantOperand(value, shape)


class _FunctionTableBuilder:
    def __init__(self):
        self._function_ids_by_identity: Dict[int, int] = {}
        self._graphs_by_id: Dict[int, SparseNet | None] = {}

    def build(self, function: Function) -> SparseNet:
        _function_id, graph = self.get_or_build(function)
        ordered_graphs = [
            self._graphs_by_id[function_id]
            for function_id in range(len(self._graphs_by_id))
        ]
        assert all(graph_item is not None for graph_item in ordered_graphs)
        function_table = list(ordered_graphs)
        for graph_item in function_table:
            graph_item.function_table = function_table
        return graph

    def get_or_build(self, function: Function) -> Tuple[int, SparseNet]:
        function_identity = id(function)
        if function_identity in self._function_ids_by_identity:
            function_id = self._function_ids_by_identity[function_identity]
            existing_graph = self._graphs_by_id[function_id]
            if existing_graph is None:
                raise NotImplementedError(
                    "Recursive function evaluation is not supported"
                )
            return function_id, existing_graph

        function_id = len(self._function_ids_by_identity)
        self._function_ids_by_identity[function_identity] = function_id
        self._graphs_by_id[function_id] = None
        graph = _create_opgraph(function, self, function_id)
        self._graphs_by_id[function_id] = graph
        return function_id, graph


def create_opgraph(function: Function):
    return _FunctionTableBuilder().build(function)


def _create_opgraph(
    function: Function,
    function_table_builder: _FunctionTableBuilder,
    function_id: int,
):
    tape = function.tape
    numpy_backend = get_backend_by_name("numpy", set_current=False)

    input_layer = InputLayer()
    node_values = {}
    node_specs: Dict[int, MemorySpec] = {}

    for input_index in tape.input_indicies:
        input_position = input_layer.add_input(tape.dim[input_index])
        spec, _shape = input_layer.input_specs[input_position]
        node_specs[input_index] = spec

    current_memory = MemorySpec(location=0, count=input_layer.dimension)
    for input_index in tape.input_indicies:
        node_values[input_index] = BilinearWeights.project(
            current_memory,
            node_specs[input_index],
            _node_shape(tape.dim[input_index]),
        )

    layers = []
    current_size = input_layer.dimension
    pending_bilinear_nodes: List[int] = []

    def extend_node_values(new_memory: MemorySpec):
        for node_index, value in list(node_values.items()):
            if (
                isinstance(value, BilinearWeights)
                and value.memory != new_memory
            ):
                node_values[node_index] = value.extend_memory(new_memory)

    def queue_bilinear(node_index: int, value: BilinearWeights):
        node_values[node_index] = value
        if (
            node_index not in pending_bilinear_nodes
            and node_index not in node_specs
        ):
            pending_bilinear_nodes.append(node_index)

    def flush_bilinear():
        nonlocal current_memory, current_size
        if not pending_bilinear_nodes:
            return

        previous_memory = current_memory
        additions = []
        next_location = current_size
        for node_index in pending_bilinear_nodes:
            bilinear_value = node_values[node_index]
            shape = _node_shape(tape.dim[node_index])
            spec = MemorySpec(next_location, tape.dim[node_index].flat())
            additions.append((node_index, bilinear_value, shape, spec))
            next_location += spec.count

        new_memory = MemorySpec(0, next_location)
        constant = dok_ndarray((next_location,))
        linear = dok_ndarray((next_location, previous_memory.count))
        quadratic = dok_ndarray(
            (next_location, previous_memory.count, previous_memory.count)
        )
        for row in range(current_size):
            linear[(row, row)] = 1
        for _node_index, bilinear_value, _shape, spec in additions:
            _append_bilinear_value(
                constant,
                linear,
                quadratic,
                spec.location,
                bilinear_value,
            )

        weights = BilinearWeights(
            previous_memory,
            (next_location,),
            constant=constant,
            linear=linear,
            quadratic=quadratic,
        )
        layers.append(
            BilinearWorkspaceLayer(previous_memory, new_memory, weights)
        )

        current_memory = new_memory
        current_size = next_location
        extend_node_values(new_memory)
        for node_index, _bilinear_value, shape, spec in additions:
            node_specs[node_index] = spec
            node_values[node_index] = BilinearWeights.project(
                new_memory, spec, shape
            )
        pending_bilinear_nodes.clear()

    def lower_function_evaluation(node_index: int, arguments):
        nonlocal current_memory, current_size
        flush_bilinear()

        function_value = node_values[arguments[0].index]
        if not isinstance(function_value, Function):
            raise NotImplementedError(
                "Function evaluation requires a statically known Function"
            )

        callee_function_id, callee_graph = function_table_builder.get_or_build(
            function_value
        )
        input_bindings = [
            _build_opaque_operand(
                node_values[argument.index],
                node_specs.get(argument.index),
                _node_shape(tape.dim[argument.index]),
            )
            for argument in arguments[1:]
        ]

        output_shape = _node_shape(tape.dim[node_index])
        output_count = tape.dim[node_index].flat()
        output_spec = MemorySpec(current_size, output_count)
        next_memory = MemorySpec(0, current_size + output_count)
        output_bindings = [output_spec]

        layer = FunctionEvaluationLayer(
            current_memory,
            next_memory,
            input_bindings,
            output_bindings,
            callee_graph,
            callee_function_id=callee_function_id,
        )
        layers.append(layer)
        current_memory = next_memory
        current_size = next_memory.count
        extend_node_values(next_memory)
        node_specs[node_index] = output_spec
        node_values[node_index] = BilinearWeights.project(
            next_memory, output_spec, output_shape
        )

    def lower_generic(node_index: int, operation, arguments):
        nonlocal current_memory, current_size
        flush_bilinear()

        output_shape = _node_shape(tape.dim[node_index])
        output_count = tape.dim[node_index].flat()
        base_size = current_size
        constant_values: List[float] = []

        def reserve_constant_rows(value, shape: Tuple[int, ...]):
            start = base_size + len(constant_values)
            rows = _flatten_constant_rows(value, shape)
            constant_values.extend(rows)
            if len(rows) == 1:
                return start
            return [start + offset for offset in range(len(rows))]

        def refs_for_arg(argument):
            argument_shape = _node_shape(tape.dim[argument.index])
            if argument.index in node_specs:
                spec = node_specs[argument.index]
                if spec.count == 1:
                    return spec.location
                return [spec.location + offset for offset in range(spec.count)]
            return reserve_constant_rows(
                node_values[argument.index], argument_shape
            )

        def row_ref(refs, row: int):
            return refs if isinstance(refs, int) else refs[row]

        appended_operations = []
        scalar_lowered = False
        if isinstance(operation, ReshapeOP):
            (argument,) = arguments
            refs = refs_for_arg(argument)
            appended_operations.extend(
                (IDENTITY_OP, row_ref(refs, row), UNUSED_REF, UNUSED_REF)
                for row in range(output_count)
            )
            scalar_lowered = True
        elif (
            isinstance(operation, ConcatenateOP)
            and operation.axis == 0
            and all(
                tape.dim[argument.index].is_scalar()
                or tape.dim[argument.index].is_vector()
                for argument in arguments
            )
        ):
            concatenated_refs = []
            for argument in arguments:
                refs = refs_for_arg(argument)
                if isinstance(refs, int):
                    concatenated_refs.append(refs)
                else:
                    concatenated_refs.extend(refs)
            appended_operations.extend(
                (IDENTITY_OP, ref, UNUSED_REF, UNUSED_REF)
                for ref in concatenated_refs
            )
            scalar_lowered = True
        elif operation in {
            OP.SIN,
            OP.COS,
            OP.TAN,
            OP.EXP,
            OP.SQRT,
            OP.LOG,
            OP.NEG,
            OP.ABS,
        }:
            (argument,) = arguments
            refs = refs_for_arg(argument)
            appended_operations.extend(
                (operation, row_ref(refs, row), UNUSED_REF, UNUSED_REF)
                for row in range(output_count)
            )
            scalar_lowered = True
        elif operation in {
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
            left_refs = refs_for_arg(arguments[0])
            right_refs = refs_for_arg(arguments[1])
            appended_operations.extend(
                (
                    operation,
                    row_ref(left_refs, row),
                    row_ref(right_refs, row),
                    UNUSED_REF,
                )
                for row in range(output_count)
            )
            scalar_lowered = True
        elif operation == OP.CASE:
            condition_refs = refs_for_arg(arguments[0])
            true_refs = refs_for_arg(arguments[1])
            false_refs = refs_for_arg(arguments[2])
            appended_operations.extend(
                (
                    operation,
                    row_ref(condition_refs, 0),
                    row_ref(true_refs, row),
                    row_ref(false_refs, row),
                )
                for row in range(output_count)
            )
            scalar_lowered = True

        if constant_values:
            constant_memory = MemorySpec(0, base_size + len(constant_values))
            layers.append(
                BilinearWorkspaceLayer(
                    current_memory,
                    constant_memory,
                    _constant_extension_weights(
                        current_memory, base_size, constant_values
                    ),
                )
            )
            current_memory = constant_memory
            current_size = constant_memory.count
            extend_node_values(constant_memory)

        output_spec = MemorySpec(current_size, output_count)
        next_memory = MemorySpec(0, current_size + output_count)
        layer_operations = [
            (IDENTITY_OP, row, UNUSED_REF, UNUSED_REF)
            for row in range(current_size)
        ]
        layer_operations.extend(appended_operations)

        opaque_programs: List[OpaqueProgram] = []
        if not scalar_lowered:
            operand_specs = []
            for argument in arguments:
                spec = node_specs.get(argument.index)
                operand_specs.append(
                    _build_opaque_operand(
                        node_values[argument.index],
                        spec,
                        _node_shape(tape.dim[argument.index]),
                    )
                )
            opaque_programs.append(
                OpaqueProgram(
                    output_spec.location,
                    output_shape,
                    operation,
                    tuple(operand_specs),
                )
            )
            layer_operations.extend(
                (OPAQUE_OP, 0, row, UNUSED_REF) for row in range(output_count)
            )

        layer = GenericVectorLayer(
            current_memory,
            next_memory,
            layer_operations,
            opaque_programs=opaque_programs,
        )
        layers.append(layer)
        current_memory = next_memory
        current_size = next_memory.count
        extend_node_values(next_memory)
        node_specs[node_index] = output_spec
        node_values[node_index] = BilinearWeights.project(
            next_memory, output_spec, output_shape
        )

    for node_index in range(len(tape)):
        if node_index in tape.input_indicies:
            continue

        operation, *arguments = tape.nodes[node_index]
        if operation == OP.VALUE:
            (constant_value,) = arguments
            node_values[node_index] = _as_numpy_value(constant_value)
            continue

        operands = [node_values[argument.index] for argument in arguments]
        if all(
            not isinstance(operand, BilinearWeights) for operand in operands
        ):
            node_values[node_index] = numpy_backend.call(operation, *operands)
            continue

        if operation == OP.EVALUATE and isinstance(
            node_values[arguments[0].index], Function
        ):
            lower_function_evaluation(node_index, arguments)
            continue

        if operation in ops:
            memories = {
                id(operand.memory)
                for operand in operands
                if isinstance(operand, BilinearWeights)
            }
            if len(memories) <= 1:
                try:
                    result = ops[operation](*operands)
                    if isinstance(result, BilinearWeights):
                        queue_bilinear(node_index, result)
                        continue
                except (AssertionError, TypeError, NotImplementedError):
                    pass

        if isinstance(operation, ConcatenateOP):
            bilinear_memory = next(
                (
                    operand.memory
                    for operand in operands
                    if isinstance(operand, BilinearWeights)
                ),
                None,
            )
            if bilinear_memory is not None:
                try:
                    queue_bilinear(
                        node_index,
                        _concatenate_bilinear_operands(
                            bilinear_memory, operands, axis=operation.axis
                        ),
                    )
                    continue
                except (AssertionError, TypeError, ValueError):
                    pass

        if isinstance(operation, ReshapeOP):
            (argument_value,) = operands
            if isinstance(argument_value, BilinearWeights):
                queue_bilinear(
                    node_index,
                    argument_value.reshape(
                        operation.newshape, order=operation.order
                    ),
                )
                continue
            node_values[node_index] = np.reshape(
                argument_value, operation.newshape, order=operation.order
            )
            continue

        lower_generic(node_index, operation, arguments)

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
                _constant_to_bw(
                    current_memory, value, _node_shape(output.dim)
                ),
            )

    flush_bilinear()

    output_layer = OutputLayer()
    for output in function.output:
        output_layer.add_output(node_specs[output.index], output.dim)

    return SparseNet(
        current_memory,
        input_layer,
        output_layer,
        layers,
        function_id=function_id,
    )
