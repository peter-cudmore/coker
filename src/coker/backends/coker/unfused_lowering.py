"""Ordinary Coker tape lowering into scheduled graph layers."""

from __future__ import annotations

from typing import Dict, List, Tuple


from coker.algebra.kernel import Function, Tracer
from coker.algebra.ops import ConcatenateOP, OP, ReshapeOP
from coker.backends.backend import get_backend_by_name
from coker.backends.coker.ast_preprocessing import FunctionTable, SparseNet
from coker.backends.coker.layers import (
    IDENTITY_OP,
    OPAQUE_OP,
    UNUSED_REF,
    BilinearWorkspaceLayer,
    FunctionEvaluationLayer,
    GenericVectorLayer,
    InputLayer,
    OpaqueProgram,
    OutputLayer,
)
from coker.backends.coker.lowering import (
    _append_bilinear_value,
    _as_numpy_value,
    _build_opaque_operand,
    _constant_extension_weights,
    _concatenate_bilinear_operands,
    _constant_to_bw,
    _flatten_constant_rows,
    _node_shape,
    _raw_output_weights,
)
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.op_impl import ops
from coker.backends.coker.sparse_tensor import dok_ndarray
from coker.backends.coker.weights import BilinearWeights


class _FunctionTableBuilder:
    def __init__(self):
        self._function_ids_by_identity: Dict[int, int] = {}
        self._graphs_by_id: Dict[int, SparseNet | None] = {}

    def build(self, function: Function) -> FunctionTable:
        entry_function_id, _graph = self.get_or_build(function)
        ordered_graphs = [
            self._graphs_by_id[function_id]
            for function_id in range(len(self._graphs_by_id))
        ]
        assert all(graph_item is not None for graph_item in ordered_graphs)
        return FunctionTable(list(ordered_graphs), entry_function_id)

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
        graph = _create_opgraph(function, self)
        self._graphs_by_id[function_id] = graph
        return function_id, graph


def _create_opgraph(
    function: Function,
    function_table_builder: _FunctionTableBuilder,
):
    tape = function.tape
    numpy_backend = get_backend_by_name("numpy", set_current=False)
    required_nodes = {
        output.index for output in function.output if output is not None
    }
    pending_nodes = list(required_nodes)
    while pending_nodes:
        node_index = pending_nodes.pop()
        if node_index in tape.input_indicies:
            continue
        _operation, *arguments = tape.nodes[node_index]
        for argument in arguments:
            if not isinstance(argument, Tracer):
                continue
            if argument.index not in required_nodes:
                required_nodes.add(argument.index)
                pending_nodes.append(argument.index)

    remaining_uses: Dict[int, int] = {}
    for node_index in required_nodes:
        if node_index in tape.input_indicies:
            continue
        _operation, *arguments = tape.nodes[node_index]
        for argument in arguments:
            if isinstance(argument, Tracer):
                remaining_uses[argument.index] = (
                    remaining_uses.get(argument.index, 0) + 1
                )
    for output in function.output:
        if output is not None:
            remaining_uses[output.index] = (
                remaining_uses.get(output.index, 0) + 1
            )
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
        retained = []
        next_location = 0
        for node_index, previous_spec in node_specs.items():
            shape = _node_shape(tape.dim[node_index])
            next_spec = MemorySpec(next_location, previous_spec.count)
            retained.append((node_index, previous_spec, shape, next_spec))
            next_location += next_spec.count

        additions = []
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
        for _node_index, previous_spec, _shape, next_spec in retained:
            for offset in range(previous_spec.count):
                linear[
                    (
                        next_spec.location + offset,
                        previous_spec.location + offset,
                    )
                ] = 1
        for _node_index, bilinear_value, output_shape, spec in additions:
            _append_bilinear_value(
                constant,
                linear,
                quadratic,
                spec.location,
                bilinear_value,
                output_shape,
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
        node_specs.clear()
        for node_index, _previous_spec, shape, spec in retained:
            node_specs[node_index] = spec
            node_values[node_index] = BilinearWeights.project(
                new_memory, spec, shape
            )
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

        argument_counts: Dict[int, int] = {}
        for argument in arguments:
            argument_counts[argument.index] = (
                argument_counts.get(argument.index, 0) + 1
            )
        retained = []
        next_location = 0
        for retained_index, previous_spec in node_specs.items():
            uses_after = remaining_uses.get(retained_index, 0) - (
                argument_counts.get(retained_index, 0)
            )
            if uses_after <= 0:
                continue
            shape = _node_shape(tape.dim[retained_index])
            next_spec = MemorySpec(next_location, previous_spec.count)
            retained.append((retained_index, previous_spec, shape, next_spec))
            next_location += next_spec.count

        output_spec = MemorySpec(next_location, output_count)
        next_memory = MemorySpec(0, next_location + output_count)
        layer_operations = []
        for _retained_index, previous_spec, _shape, next_spec in retained:
            layer_operations.extend(
                (
                    IDENTITY_OP,
                    previous_spec.location + offset,
                    UNUSED_REF,
                    UNUSED_REF,
                )
                for offset in range(next_spec.count)
            )
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
        node_specs.clear()
        for retained_index, _previous_spec, shape, spec in retained:
            node_specs[retained_index] = spec
            node_values[retained_index] = BilinearWeights.project(
                next_memory, spec, shape
            )
        node_specs[node_index] = output_spec
        node_values[node_index] = BilinearWeights.project(
            next_memory, output_spec, output_shape
        )

    def release_arguments(arguments):
        for argument in arguments:
            if not isinstance(argument, Tracer):
                continue
            remaining_uses[argument.index] -= 1
            if remaining_uses[argument.index] != 0:
                continue
            node_values.pop(argument.index, None)
            node_specs.pop(argument.index, None)
            if argument.index in pending_bilinear_nodes:
                pending_bilinear_nodes.remove(argument.index)

    for node_index in range(len(tape)):
        if node_index in tape.input_indicies:
            continue

        if node_index not in required_nodes:
            continue
        operation, *arguments = tape.nodes[node_index]
        try:
            if operation == OP.VALUE:
                (constant_value,) = arguments
                node_values[node_index] = _as_numpy_value(constant_value)
                continue

            operands = [node_values[argument.index] for argument in arguments]
            if all(
                not isinstance(operand, BilinearWeights)
                for operand in operands
            ):
                node_values[node_index] = numpy_backend.call(
                    operation, *operands
                )
                continue

            if operation == OP.EVALUATE and isinstance(
                node_values[arguments[0].index], Function
            ):
                lower_function_evaluation(node_index, arguments)
                continue
            if operation == OP.DOT and any(
                isinstance(operand, BilinearWeights)
                and not operand.is_linear
                and not operand.is_constant
                for operand in operands
            ):
                # A dot product is quadratic in its current workspace
                # coordinates. Materialise an earlier quadratic result before
                # forming another dot so the product stays degree two.
                flush_bilinear()
                operands = [
                    node_values[argument.index] for argument in arguments
                ]
            elif operation == OP.MATMUL:
                flush_bilinear()
                operands = [
                    node_values[argument.index] for argument in arguments
                ]

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
                    except (
                        AssertionError,
                        TypeError,
                        NotImplementedError,
                    ):
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
                                bilinear_memory,
                                operands,
                                axis=operation.axis,
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
                            operation.newshape,
                            order=operation.order,
                        ),
                    )
                    continue

            lower_generic(node_index, operation, arguments)

        finally:
            release_arguments(arguments)

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
    output_weights = []
    for output in function.output:
        output_layer.add_output(node_specs[output.index], output.dim)
        output_weights.append(node_values.get(output.index))

    workspace_count = max(
        [
            input_layer.dimension,
            current_memory.count,
            *(
                max(layer.memory_in.count, layer.memory_out.count)
                for layer in layers
            ),
        ]
    )

    graph = SparseNet(
        MemorySpec(0, workspace_count),
        input_layer,
        output_layer,
        layers,
    )
    # Kept as compile-time metadata for QP coefficient extraction. Runtime
    # graphs do not inspect this attribute.
    graph.output_weights = output_weights
    raw_weights = _raw_output_weights(graph, output_weights)
    graph.output_weights_are_raw = raw_weights is not None
    if raw_weights is not None:
        graph.output_weights = raw_weights
    return graph


def build_function_table(function: Function) -> FunctionTable:
    """Build the ordinary lowering module and own all nested programs."""
    return _FunctionTableBuilder().build(function)


def build_unfused_opgraph(function: Function) -> SparseNet:
    """Build the ordinary graph-lowering path for compatibility callers."""
    return build_function_table(function).entry
