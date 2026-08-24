from typing import Dict, List, Tuple, Type

import numpy as np
import scipy.sparse

from coker.algebra.dimensions import FunctionSpace
from coker.algebra.kernel import Function, Tracer
from coker.algebra.ops import ConcatenateOP, OP, ReshapeOP, ModuleCallOP
from coker.backends.backend import ArrayLike, Backend, get_backend_by_name
from coker.backends.coker.module import CokerModule
from coker.backends.coker.optimisation import (
    build_optimisation_problem as build_qp_optimisation_problem,
)
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


class CokerFunction:
    """A Coker-lowered function graph and its backend-specific operations."""

    def __init__(self, function: Function):
        self.function = function
        self._graph = None

    @property
    def graph(self) -> SparseNet:
        """Build the Coker graph only when compilation needs it."""
        if self._graph is None:
            self._graph = create_opgraph(self.function)
        return self._graph

    @property
    def function_id(self) -> int:
        """Return the graph's stable function-table identifier."""
        return self.graph.function_id

    def __call__(self, inputs):
        """Evaluate through the reference interpreter on the Python host."""
        numpy_backend = get_backend_by_name("numpy", set_current=False)
        return numpy_backend.evaluate(self.function, inputs)

    def compile_bytecode(self) -> bytes:
        """Compile this lowered graph into a mapped Coker bytecode module."""
        from coker.backends.coker.runtime import CompiledGraph

        return bytes(CompiledGraph.compile(self.graph).program)

    def compile_artifact(
        self, *, name: str = "coker_function", version: str = "1"
    ):
        """Compile this lowered Coker function into a mapped artifact."""
        from coker.backends.coker.artifacts import _compile_artifact

        return _compile_artifact(self, name=name, version=version)

    def export_payload(self) -> dict[str, object]:
        """Return the deterministic graph payload consumed by the compiler."""
        return self.graph.export_payload()


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
        if (
            any(
                isinstance(function.tape.nodes[index][0], ModuleCallOP)
                for index in range(len(function.tape.nodes))
                if index not in function.tape.input_indicies
            )
            or any(
                isinstance(function.tape.dim[input_index], FunctionSpace)
                for input_index in function.tape.input_indicies
            )
            or any(output is None for output in function.output)
        ):
            numpy_backend = get_backend_by_name("numpy", set_current=False)
            return numpy_backend.lower(function)

        return CokerFunction(function)

    def build_optimisation_problem(
        self,
        cost: Tracer,
        constraints: List[Tracer],
        parameters: List[Tracer],
        outputs: List[Tracer],
        initial_conditions: Dict[int, ArrayLike],
    ):
        return build_qp_optimisation_problem(
            self,
            cost,
            constraints,
            parameters,
            outputs,
            initial_conditions,
        )

    def make_optimisation_module(self, implementation):
        """Wrap a prebuilt QP solver for numerical module composition."""
        return CokerModule(implementation)


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


def create_unfused_opgraph(function: Function):
    """Lower a graph without materialising a whole-function bilinear tensor."""
    return _FunctionTableBuilder().build(function)


def create_opgraph(function: Function):
    compact = create_compact_bilinear_opgraph(function)
    if compact is not None:
        return compact
    return create_unfused_opgraph(function)


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


def _create_opgraph(
    function: Function,
    function_table_builder: _FunctionTableBuilder,
    function_id: int,
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
            if operation == OP.MATMUL:
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
        function_id=function_id,
    )
    # Kept as compile-time metadata for QP coefficient extraction. Runtime
    # graphs do not inspect this attribute.
    graph.output_weights = output_weights
    raw_weights = _raw_output_weights(graph, output_weights)
    graph.output_weights_are_raw = raw_weights is not None
    if raw_weights is not None:
        graph.output_weights = raw_weights
    return graph
