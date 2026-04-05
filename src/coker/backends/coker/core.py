from typing import Tuple, Type, List, Dict
from coker.algebra.kernel import Tracer, Function
from coker.algebra.ops import OP
from coker.algebra.dimensions import FunctionSpace
from coker.backends.backend import Backend, ArrayLike, get_backend_by_name
from coker.backends.coker.ast_preprocessing import SparseNet
from coker.backends.coker.layers import (
    MemorySpec,
    InputLayer,
    OutputLayer,
    GenericLayerOP,
)
from coker.backends.coker.weights import BilinearWeights
from coker.backends.coker.op_impl import ops
import numpy as np
import scipy.sparse


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

        # Fall back to numpy for higher-order functions with FunctionSpace inputs.
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
        # FunctionSpace inputs and None outputs can't be lowered to a static
        # graph; fall back to the numpy plan for those cases.
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
        cost: Tracer,  # cost
        constraints: List[Tracer],
        parameters: List[Tracer],
        outputs: List[Tracer],
        initial_conditions: Dict[int, ArrayLike],
    ):
        raise NotImplementedError


def create_opgraph(function: Function):
    tape = function.tape

    input_layer = InputLayer()
    for i in tape.input_indicies:
        input_layer.add_input(tape.dim[i])

    input_memory = MemorySpec(location=0, count=input_layer.dimension)

    node_values = {
        i: BilinearWeights(
            linear=input_layer.get_projection(i),
            memory=input_memory,
            shape=tape.dim[i].shape,
        )
        for i in tape.input_indicies
    }

    layers = []
    next_location = input_layer.dimension
    output_indices = {
        output.index for output in function.output if output is not None
    }
    numpy_backend = get_backend_by_name("numpy", set_current=False)

    for idx in range(len(tape)):
        if idx in tape.input_indicies:
            continue

        op, *args = tape.nodes[idx]

        if op == OP.VALUE:
            (constant_value,) = args
            if scipy.sparse.issparse(constant_value):
                constant_value = constant_value.toarray()
            node_values[idx] = constant_value
            continue

        operands = [node_values[a.index] for a in args]
        # Collapse constant BilinearWeights to plain numpy arrays so the
        # constant-fold path below can consume them without needing a layer.
        operands = [
            (
                o.constant.toarray().reshape(o.shape)
                if isinstance(o, BilinearWeights) and o.is_constant
                else o
            )
            for o in operands
        ]
        bw_operands = [
            operand
            for operand in operands
            if isinstance(operand, BilinearWeights)
        ]

        if not bw_operands:
            node_values[idx] = numpy_backend.call(op, *operands)
            continue

        if idx not in output_indices and op in ops:
            memories = {id(operand.memory) for operand in bw_operands}
            if len(memories) == 1:
                try:
                    result = ops[op](*operands)
                    if isinstance(result, BilinearWeights):
                        node_values[idx] = result
                        continue
                except (AssertionError, TypeError, NotImplementedError):
                    pass

        count = tape.dim[idx].flat()
        mem = MemorySpec(location=next_location, count=count)
        next_location += count
        compiled_operands = [
            o.compile() if isinstance(o, BilinearWeights) else o
            for o in operands
        ]
        layers.append(GenericLayerOP(mem, op, *compiled_operands))
        shape = tape.dim[idx].shape or (1,)
        node_values[idx] = BilinearWeights.reshape_identity(
            memory=mem, shape=shape
        )

    output_layer = OutputLayer()
    for output in function.output:
        output_layer.add_output(node_values[output.index].memory, output.dim)

    return SparseNet([input_memory], input_layer, output_layer, layers)
