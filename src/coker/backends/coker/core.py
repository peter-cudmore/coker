from typing import Tuple, Type, List, Dict
from coker import Tracer, Function, OP
from coker.algebra.dimensions import FunctionSpace
from coker.backends.backend import Backend, ArrayLike, get_backend_by_name
from coker.backends.coker.sparse_tensor import is_constant
from coker.backends.coker.ast_preprocessing import (
    SparseNet,
    label_layers,
    label_sinks,
    label_sources,
)
from coker.backends.coker.ast_rewriting import rewrite_graph
from coker.backends.coker.layers import (
    MemorySpec,
    InputLayer,
    OutputLayer,
    GenericLayerOP,
    IdentityLayer,
)
from coker.backends.coker.weights import BilinearWeights
from coker.backends.coker.op_impl import ops
import numpy as np


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
        backend = self

        def compiled(inputs):
            return backend.evaluate(function, inputs)

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

    sinks, constants = label_sinks(function)
    edges, distance = label_layers(function, sinks)

    sources = label_sources(function, sinks, constants)
    tape = function.tape

    # label edges
    #    for

    actual_edges = {
        i: edges[i] | v for i, v in sources.items() if v and i not in sinks
    }

    assert set(actual_edges.keys()) | constants | set(
        tape.input_indicies
    ) | set(sinks) == set(range(len(function.tape)))

    # for each input, we want to create a map from the argument into a general input stack
    # so that the input map takes (x_1, x_2, x_3, x_4) -> X
    # and then each element of nodes is a linear map from X -> x_i  (i.e a projection)

    # for each 'sink' we append onto the X vector the dimension of the sink output
    # and so that we have X = f_i(W(X,X)) for each nonlinearity

    # for each 'output' we then have a set of projections that map from X -> y_0, y_1, ...
    #
    weights = {}

    input_layer = InputLayer()
    for i in tape.input_indicies:
        input_layer.add_input(tape.dim[i])

    layers = []

    # contains the memory spec for the 'outputs' of each layer, including the input layer
    memory = {0: MemorySpec(location=0, count=input_layer.dimension)}

    # contains the projections from the memory to the argument space
    # for each set of weights. In the case of the input layer, this is a map from the flattened memory
    # of the input domain, to the specific arguments/shapes in used in the consequtive layers.
    weights = {
        i: BilinearWeights(
            linear=input_layer.get_projection(i),
            memory=memory[0],
            shape=tape.dim[i].shape,
        )
        for i in function.tape.input_indicies
    }

    def eval_numeric(shape, op, *args):
        backend = get_backend_by_name("numpy", set_current=False)
        result = backend.call(op, *args)

        return backend.reshape(result, shape)

    bridge_cache = {}
    bridge_counter = [-1]  # negative locations for bridge sinks

    def get_recursive(arg: Tracer):
        if arg.index in sinks:
            return weights[arg.index]

        if arg.index in bridge_cache:
            return bridge_cache[arg.index]

        if arg.is_constant():
            return arg.value()

        op, *args = arg.value()

        args = [get_recursive(arg) for arg in args]

        bw_args = [a for a in args if isinstance(a, BilinearWeights)]
        if bw_args:
            memories = {id(a.memory) for a in bw_args}
            if len(memories) > 1 or op not in ops:
                # Multiple memory sources or unsupported op: create an intermediate
                # bridge layer evaluated at runtime by the numpy backend.
                count = tape.dim[arg.index].flat()
                bridge_mem = MemorySpec(
                    location=bridge_counter[0], count=count
                )
                bridge_counter[0] -= count
                layers.append(GenericLayerOP(bridge_mem, op, *args))
                shape = tape.dim[arg.index].shape or (1,)
                bw = BilinearWeights.reshape_identity(
                    memory=bridge_mem, shape=shape
                )
                bridge_cache[arg.index] = bw
                return bw
            result = ops[op](*args)
            return result
        v = eval_numeric(arg.shape, op, *args)
        return v

    stack = sorted(
        [s for s in sinks if s not in tape.input_indicies],
        key=lambda i: distance[i],
        reverse=True,
    )
    next_location = [
        input_layer.dimension
    ]  # global counter, starts after input
    while stack:
        sink = stack.pop()
        count = tape.dim[sink].flat()
        location = next_location[0]
        next_location[0] += count
        memory[sink] = MemorySpec(location=location, count=count)
        dim = tape.dim[sink]

        weights[sink] = BilinearWeights.identity2(memory=memory[sink])

        op, *args = tape.nodes[sink]
        args = [get_recursive(a) for a in args]
        has_constants = any(is_constant(a) for a in args)
        if (
            op
            in {
                OP.ADD,
                OP.SUB,
                OP.MATMUL,
                OP.NEG,
                OP.DOT,
                OP.CROSS,
            }
            and has_constants
        ):
            w = ops[op](*args)

            layers.append(IdentityLayer(memory[sink], w))
        # possible componentwise operations:
        elif op is OP.MUL and has_constants:
            if any(isinstance(a, (int, float)) for a in args):
                w = ops[op](*args)
                layers.append(IdentityLayer(memory[sink], w))
            else:
                first, second = args
                if isinstance(first, BilinearWeights):
                    diag_b = diag(second)
                    a = first
                else:
                    diag_b = diag(first)
                    a = second
                w = ops[OP.MATMUL](diag_b, a)
                layers.append(IdentityLayer(memory[sink], w))

        elif op is OP.DIV and has_constants:
            if any(isinstance(a, (int, float)) for a in args):
                w = ops[op](*args)
                layers.append(IdentityLayer(memory[sink], w))
            else:
                a, b = args
                diag_b = diag(1 / b)
                w = ops[OP.MATMUL](diag_b, a)
                layers.append(IdentityLayer(memory[sink], w))
        else:
            assert not isinstance(op, BilinearWeights)
            assert not isinstance(op, GenericLayerOP)
            try:

                layers.append(GenericLayerOP(memory[sink], op, *args))
            except AssertionError as ex:
                op, *args = tape.nodes[sink]
                args = [get_recursive(a) for a in args]
                raise ex

    output_layer = OutputLayer()
    for output in function.output:
        idx = output.index
        dim = output.dim
        output_layer.add_output(memory[idx], dim)

    return SparseNet(list(memory.values()), input_layer, output_layer, layers)


class BilinearMatrixGroup:
    def __init__(self, memory: MemorySpec, shape: Tuple[int, ...]):
        self.shape = shape
        self.memory = memory

    def push_forwards(self, x, dx):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def diag(vector: np.ndarray) -> np.ndarray:
    assert (
        len(vector.shape) == 1
        or len(vector.shape) == 2
        and vector.shape[-1] == 1
    )
    return np.diag(vector)
