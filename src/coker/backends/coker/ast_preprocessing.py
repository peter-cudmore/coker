
from collections import defaultdict
from typing import Set, Dict, Tuple, List


from coker import OP, Kernel, Tracer
from .sparse_tensor import dok_ndarray, scalar
from coker.backends.backend import get_backend_by_name
import numpy as np
from coker.backends.coker.layers import InputLayer, OutputLayer, GenericLayerOP
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.weights import BilinearWeights


def label_sinks(kernel: Kernel) -> Tuple[Set[int], Set[int]]:
    """
    Summary:

        Forward pass through the graph, determining which nodes are
        to be considered "sources"

        Criteria is either
            a) nonlinear
            b) used as inputs to multiple different nonlinear terms

    """
    tape = kernel.tape
    constants = set()
    tape_outdegree = [0] * len(tape)

    sink_nodes = {o.index for o in kernel.output}
    # output of these nodes are \considered 'new variables'

    for i, node in enumerate(tape.nodes):

        if isinstance(node, Tracer):
            sink_nodes.add(i)
            continue

        op, *args = node
        if op == OP.VALUE:
            constants.add(i)
            continue

        indices = [a.index for a in args]
        in_nodes = [idx for idx in indices if idx not in constants]

        if not in_nodes:
            constants.add(i)
            continue

        # non-constant op
        #
        for j in in_nodes:
            tape_outdegree[j] += 1

        # Strictly Linear nodes
        if op in {op.ADD, OP.SUB, OP.NEG}:
            continue

        # Multi-linear terms that mayne nonlinear
        if op in {OP.MUL, OP.CROSS, OP.MATMUL, OP.DOT} and len(in_nodes) == 1:
            continue

        sink_nodes.add(i)

    for i, degree in enumerate(tape_outdegree):
        if degree >= 2:
            sink_nodes.add(i)

    return sink_nodes, constants


def label_layers(kernel: Kernel, sink_nodes: Dict):
    edges = defaultdict(set)
    tape = kernel.tape
    distance = [0] * len(tape)

    def recurse_node(sink, n):
        if n in tape.input_indicies:
            edges[sink].add(n)
            return
        op, *args = tape.nodes[n]

        if op == OP.VALUE:
            edges[n].add(sink)
            return {}

        for a in args:
            idx = a.index
            edges[n] |= {sink}
            if idx in sink_nodes:
                distance[idx] = max(distance[idx], 1 + distance[n])
                edges[sink].add(a.index)
            else:
                distance[idx] = max(distance[idx], distance[n])
                recurse_node(sink, a.index)

    for o in reversed(list(sink_nodes)):
        recurse_node(o, o)

    edges.update({i: {i} for i in tape.input_indicies})
    max_layers = max(distance)
    distance = [max_layers - d for d in distance]

    return edges, distance


def label_sources(kernel: Kernel, sink_nodes=None, constants=None) -> Dict[int, Set[int]]:
    """

    Starting with the inputs and sink nodes, label all downstream nodes that depend on those sinks

    we then end up with

    """
    if sink_nodes is None or constants is None:
        sink_nodes, constants = label_sinks(kernel)

    arguments = {i: set() for i in constants}
    arguments.update({i:{i} for i in sink_nodes})
    arguments.update({i:{i} for i in kernel.tape.input_indicies})
    workset = [i for i in range(len(kernel.tape)) if i not in arguments]

    for idx in workset:
        _, *args = kernel.tape.nodes[idx]

        arguments[idx] = set.union(*(arguments[a.index] for a in args))

    return arguments


ops = {
    OP.MUL: lambda x, y: x * y,
    OP.ADD: lambda x, y: x + y,
    OP.SUB: lambda x, y: x - y,
    OP.MATMUL: lambda x, y: x @ y,
}


def create_opgraph(kernel: Kernel):

    sinks, constants = label_sinks(kernel)
    edges, distance = label_layers(kernel, sinks)

    sources = label_sources(kernel, sinks, constants)
    tape = kernel.tape

    # label edges
#    for

    actual_edges = {
        i: edges[i] | v for i, v in sources.items()
        if v and i not in sinks
    }

    assert set(actual_edges.keys()) | constants | set(tape.input_indicies) | set(sinks) == set(range(len(kernel.tape)))

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
    memory = {
        0: MemorySpec(location=0, count=input_layer.dimension)
    }

    # contains the projections from the memory to the argument space
    # for each set of weights. In the case of the input layer, this is a map from the flattened memory
    # of the input domain, to the specific arguments/shapes in used in the consequtive layers.
    weights = {
        i: BilinearWeights(linear=input_layer.get_projection(i), memory=memory[0]) for i in kernel.tape.input_indicies
    }

    def eval_numeric(op, *args):
        backend = get_backend_by_name('numpy')
        return backend.call(op, *args)

    def get_recursive(arg: Tracer):
        if arg.index in sinks:
            return weights[arg.index]

        if arg.is_constant():
            return arg.value()

        op, *args = arg.value()
        args = [get_recursive(arg) for arg in args]

        if any(isinstance(arg, BilinearWeights) for arg in args):
            return ops[op](*args)
        return eval_numeric(op, *args)

    stack = sorted([s for s in sinks if s not in tape.input_indicies], key=lambda i: distance[i], reverse=True)

    while stack:
        sink = stack.pop()
        count = tape.dim[sink].flat()
        layer_idx = distance[sink]
        memory[sink] = MemorySpec(location=layer_idx, count=count)
        weights[sink] = BilinearWeights(linear=dok_ndarray.eye(count), memory=memory[sink])

        op, *args = tape.nodes[sink]

        args = [
            get_recursive(a) for a in args
        ]
        assert not isinstance(op, BilinearWeights)
        layers.append(GenericLayerOP(memory[sink], op, *args))

    output_layer = OutputLayer()
    for output in kernel.output:
        idx = output.index
        dim = output.dim
        output_layer.add_output(memory[idx], dim)

    return SparseNet(list(memory.values()), input_layer, output_layer, layers)




class SparseNet:
    def __init__(self, memory, input_layer: InputLayer, output_layer: OutputLayer, intermediate_layers):
        self.memory = memory
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.intermediate_layers = intermediate_layers

    @property
    def layers(self):
        return [self.input_layer, *self.intermediate_layers, self.output_layer]

    def __call__(self, *args):
        workspace = self.apply_input_map(*args)

        for layer in self.intermediate_layers:
            in_specs = layer.inputs()
            out_spec = layer.output
            out = layer(*[workspace[k] for k in in_specs])
            workspace[out_spec] = out
        return self.output_layer.call(workspace)

    def push_forward(self, *tangent_spaces):
        n_args = len(self.input_layer.vec_to_arg_maps)
        x, dx = tangent_spaces[0:n_args], tangent_spaces[n_args:]
        workspace = self.apply_input_map(*x)
        dworkspace = self.apply_input_map(*dx)

        for layer in self.intermediate_layers:
            in_specs = layer.inputs()
            out_spec = layer.output
            x_i = [workspace[k] for k in in_specs]
            dx_i = [dworkspace[k] for k in in_specs]
            out, dout = layer.push_forward(*x_i, *dx_i)
            workspace[out_spec] = out
            dworkspace[out_spec] = dout

        y = self.output_layer.call(workspace)
        dy = self.output_layer.call(dworkspace)
        return y, dy

    def apply_input_map(self, *args) -> Dict[MemorySpec, np.ndarray]:
        return {
            self.memory[0]: self.input_layer(*args)
        }

    def apply_output_map(self, context):
        return self.output_layer.call(context)




