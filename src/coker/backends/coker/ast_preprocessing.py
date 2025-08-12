from collections import defaultdict
from typing import Set, Dict, Tuple

from coker import Function
import numpy as np
from coker.backends.coker.layers import InputLayer, OutputLayer
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.op_impl import *


def label_sinks(function: Function) -> Tuple[Set[int], Set[int]]:
    """
    Summary:

        Forward pass through the graph, determining which nodes are
        to be considered "sources"

        Criteria is either
            a) nonlinear
            b) used as inputs to multiple different nonlinear terms

    """
    tape = function.tape
    constants = set()
    tape_outdegree = [0] * len(tape)
    sources = {}
    sink_nodes = {o.index for o in function.output}
    # output of these nodes are \considered 'new variables'

    for i, node in enumerate(tape.nodes):

        if i in tape.input_indicies:
            sources[i] = [-1]
            sink_nodes.add(i)
            continue
        else:
            sources[i] = []

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

        # Strictly Linear nodes
        if op.is_linear():
            for j in in_nodes:
                for source in sources[j]:
                    if source not in sources[i]:
                        sources[i].append(source)
            continue

        for j in in_nodes:
            sources[i] += sources[j]

        # Multi-linear terms that mayne nonlinear
        if op.is_bilinear():
            if len(set(sources[i])) == 1:
                continue

        if op == OP.DIV and indices[1] in constants:
            continue

        sink_nodes.add(i)

    for i, degree in enumerate(tape_outdegree):
        if degree >= 2:
            sink_nodes.add(i)

    return sink_nodes, constants


def label_layers(function: Function, sink_nodes: Dict):
    edges = defaultdict(set)
    tape = function.tape
    distance = [0] * len(tape)

    def recurse_node(sink, node):
        if node in tape.input_indicies:
            edges[sink].add(node)
            return
        op, *args = tape.nodes[node]

        if op == OP.VALUE:
            edges[node].add(sink)
            return

        for a in args:
            idx = a.index
            edges[node] |= {sink}
            if idx in sink_nodes:
                distance[idx] = max(distance[idx], 1 + distance[node])
                edges[sink].add(a.index)
            else:
                distance[idx] = max(distance[idx], distance[node])
                recurse_node(sink, a.index)

    for o in reversed(list(sink_nodes)):
        recurse_node(o, o)

    edges.update({i: {i} for i in tape.input_indicies})
    max_layers = max(distance)
    distance = [max_layers - d for d in distance]

    return edges, distance


def label_sources(
    function: Function, sink_nodes=None, constants=None
) -> Dict[int, Set[int]]:
    """

    Starting with the inputs and sink nodes, label all downstream nodes that depend on those sinks

    we then end up with

    """
    if sink_nodes is None or constants is None:
        sink_nodes, constants = label_sinks(function)

    arguments = {i: set() for i in constants}
    arguments.update({i: {i} for i in sink_nodes})
    arguments.update({i: {i} for i in function.tape.input_indicies})
    workset = [i for i in range(len(function.tape)) if i not in arguments]

    for idx in workset:
        _, *args = function.tape.nodes[idx]

        arguments[idx] = set.union(*(arguments[a.index] for a in args))

    return arguments


class SparseNet:
    def __init__(
        self,
        memory,
        input_layer: InputLayer,
        output_layer: OutputLayer,
        intermediate_layers,
    ):
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
            (out_spec,) = layer.outputs()
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
            (out_spec,) = layer.outputs()
            x_i = [workspace[k] for k in in_specs]
            dx_i = [dworkspace[k] for k in in_specs]
            out, dout = layer.push_forward(*x_i, *dx_i)
            workspace[out_spec] = out
            dworkspace[out_spec] = dout

        y = self.output_layer.call(workspace)
        dy = self.output_layer.call(dworkspace)
        return y, dy

    def apply_input_map(self, *args) -> Dict[MemorySpec, np.ndarray]:
        return {self.memory[0]: self.input_layer(*args)}

    def apply_output_map(self, context):
        return self.output_layer.call(context)
