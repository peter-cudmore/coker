from collections import defaultdict
from typing import Set, Dict, Tuple

from coker.algebra.kernel import Function, Tracer
import numpy as np
from coker.backends.coker.layers import InputLayer, OutputLayer
from coker.backends.coker.memory import MemorySpec
from coker.algebra.ops import OP


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
    sink_nodes = {o.index for o in function.output if o is not None}
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

    def recurse_node(sink, node, depth):
        if node in tape.input_indicies:
            edges[sink].add(node)
            return
        op, *args = tape.nodes[node]

        if node is tape.NONE or node is tape.MAP_TO_NONE:
            edges[sink].add(node)
            return

        assert not isinstance(op, Tracer)
        if op == OP.VALUE:
            edges[node].add(sink)
            return

        for a in args:
            idx = a.index
            edges[node] |= {sink}
            if idx in sink_nodes:
                distance[idx] = max(distance[idx], depth + 1)
                edges[sink].add(a.index)
            else:
                recurse_node(sink, a.index, depth)

    for o in sorted(sink_nodes, reverse=True):
        recurse_node(o, o, distance[o])

    edges.update({i: {i} for i in tape.input_indicies})
    max_layers = max(distance)
    distance = [max_layers - d for d in distance]

    return edges, distance


def label_sources(
    function: Function, sink_nodes=None, constants=None
) -> Dict[int, Set[int]]:
    """

    Starting with the inputs and sink nodes, label all downstream nodes
    that depend on those sinks.

    """
    if sink_nodes is None or constants is None:
        sink_nodes, constants = label_sinks(function)

    arguments = {i: set() for i in constants}
    arguments.update({i: {i} for i in sink_nodes})
    arguments.update({i: {i} for i in function.tape.input_indicies})
    workset = [i for i in range(len(function.tape)) if i not in arguments]

    for idx in workset:
        op, *args = function.tape.nodes[idx]

        arguments[idx] = set.union(*(arguments[a.index] for a in args))

    return arguments


class SparseNet:
    def __init__(
        self,
        memory: MemorySpec,
        input_layer: InputLayer,
        output_layer: OutputLayer,
        intermediate_layers,
        function_id: int = 0,
        function_table=None,
    ):
        self.memory = memory
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.intermediate_layers = intermediate_layers
        self.function_id = function_id
        self.function_table = function_table

    @property
    def layers(self):
        return [self.input_layer, *self.intermediate_layers, self.output_layer]

    def __call__(self, *args):
        workspace = self.apply_input_map(*args)
        for layer in self.intermediate_layers:
            workspace = layer(workspace)
        return self.output_layer.call(workspace)

    def push_forward(self, *tangent_spaces):
        n_args = len(self.input_layer.input_specs)
        x, dx = tangent_spaces[0:n_args], tangent_spaces[n_args:]
        workspace = self.apply_input_map(*x)
        dworkspace = self.apply_input_map(*dx)

        for layer in self.intermediate_layers:
            workspace, dworkspace = layer.push_forward(workspace, dworkspace)

        y = self.output_layer.call(workspace)
        dy = self.output_layer.call(dworkspace)
        return y, dy

    def apply_input_map(self, *args) -> np.ndarray:
        return self.input_layer(*args)

    def apply_output_map(self, workspace):
        return self.output_layer.call(workspace)

    def export_program_payload(self):
        return {
            "workspace": self.memory.to_export_dict(),
            "input_layer": self.input_layer.to_export_dict(),
            "output_layer": self.output_layer.to_export_dict(),
            "intermediate_layers": [
                layer.to_export_dict() for layer in self.intermediate_layers
            ],
        }

    def export_payload(self):
        function_table = self.function_table or [self]
        return {
            "functions": [
                {
                    "function_id": graph.function_id,
                    "program": graph.export_program_payload(),
                }
                for graph in sorted(
                    function_table, key=lambda graph: graph.function_id
                )
            ]
        }
