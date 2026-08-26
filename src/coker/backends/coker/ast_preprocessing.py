from collections import defaultdict
from typing import Set, Dict, Tuple

from coker.algebra.kernel import Function, Tracer
import numpy as np
from coker.backends.coker.layers import InputLayer, OutputLayer
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.residual import (
    BilinearStage,
    CallStage,
    InputMap,
    NonlinearStage,
    OutputMap,
    apply_bilinear_stage,
    apply_call_stage,
    apply_nonlinear_stage,
    push_forward_bilinear_stage,
    push_forward_call_stage,
    push_forward_nonlinear_stage,
)
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
    """A lowered Coker program with either legacy or stable-slot stages."""

    def __init__(
        self,
        memory: MemorySpec | int,
        input_layer: InputLayer | InputMap,
        output_layer: OutputLayer | OutputMap,
        intermediate_layers=None,
        *,
        residual_stages: (
            Tuple[BilinearStage | NonlinearStage | CallStage, ...] | None
        ) = None,
    ):
        if intermediate_layers is not None and residual_stages is not None:
            raise ValueError("SparseNet cannot mix legacy and residual stages")
        if residual_stages is None:
            if not isinstance(memory, MemorySpec):
                raise TypeError("legacy SparseNet requires a MemorySpec")
            if not isinstance(input_layer, InputLayer) or not isinstance(
                output_layer, OutputLayer
            ):
                raise TypeError("legacy SparseNet requires legacy input/output maps")
        else:
            if not isinstance(memory, int) or memory < 0:
                raise TypeError("residual SparseNet requires a workspace size")
            if not isinstance(input_layer, InputMap) or not isinstance(
                output_layer, OutputMap
            ):
                raise TypeError("residual SparseNet requires residual input/output maps")
        self.memory = memory
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.intermediate_layers = (
            [] if intermediate_layers is None else list(intermediate_layers)
        )
        self.residual_stages = residual_stages

    @property
    def layers(self):
        stages = (
            self.intermediate_layers
            if self.residual_stages is None
            else self.residual_stages
        )
        return [self.input_layer, *stages, self.output_layer]

    def _residual_workspace(self, *args) -> np.ndarray:
        workspace = np.zeros(self.memory, dtype=float)
        self.input_layer.write_into(args, workspace)
        return workspace

    def __call__(self, *args):
        if self.residual_stages is not None:
            workspace = self._residual_workspace(*args)
            for stage in self.residual_stages:
                if isinstance(stage, BilinearStage):
                    apply_bilinear_stage(stage, workspace)
                elif isinstance(stage, NonlinearStage):
                    apply_nonlinear_stage(stage, workspace)
                else:
                    apply_call_stage(stage, workspace)
            return self.output_layer.read(workspace)

        workspace = self.apply_input_map(*args)
        if workspace.size < self.memory.count:
            workspace = np.pad(
                workspace, (0, self.memory.count - workspace.size)
            )
        for layer in self.intermediate_layers:
            workspace = layer(workspace)
        return self.output_layer.call(workspace)

    def push_forward(self, *tangent_spaces):
        if self.residual_stages is not None:
            n_args = len(self.input_layer.bindings)
            x, dx = tangent_spaces[0:n_args], tangent_spaces[n_args:]
            if len(dx) != n_args:
                raise ValueError("residual push_forward requires one tangent per input")
            workspace = self._residual_workspace(*x)
            dworkspace = self._residual_workspace(*dx)
            for stage in self.residual_stages:
                if isinstance(stage, BilinearStage):
                    push_forward_bilinear_stage(stage, workspace, dworkspace)
                elif isinstance(stage, NonlinearStage):
                    push_forward_nonlinear_stage(stage, workspace, dworkspace)
                else:
                    push_forward_call_stage(stage, workspace, dworkspace)
            return self.output_layer.read(workspace), self.output_layer.read(
                dworkspace
            )

        n_args = len(self.input_layer.input_specs)
        x, dx = tangent_spaces[0:n_args], tangent_spaces[n_args:]
        workspace = self.apply_input_map(*x)
        dworkspace = self.apply_input_map(*dx)
        if workspace.size < self.memory.count:
            workspace = np.pad(
                workspace, (0, self.memory.count - workspace.size)
            )
        if dworkspace.size < self.memory.count:
            dworkspace = np.pad(
                dworkspace, (0, self.memory.count - dworkspace.size)
            )

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
        if self.residual_stages is not None:
            raise RuntimeError(
                "residual SparseNet requires the typed bytecode builder"
            )
        return {
            "workspace": self.memory.to_export_dict(),
            "input_layer": self.input_layer.to_export_dict(),
            "output_layer": self.output_layer.to_export_dict(),
            "intermediate_layers": [
                layer.to_export_dict() for layer in self.intermediate_layers
            ],
        }


class FunctionTable:
    """Own the SparseNet programs that make up one lowered Coker module."""

    def __init__(self, functions: list[SparseNet], entry_function_id: int = 0):
        self._functions = tuple(functions)
        self.entry_function_id = entry_function_id
        if not self._functions:
            raise ValueError("function table requires an entry program")
        if not 0 <= entry_function_id < len(self._functions):
            raise ValueError("entry function id is outside the function table")

    @property
    def entry(self) -> SparseNet:
        """Return the program invoked at the module entry point."""
        return self._functions[self.entry_function_id]

    @property
    def functions(self) -> tuple[SparseNet, ...]:
        """Return the owned programs in stable function-id order."""
        return self._functions

    def export_payload(self):
        """Export the complete module while retaining table-level ownership."""
        return {
            "functions": [
                {
                    "function_id": function_id,
                    "program": graph.export_program_payload(),
                }
                for function_id, graph in enumerate(self._functions)
            ]
        }
