from typing import List
import numpy as np
from coker.backends.backend import ArrayLike, Backend
from coker.algebra.kernel import Tracer, Tensor, OP


def evaluate_inner(graph, args, outputs, backend: Backend, workspace: dict):
    for index, arg in zip(graph.input_indicies, args):
        workspace[index] = arg

    work_list = [i for i in range(len(graph.nodes)) if i not in workspace]

    def get_node(node):

        if isinstance(node, Tracer):
            return workspace[node.index]
        return backend.from_native(node)

    for w in work_list:
        op, *nodes = graph.nodes[w]

        try:
            args = [get_node(n) for n in nodes]
            if op == OP.VALUE:
                value, = args
            else:
                value = backend.call(op, *[get_node(n) for n in nodes])

        except KeyError as ex:
            raise NotImplementedError(f"Op {op} not implemented in python")
        workspace[w] = backend.reshape(value, graph.dim[w])

    return workspace[outputs.index]


def evaluate(graph, args, outputs, backend='numpy'):
    from coker.backends import get_backend_by_name

    backend_impl: Backend = get_backend_by_name(backend)
    return evaluate_inner(graph, args, outputs, backend_impl, {})
