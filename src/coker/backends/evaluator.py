from typing import List
import numpy as np

import coker
from coker.backends.backend import ArrayLike, Backend
from coker.algebra.kernel import Tracer, OP


def evaluate_inner(graph, args, outputs, backend: Backend, workspace: dict):

    for index, arg in zip(graph.input_indicies, args):
        if isinstance(arg, coker.Function):
            workspace[index] = arg
        else:
            workspace[index] = backend.to_backend_array(arg)

    work_list = [i for i in range(len(graph.nodes)) if i not in workspace]

    def cast_node(node):
        try:
            if isinstance(node, Tracer):
                if node.tape == graph:
                    return workspace[node.index]
                else:
                    return node
            elif isinstance(node, coker.Function):
                return node

            return backend.to_backend_array(node)
        except Exception as ex:
            ex.add_note(f"Node index: {node.index}")
            raise ex from ex

    for w in work_list:
        op, *nodes = graph.nodes[w]

        args = [cast_node(n) for n in nodes]
        if op == OP.VALUE:
            (value,) = args
        else:
            try:
                value = backend.call(op, *args)
            except Exception as ex:
                ex.add_note(f"Node index: {w}")
                raise ex from ex

        workspace[w] = (
            backend.reshape(value, graph.dim[w])
            if not isinstance(value, Tracer)
            else value
        )

    def cast_output(o):
        if o is None:
            return None
        if o.tape != graph:
            return o
        if not o.dim.is_scalar():
            try:
                output = backend.to_numpy_array(workspace[o.index])
                if isinstance(output, np.ndarray):
                    return np.reshape(output, shape=o.shape)
            except ValueError:
                output = workspace[o.index]
            backend.reshape(output, o.dim)
            return output
        return backend.to_numpy_array(workspace[o.index])

    outputs = [cast_output(o) for o in outputs]

    return outputs


def evaluate(function, args, backend=None):

    from coker.backends import get_backend_by_name, get_current_backend

    if not backend:
        backend_impl: Backend = get_current_backend()
    else:
        backend_impl: Backend = get_backend_by_name(backend)
    return backend_impl.evaluate(function, args)
