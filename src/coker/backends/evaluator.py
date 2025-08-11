import numpy as np

import coker
from coker.backends.backend import Backend
from coker.algebra.kernel import Tracer, OP


def evaluate_inner(graph, args, outputs, backend: Backend, workspace: dict):

    for index, arg in zip(graph.input_indicies, args):
        if isinstance(arg, coker.Function):
            workspace[index] = arg
        else:
            workspace[index] = backend.to_backend_array(arg)

    work_list = [i for i in range(len(graph.nodes)) if i not in workspace]

    def cast_node(node):

        if isinstance(node, Tracer):
            return workspace[node.index]
        elif isinstance(node, coker.Function):
            return node

        return backend.to_backend_array(node)

    for w in work_list:
        op, *nodes = graph.nodes[w]

        try:
            args = [cast_node(n) for n in nodes]
            if op == OP.VALUE:
                (value,) = args
            else:
                value = backend.call(op, *[cast_node(n) for n in nodes])

        except KeyError as ex:
            raise NotImplementedError(
                f"Op {op} not implemented in python"
            ) from ex

        workspace[w] = backend.reshape(value, graph.dim[w])

    def cast_output(o):
        if o is None:
            return None
        if not o.dim.is_scalar():
            return np.reshape(
                backend.to_numpy_array(workspace[o.index]), o.shape
            )
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
