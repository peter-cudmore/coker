from typing import List, NamedTuple, Callable
import numpy as np

import coker
from coker.backends.backend import ArrayLike, Backend
from coker.algebra.kernel import Tracer, OP


# ---------------------------------------------------------------------------
# Compiled execution plan
# ---------------------------------------------------------------------------

class _PlanStep(NamedTuple):
    fn: Callable
    arg_indices: list
    out_idx: int
    dim: object
    post_fn: Callable  # pre-resolved reshape (or identity) applied after fn


class CompiledPlan:
    """Pre-compiled execution plan for a (tape, backend) pair.

    Built once via _build_plan; subsequent calls skip per-node isinstance
    dispatch and dict lookups by working from pre-resolved callables and
    workspace indices. Not thread-safe — workspace is mutated in place.
    """

    def __init__(self, steps, workspace, input_indices):
        self._steps = steps
        self._workspace = workspace   # constants pre-filled; reused each call
        self._input_indices = input_indices

    def execute(self, inputs, backend):
        ws = self._workspace
        ws[-1] = None

        for ws_idx, arg in zip(self._input_indices, inputs):
            if ws_idx >= 0:
                ws[ws_idx] = arg if isinstance(arg, coker.Function) else backend.to_backend_array(arg)

        for step in self._steps:
            ws[step.out_idx] = step.post_fn(step.fn(*[ws[i] for i in step.arg_indices]))

        return ws


def _build_plan(graph, backend):
    """Walk the tape once and return a CompiledPlan."""

    # Pass 1 — mark which nodes depend on inputs (dynamic) vs are pure constants.
    is_dynamic = {}
    for i, node in enumerate(graph.nodes):
        if isinstance(node, Tracer):
            # Bare Tracer in nodes list means this is an input node.
            is_dynamic[i] = True
        else:
            op, *args = node
            is_dynamic[i] = any(
                isinstance(a, Tracer) and a.tape is graph and is_dynamic.get(a.index, False)
                for a in args
            )

    # Pass 2 — pre-evaluate constant nodes into the workspace.
    # Negative slots below -(len+10) are reserved for any inline constants
    # (cross-tape Tracers or bare values) that appear as node arguments.
    workspace = {-1: None}
    next_slot = [-(len(graph.nodes) + 10)]

    def alloc_inline(value):
        s = next_slot[0]
        next_slot[0] -= 1
        workspace[s] = value
        return s

    for i, node in enumerate(graph.nodes):
        if is_dynamic.get(i, True) or isinstance(node, Tracer):
            continue
        op, *args = node
        resolved = []
        for a in args:
            if isinstance(a, Tracer) and a.tape is graph:
                resolved.append(workspace[a.index])
            elif isinstance(a, Tracer):
                resolved.append(a)  # cross-tape: pass through as-is
            else:
                resolved.append(backend.to_backend_array(a))
        value = resolved[0] if op == OP.VALUE else backend.call(op, *resolved)
        if not isinstance(value, Tracer):
            value = backend.reshape(value, graph.dim[i])
        workspace[i] = value

    # Pass 3 — build execution steps for dynamic non-input nodes only.
    steps = []
    for i, node in enumerate(graph.nodes):
        if not is_dynamic.get(i, False) or isinstance(node, Tracer):
            continue
        op, *args = node
        arg_indices = []
        for a in args:
            if isinstance(a, Tracer) and a.tape is graph:
                arg_indices.append(a.index)
            elif isinstance(a, Tracer):
                arg_indices.append(alloc_inline(a))
            else:
                arg_indices.append(alloc_inline(backend.to_backend_array(a)))
        dim = graph.dim[i]
        steps.append(_PlanStep(backend.resolve_fn(op), arg_indices, i, dim, backend.resolve_post_fn(dim)))

    return CompiledPlan(steps, workspace, graph.input_indicies)


def _cast_outputs(outputs, graph, workspace, backend):
    """Extract and reshape outputs from the workspace after plan execution."""
    result = []
    for o in outputs:
        if o is None:
            result.append(None)
            continue
        if o.tape is not graph:
            result.append(o)
            continue
        output = workspace[o.index]
        if isinstance(output, Tracer):
            result.append(output)
            continue
        if not o.dim.is_scalar():
            try:
                output = backend.to_numpy_array(output)
                if isinstance(output, np.ndarray):
                    result.append(np.reshape(output, o.shape))
                    continue
            except ValueError:
                pass
            backend.reshape(output, o.dim)
            result.append(output)
            continue
        result.append(backend.to_numpy_array(output))
    return result


# ---------------------------------------------------------------------------
# Original interpreted evaluator (kept for sympy backend and optimisation)
# ---------------------------------------------------------------------------

def evaluate_inner(graph, args, outputs, backend: Backend, workspace: dict):
    workspace[-1] = None
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
                ex.add_note(f"Node: {op}({args})")
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
        output = workspace[o.index]
        if isinstance(output, Tracer):
            return output
        if not o.dim.is_scalar():
            try:
                output = backend.to_numpy_array(output)
                if isinstance(output, np.ndarray):
                    return np.reshape(output, shape=o.shape)
            except ValueError:
                pass
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
