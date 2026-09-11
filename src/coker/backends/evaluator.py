from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    ResultBundleDimension,
)
from coker.algebra.kernel import (
    CallableReference,
    Function,
    OP,
    SymbolicCallable,
    Tape,
    Tracer,
)
from coker.algebra.ops import EvaluateOP, Operator, normalize_evaluate_result
from coker.backends.backend import Backend

NodeDimension = Dimension | FunctionSpace | ResultBundleDimension

_SYMBOLIC_CALLABLE_TYPES = SymbolicCallable | CallableReference
_SYMBOLIC_TYPES = Tracer | _SYMBOLIC_CALLABLE_TYPES


def _normalize_evaluate_result(
    op: OP | Operator,
    args: Sequence[Any],
    value: Any,
    dimension: NodeDimension,
) -> Any:
    if isinstance(op, EvaluateOP) and isinstance(args[0], CallableReference):
        return normalize_evaluate_result(value, dimension)
    return value


# ---------------------------------------------------------------------------
# Compiled execution plan
# ---------------------------------------------------------------------------


class _PlanStep(NamedTuple):
    fn: Callable[..., Any]
    arg_indices: list[int]
    out_idx: int
    dim: NodeDimension
    post_fn: Callable[[Any], Any]


class CompiledPlan:
    """Pre-compiled execution plan for a (tape, backend) pair.

    Built once via _build_plan; subsequent calls skip per-node isinstance
    dispatch and dict lookups by working from pre-resolved callables and
    workspace indices. Not thread-safe — workspace is mutated in place.
    """

    def __init__(
        self,
        steps: Sequence[_PlanStep],
        workspace: dict[int, Any],
        input_indices: Sequence[int],
    ) -> None:
        self._steps = steps
        self._workspace = workspace  # constants pre-filled; reused each call
        self._input_indices = input_indices

    def execute(
        self, inputs: Sequence[Any], backend: Backend
    ) -> dict[int, Any]:
        ws = self._workspace
        for ws_idx, arg in zip(self._input_indices, inputs):
            if ws_idx >= 0:
                ws[ws_idx] = (
                    arg
                    if isinstance(arg, _SYMBOLIC_CALLABLE_TYPES)
                    else backend.to_backend_array(arg)
                )

        for step in self._steps:
            ws[step.out_idx] = step.post_fn(
                step.fn(*[ws[i] for i in step.arg_indices])
            )

        return ws


def _build_plan(graph: Tape, backend: Backend) -> CompiledPlan:
    """Walk the tape once and return a CompiledPlan."""

    # Pass 1 — mark nodes that depend on inputs (dynamic) versus
    # pure constants.
    is_dynamic = {}
    for i, node in enumerate(graph.nodes):
        if isinstance(node, Tracer):
            # Bare Tracer in nodes list means this is an input node.
            is_dynamic[i] = True
        else:
            op, *args = node
            is_dynamic[i] = any(
                isinstance(a, Tracer)
                and a.tape is graph
                and is_dynamic.get(a.index, False)
                for a in args
            )

    # Pass 2 — pre-evaluate constant nodes into the workspace.
    # Negative slots below -(len+10) are reserved for any inline constants
    # (cross-tape Tracers or bare values) that appear as node arguments.
    workspace: dict[int, Any] = {-1: None}
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
            elif isinstance(a, _SYMBOLIC_CALLABLE_TYPES):
                resolved.append(a)
            else:
                resolved.append(backend.to_backend_array(a))
        value = resolved[0] if op == OP.VALUE else backend.call(op, *resolved)
        value = _normalize_evaluate_result(op, resolved, value, graph.dim[i])
        if not isinstance(value, _SYMBOLIC_TYPES) and not isinstance(
            graph.dim[i], ResultBundleDimension
        ):
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
            elif isinstance(a, _SYMBOLIC_TYPES):
                arg_indices.append(alloc_inline(a))
            else:
                arg_indices.append(alloc_inline(backend.to_backend_array(a)))
        dim = graph.dim[i]
        fn = backend.resolve_fn(op)
        if isinstance(op, EvaluateOP):

            def fn(*values, _fn=fn, _op=op, _dim=dim):
                value = _fn(*values)
                return _normalize_evaluate_result(_op, values, value, _dim)

        post_fn = (
            (lambda value: value)
            if isinstance(dim, ResultBundleDimension)
            else backend.resolve_post_fn(dim)
        )
        steps.append(
            _PlanStep(
                fn,
                arg_indices,
                i,
                dim,
                post_fn,
            )
        )

    return CompiledPlan(steps, workspace, graph.input_indicies)


def _cast_outputs(
    outputs: Sequence[Tracer | None],
    graph: Tape,
    workspace: dict[int, Any],
    backend: Backend,
) -> list[Any | None]:
    """Extract and reshape outputs from the workspace after plan execution."""
    result: list[Any | None] = []
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


def evaluate_inner(
    graph: Tape,
    args: Sequence[Any],
    outputs: Sequence[Tracer | None],
    backend: Backend,
    workspace: dict[int, Any],
) -> list[Any | None]:
    workspace[-1] = None
    for index, arg in zip(graph.input_indicies, args):
        if isinstance(arg, _SYMBOLIC_CALLABLE_TYPES):
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
            elif isinstance(node, _SYMBOLIC_CALLABLE_TYPES):
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

        value = _normalize_evaluate_result(op, args, value, graph.dim[w])
        workspace[w] = (
            value
            if isinstance(value, _SYMBOLIC_TYPES)
            or isinstance(graph.dim[w], ResultBundleDimension)
            else backend.reshape(value, graph.dim[w])
        )

    return _cast_outputs(outputs, graph, workspace, backend)


def evaluate(
    function: Function,
    args: Sequence[Any],
    backend: str | None = None,
) -> list[Any | None]:

    from coker.backends import get_backend_by_name, get_current_backend

    backend_impl: Backend = (
        get_current_backend()
        if backend is None
        else get_backend_by_name(backend)
    )
    return backend_impl.evaluate(function, args)
