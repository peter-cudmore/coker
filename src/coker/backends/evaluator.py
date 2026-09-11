from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    ResultBundleDimension,
)
from coker.interfaces import SymbolicCallable
from coker.algebra.graph import CallableReference, Tape, Tracer
from coker.algebra.ops import OP, Operator, normalize_evaluate_result

if TYPE_CHECKING:
    from coker.algebra.function import Function
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
    if op == OP.EVALUATE and isinstance(args[0], CallableReference):
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

    Built once by an :class:`Evaluator`; subsequent calls skip per-node
    isinstance dispatch and dictionary lookups by working from pre-resolved
    callables and workspace indices. Not thread-safe — workspace is mutated
    in place.
    """

    def __init__(
        self,
        steps: Sequence[_PlanStep],
        workspace: dict[int, Any],
        input_indices: Sequence[int],
        to_backend_array: Callable[[Any], Any],
    ) -> None:
        self._steps = steps
        self._workspace = workspace  # constants pre-filled; reused each call
        self._input_indices = input_indices
        self._to_backend_array = to_backend_array

    def execute(self, inputs: Sequence[Any]) -> dict[int, Any]:
        ws = self._workspace
        for ws_idx, arg in zip(self._input_indices, inputs):
            if ws_idx >= 0:
                ws[ws_idx] = (
                    arg
                    if isinstance(arg, _SYMBOLIC_CALLABLE_TYPES)
                    else self._to_backend_array(arg)
                )

        for step in self._steps:
            ws[step.out_idx] = step.post_fn(
                step.fn(*[ws[i] for i in step.arg_indices])
            )

        return ws


class Evaluator(ABC):
    """Backend-specific compiler for reusable tape execution plans."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    @abstractmethod
    def build_plan(self, graph: Tape) -> CompiledPlan:
        """Compile ``graph`` into a reusable execution plan."""


class GenericEvaluator(Evaluator):
    """Compiler configurable with optional native operation tables."""

    def __init__(
        self,
        backend: Backend,
        *,
        operations: Mapping[object, Callable[..., Any]] | None = None,
        parameterised_operations: (
            Mapping[type, Callable[..., Any]] | None
        ) = None,
        preserve_nonscalar_shapes: bool = False,
    ) -> None:
        super().__init__(backend)
        self._operations = operations
        self._parameterised_operations = parameterised_operations
        self._preserve_nonscalar_shapes = preserve_nonscalar_shapes

    def _resolve_operation(self, op) -> Callable[..., Any]:
        if self._operations is not None and op in self._operations:
            return self._operations[op]
        if (
            self._parameterised_operations is not None
            and type(op) in self._parameterised_operations
        ):
            operation_type = type(op)
            return lambda *args: self._parameterised_operations[
                operation_type
            ](op, *args)
        call = self.backend.call
        return lambda *args: call(op, *args)

    def _resolve_post(self, dim: NodeDimension) -> Callable[[Any], Any]:
        if self._preserve_nonscalar_shapes and not dim.is_scalar():
            return lambda value: value
        reshape = self.backend.reshape

        def post(value):
            if not isinstance(value, Tracer):
                return reshape(value, dim)
            return value

        return post

    def build_plan(self, graph: Tape) -> CompiledPlan:
        """Walk the tape once and return a compiled execution plan."""
        backend = self.backend

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
        # Negative slots below -(len+10) are reserved for inline constants
        # (cross-tape Tracers or bare values) used as node arguments.
        workspace: dict[int, Any] = {-1: None}
        next_slot = [-(len(graph.nodes) + 10)]

        def alloc_inline(value):
            slot = next_slot[0]
            next_slot[0] -= 1
            workspace[slot] = value
            return slot

        for i, node in enumerate(graph.nodes):
            if is_dynamic.get(i, True) or isinstance(node, Tracer):
                continue
            op, *args = node
            resolved = []
            for arg in args:
                if isinstance(arg, Tracer) and arg.tape is graph:
                    resolved.append(workspace[arg.index])
                elif isinstance(arg, Tracer):
                    resolved.append(arg)
                elif isinstance(arg, _SYMBOLIC_CALLABLE_TYPES):
                    resolved.append(arg)
                else:
                    resolved.append(backend.to_backend_array(arg))
            value = (
                resolved[0]
                if op in {OP.VALUE, OP.FUNCTION_VALUE}
                else backend.call(op, *resolved)
            )
            value = _normalize_evaluate_result(
                op, resolved, value, graph.dim[i]
            )
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
            for arg in args:
                if isinstance(arg, Tracer) and arg.tape is graph:
                    arg_indices.append(arg.index)
                elif isinstance(arg, _SYMBOLIC_TYPES):
                    arg_indices.append(alloc_inline(arg))
                else:
                    arg_indices.append(
                        alloc_inline(backend.to_backend_array(arg))
                    )
            dim = graph.dim[i]
            operation_fn = self._resolve_operation(op)
            if op == OP.EVALUATE:

                def evaluate_fn(
                    *values,
                    _operation_fn=operation_fn,
                    _op=op,
                    _dim=dim,
                ):
                    value = _operation_fn(*values)
                    return _normalize_evaluate_result(_op, values, value, _dim)

                step_fn = evaluate_fn
            else:
                step_fn = operation_fn
            post_fn = (
                (lambda value: value)
                if isinstance(dim, ResultBundleDimension)
                else self._resolve_post(dim)
            )
            steps.append(
                _PlanStep(
                    step_fn,
                    arg_indices,
                    i,
                    dim,
                    post_fn,
                )
            )

        return CompiledPlan(
            steps,
            workspace,
            graph.input_indicies,
            backend.to_backend_array,
        )


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
        if op in {OP.VALUE, OP.FUNCTION_VALUE}:
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
