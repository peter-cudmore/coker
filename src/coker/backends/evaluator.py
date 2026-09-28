from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from itertools import count
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    ResultBundleDimension,
)
from coker.interfaces import SymbolicCallable
from coker.algebra.graph import FunctionSymbol, Tape, Tracer
from coker.algebra.ops import Noop, OP, normalize_evaluate_result

if TYPE_CHECKING:
    from coker.algebra.function import Function
    from coker.backends.backend import Backend

NodeDimension = Dimension | FunctionSpace | ResultBundleDimension

_SYMBOLIC_CALLABLE_TYPES = SymbolicCallable | FunctionSymbol
_SYMBOLIC_TYPES = Tracer | _SYMBOLIC_CALLABLE_TYPES


def _is_symbolic_value(value: Any) -> bool:
    return isinstance(value, _SYMBOLIC_TYPES) or callable(value)


class _NativeCallable:
    """Invoke an imported callable and validate its declared result."""

    __slots__ = ("_target", "_result_dimension")

    def __init__(self, target: Callable[..., Any], result_dimension) -> None:
        self._target = target
        self._result_dimension = result_dimension

    def __call__(self, *arguments: Any) -> Any:
        return normalize_evaluate_result(
            self._target(*arguments),
            self._result_dimension,
        )


class _BoundNativeCallable:
    """Callable with one positional argument bound at a public position."""

    __slots__ = ("_target", "_argument", "_position")

    def __init__(
        self, target: Callable[..., Any], argument: Any, position: int
    ) -> None:
        self._target = target
        self._argument = argument
        self._position = position

    def __call__(self, *arguments: Any) -> Any:
        position = self._position
        return self._target(
            *arguments[:position],
            self._argument,
            *arguments[position:],
        )


def bind_callable(
    callable_value: Callable[..., Any], argument: Any, position: int
) -> Callable[..., Any]:
    """Return ``callable_value`` with one argument bound at ``position``."""
    return _BoundNativeCallable(callable_value, argument, position)


class FunctionSymbolResolver:
    """Resolve immutable function symbols to backend-native callables."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._lowered_function_targets: dict[Function, Any] = {}

    def resolve(self, symbol: FunctionSymbol) -> Callable[..., Any]:
        from coker.algebra.function import Function

        target = symbol.target
        if not isinstance(target, Function):
            return _NativeCallable(target, symbol.result_dimension)

        try:
            lowered = self._lowered_function_targets[target]
        except KeyError:
            lowered = self._backend.lower(target)
            self._lowered_function_targets[target] = lowered

        input_spaces = tuple(
            input_spec.space for input_spec in target.signature.inputs
        )
        output_indices = tuple(
            index
            for index, output_spec in enumerate(target.signature.outputs)
            if output_spec.shape is not None
        )
        input_count = sum(
            input_space is not None and not isinstance(input_space, Noop)
            for input_space in input_spaces
        )

        def invoke(*arguments):
            if len(arguments) != input_count:
                raise TypeError(
                    f"Expected {input_count} present inputs, got "
                    f"{len(arguments)}"
                )
            supplied_index = 0
            target_inputs = []
            for input_space in input_spaces:
                if input_space is None:
                    target_inputs.append(None)
                elif isinstance(input_space, Noop):
                    target_inputs.append(Noop())
                else:
                    target_inputs.append(arguments[supplied_index])
                    supplied_index += 1

            outputs = lowered.execute(target_inputs)
            if len(output_indices) == 1:
                return outputs[output_indices[0]]
            return tuple(outputs[index] for index in output_indices)

        return invoke


def _identity(value: Any) -> Any:
    return value


def _resolve_special_operation(
    op, function_symbols: FunctionSymbolResolver
) -> Callable[..., Any] | None:
    if op == OP.VALUE:
        return _identity
    if op == OP.FUNCTION:
        return function_symbols.resolve
    if op == OP.BIND:
        return bind_callable
    return None


# ---------------------------------------------------------------------------
# Compiled execution plan
# ---------------------------------------------------------------------------


class _PlanStep(NamedTuple):
    fn: Callable[..., Any]
    arg_indices: list[int]
    out_idx: int
    scalar_dimension: NodeDimension | None


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
        reshape: Callable[[Any, Dimension], Any],
    ) -> None:
        self._steps = steps
        self._workspace = workspace  # constants pre-filled; reused each call
        self._input_indices = input_indices
        self._to_backend_array = to_backend_array
        self._reshape = reshape

    def execute(self, inputs: Sequence[Any]) -> dict[int, Any]:
        ws = self._workspace
        for ws_idx, arg in zip(self._input_indices, inputs):
            if ws_idx >= 0:
                ws[ws_idx] = (
                    arg
                    if isinstance(arg, _SYMBOLIC_CALLABLE_TYPES)
                    or callable(arg)
                    else self._to_backend_array(arg)
                )

        for step in self._steps:
            value = step.fn(*[ws[i] for i in step.arg_indices])
            if step.scalar_dimension is not None and not _is_symbolic_value(
                value
            ):
                value = self._reshape(value, step.scalar_dimension)
            ws[step.out_idx] = value

        return ws


class Evaluator(ABC):
    """Backend-specific compiler for reusable tape execution plans."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    @abstractmethod
    def build_plan(self, graph: Tape) -> CompiledPlan:
        """Compile ``graph`` into a reusable execution plan."""

    def evaluate(
        self, function: Function, inputs: Sequence[Any]
    ) -> list[Any | None]:
        workspace = self.build_plan(function.tape).execute(inputs)
        return _cast_outputs(
            function.output, function.tape, workspace, self.backend
        )


class GenericEvaluator(Evaluator):
    """Compiler resolving operations from backend-native tables."""

    def __init__(
        self,
        backend: Backend,
        *,
        operations: Mapping[object, Callable[..., Any]],
        parameterised_operations: Mapping[type, Callable[..., Any]],
    ) -> None:
        super().__init__(backend)
        self._operations = operations
        self._parameterised_operations = parameterised_operations
        self._function_symbols = FunctionSymbolResolver(backend)

    def _resolve_operation(self, op) -> Callable[..., Any]:
        special_operation = _resolve_special_operation(
            op, self._function_symbols
        )
        if special_operation is not None:
            return special_operation
        try:
            return self._operations[op]
        except KeyError:
            pass
        try:
            operation = self._parameterised_operations[type(op)]
        except KeyError as ex:
            raise NotImplementedError(f"{op} is not implemented") from ex
        return partial(operation, op)

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
        inline_slots = count(start=-(len(graph.nodes) + 10), step=-1)

        for i, node in enumerate(graph.nodes):
            if is_dynamic.get(i, True) or isinstance(node, Tracer):
                continue
            op, *args = node
            resolved = []
            for argument_index, arg in enumerate(args):
                if op == OP.BIND and argument_index == 2:
                    resolved.append(arg)
                elif isinstance(arg, Tracer) and arg.tape is graph:
                    resolved.append(workspace[arg.index])
                elif isinstance(arg, Tracer):
                    resolved.append(arg)
                elif isinstance(arg, _SYMBOLIC_CALLABLE_TYPES):
                    resolved.append(arg)
                else:
                    resolved.append(backend.to_backend_array(arg))
            value = self._resolve_operation(op)(*resolved)
            if (
                op not in {OP.FUNCTION, OP.BIND}
                and not _is_symbolic_value(value)
                and not isinstance(graph.dim[i], ResultBundleDimension)
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
            for argument_index, arg in enumerate(args):
                if op == OP.BIND and argument_index == 2:
                    value = arg
                elif isinstance(arg, Tracer) and arg.tape is graph:
                    arg_indices.append(arg.index)
                    continue
                elif isinstance(arg, _SYMBOLIC_TYPES):
                    value = arg
                else:
                    value = backend.to_backend_array(arg)
                slot = next(inline_slots)
                workspace[slot] = value
                arg_indices.append(slot)
            dim = graph.dim[i]
            step_fn = self._resolve_operation(op)
            scalar_dimension = (
                dim
                if op not in {OP.FUNCTION, OP.BIND}
                and not isinstance(dim, ResultBundleDimension)
                and dim.is_scalar()
                else None
            )
            steps.append(
                _PlanStep(
                    step_fn,
                    arg_indices,
                    i,
                    scalar_dimension,
                )
            )

        return CompiledPlan(
            tuple(steps),
            workspace,
            graph.input_indicies,
            backend.to_backend_array,
            backend.reshape,
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
# Interpreter retained for optimiser callbacks with ad hoc workspaces
# ---------------------------------------------------------------------------


def evaluate_inner(
    graph: Tape,
    args: Sequence[Any],
    outputs: Sequence[Tracer | None],
    backend: Backend,
    workspace: dict[int, Any],
) -> list[Any | None]:
    workspace[-1] = None
    function_symbols = FunctionSymbolResolver(backend)
    for index, arg in zip(graph.input_indicies, args):
        if isinstance(arg, _SYMBOLIC_CALLABLE_TYPES) or callable(arg):
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
            elif isinstance(node, _SYMBOLIC_CALLABLE_TYPES) or callable(node):
                return node

            return backend.to_backend_array(node)
        except Exception as ex:
            ex.add_note(f"Node index: {node.index}")
            raise ex from ex

    for w in work_list:
        op, *nodes = graph.nodes[w]

        args = [
            (
                node
                if op == OP.BIND and argument_index == 2
                else cast_node(node)
            )
            for argument_index, node in enumerate(nodes)
        ]
        special_operation = _resolve_special_operation(op, function_symbols)
        if special_operation is not None:
            value = special_operation(*args)
        else:
            try:
                value = backend.call(op, *args)
            except Exception as ex:
                ex.add_note(f"Node index: {w}")
                ex.add_note(f"Node: {op}({args})")
                raise ex from ex

        workspace[w] = (
            value
            if op in {OP.FUNCTION, OP.BIND}
            or _is_symbolic_value(value)
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
