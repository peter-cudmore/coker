"""Ordinary Coker tape lowering into scheduled graph layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from coker.algebra.kernel import Function, Tracer
from coker.algebra.ops import ConcatenateOP, OP, ReshapeOP


from coker.backends.coker.ast_preprocessing import FunctionTable, SparseNet
from coker.backends.coker.weights import BilinearWeights, compiler_binary
from coker.backends.coker.residual import (
    BilinearRow,
    BilinearStage,
    BilinearTerm,
    CallStage,
    InputBinding,
    InputMap,
    NonlinearOperation,
    NonlinearStage,
    OutputBinding,
    OutputMap,
    RetainedExpression,
    SlotOperand,
)
from coker.backends.coker.lowering_support import (
    _append_bilinear_value,
    _as_numpy_value,
    _build_opaque_operand,
    _constant_extension_weights,
    _concatenate_bilinear_operands,
    _constant_to_bw,
    _flatten_constant_rows,
    _node_shape,
    _raw_output_weights,
)
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.op_impl import ops
from coker.backends.coker.sparse_tensor import dok_ndarray


@dataclass(frozen=True)
class SlotRange:
    """Contiguous caller-workspace range assigned to one live tape value."""

    start: int
    length: int


@dataclass(frozen=True)
class ValueLifetime:
    """Compiler-only lifetime and stable workspace assignment for a tape value."""

    value: int
    width: int
    first_definition: int
    final_use: int
    slot: SlotRange
@dataclass(frozen=True)
class EpochValue:
    """Compiler-only expression and the symbolic metadata that roots it."""

    value: object
    roots: Tuple[int, ...]
    degree: int
    symbolic_dependencies: Tuple[int, ...] = ()
    earliest_consumer_wave: int | None = None


@dataclass
class _EpochFrontier:
    """Deterministic compiler state for one maximal bilinear epoch."""

    roots: Tuple[int, ...]
    values: Dict[int, EpochValue]
    uses: Dict[int, int]
    closed_reason: str | None = None

    def track(
        self,
        node_index: int,
        value: object,
        degree: int,
        symbolic_dependencies: Tuple[int, ...] = (),
        earliest_consumer_wave: int | None = None,
    ) -> None:
        """Retain expression-local roots rather than stamping the input basis.

        A dependency already in this frontier contributes its roots; a
        materialized/input predecessor is itself a root.  This makes the
        metadata valid across selective closure and keeps symbolic
        predecessors available for boundary selection.
        """
        dependencies = tuple(sorted(set(symbolic_dependencies)))
        expression_roots = {
            root
            for dependency in dependencies
            for root in (
                self.values[dependency].roots
                if dependency in self.values
                else (dependency,)
            )
        }
        if not expression_roots:
            expression_roots.update(self.roots)
        self.values[node_index] = EpochValue(
            value,
            tuple(sorted(expression_roots)),
            degree,
            dependencies,
            earliest_consumer_wave,
        )

    def close(self, reason: str) -> None:
        self.closed_reason = reason


def _stable_use_counts(tape, required_nodes, outputs):
    """Count uses in tape order, rather than relying on set iteration order."""
    uses: Dict[int, int] = {}
    for node_index in sorted(required_nodes):
        if node_index in tape.input_indicies:
            continue
        _operation, *arguments = tape.nodes[node_index]
        for argument in arguments:
            if isinstance(argument, Tracer):
                uses[argument.index] = uses.get(argument.index, 0) + 1
    for output in outputs:
        if output is not None:
            uses[output.index] = uses.get(output.index, 0) + 1
    return uses

@dataclass(frozen=True)
class _SemanticWave:
    """Immutable emitted ready antichain."""

    ordinal: int
    nodes: Tuple[int, ...]


@dataclass(frozen=True)
class _SemanticDag:
    """Required graph with direct execution and collapsed materialized edges."""

    nodes: Tuple[int, ...]
    execution_dependencies: Dict[int, Tuple[int, ...]]
    materialized_dependencies: Dict[int, Tuple[int, ...]]
    waves: Tuple[_SemanticWave, ...]
    order: Tuple[int, ...]

@dataclass(frozen=True)
class ColoredBarrierNode:
    """A compiler-only node in the collapsed generic/bilinear barrier DAG."""

    node_id: int
    color: str
    members: Tuple[int, ...]
    dependencies: Tuple[int, ...]


@dataclass(frozen=True)
class ColoredBarrierDag:
    """Collapsed barrier graph and deterministic scheduling diagnostics."""

    nodes: Tuple[ColoredBarrierNode, ...]
    critical_path: int
    schedule: Tuple[Tuple[int, ...], ...]
    color_switches: int
    lower_bound: int
    theoretically_attainable_le_50: bool
    @property
    def critical_path_lower_bound(self) -> int:
        return self.critical_path

    @property
    def scheduled_color_switch_count(self) -> int:
        return self.color_switches


def _colored_barrier_dag(semantic_dag: _SemanticDag, tape) -> ColoredBarrierDag:
    """Collapse algebraic regions and schedule the resulting colored DAG.

    This is deliberately independent of emitted layers. A bilinear component
    starts as a maximal connected component of algebraic operations; components
    that cross a generic barrier are split before projection. Generic
    operations remain individual barriers. Edges are projected from the direct
    execution DAG, retaining every barrier-to-barrier RAW edge.
    """
    algebraic = {OP.ADD, OP.SUB, OP.MUL, OP.DOT, OP.MATMUL, OP.CROSS}
    required = set(semantic_dag.nodes)
    algebraic_nodes = {
        index for index in required
        if not isinstance(tape.nodes[index], Tracer)
        and tape.nodes[index][0] in algebraic
    }
    parent = {index: index for index in algebraic_nodes}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for consumer in sorted(algebraic_nodes):
        for producer in semantic_dag.execution_dependencies.get(consumer, ()):
            if producer in algebraic_nodes:
                union(consumer, producer)

    components: Dict[int, List[int]] = {}
    for index in sorted(algebraic_nodes):
        components.setdefault(find(index), []).append(index)
    # Keep the initial connected components where possible, but never allow a
    # component to straddle a generic barrier.  Such a non-convex component
    # projects to A -> G -> A and is a cycle in the quotient.
    initial_components = [
        tuple(members)
        for members in sorted(components.values(), key=lambda values: values[0])
    ]
    generic_descriptors: List[Tuple[int, str, Tuple[int, ...]]] = []
    for index in sorted(required):
        if index in algebraic_nodes or index in tape.input_indicies:
            continue
        if isinstance(tape.nodes[index], Tracer):
            continue
        operation = tape.nodes[index][0]
        if (
            operation == OP.VALUE
            or isinstance(operation, (ReshapeOP, ConcatenateOP))
            or operation is OP.TRANSPOSE
        ):
            continue
        generic_descriptors.append((0, "generic", (index,)))

    direct = semantic_dag.execution_dependencies

    def make_nodes(
        algebraic_components: List[Tuple[int, ...]],
    ) -> Tuple[ColoredBarrierNode, ...]:
        descriptors: List[Tuple[int, str, Tuple[int, ...]]] = []
        node_for: Dict[int, int] = {}
        next_id = 0
        for members in algebraic_components:
            descriptors.append((next_id, "bilinear", members))
            for member in members:
                node_for[member] = next_id
            next_id += 1
        for _, _, (index,) in generic_descriptors:
            descriptors.append((next_id, "generic", (index,)))
            node_for[index] = next_id
            next_id += 1
        dependencies: Dict[int, Set[int]] = {
            node_id: set() for node_id, _, _ in descriptors
        }

        def barrier_sources(
            index: int, visiting: Set[int] | None = None
        ) -> Set[int]:
            if index in node_for:
                return {node_for[index]}
            if (
                index in tape.input_indicies
                or isinstance(tape.nodes[index], Tracer)
                or tape.nodes[index][0] == OP.VALUE
            ):
                return set()
            if visiting is None:
                visiting = set()
            if index in visiting:
                return set()
            visiting.add(index)
            sources: Set[int] = set()
            for predecessor in direct.get(index, ()):
                sources.update(barrier_sources(predecessor, visiting))
            visiting.remove(index)
            return sources

        for consumer in sorted(node_for):
            consumer_id = node_for[consumer]
            for predecessor in direct.get(consumer, ()):
                for producer_id in barrier_sources(predecessor):
                    if producer_id != consumer_id:
                        dependencies[consumer_id].add(producer_id)
        return tuple(
            ColoredBarrierNode(
                node_id, color, members, tuple(sorted(dependencies[node_id]))
            )
            for node_id, color, members in descriptors
        )

    algebraic_components = initial_components
    nodes = make_nodes(algebraic_components)
    # A component is non-convex exactly when its collapsed projection
    # participates in a cycle.  Split that component into its individual
    # operations and rebuild; this preserves every semantic edge while making
    # the generic-mediated boundary explicit.
    def cycle_nodes(graph_nodes: Tuple[ColoredBarrierNode, ...]) -> Set[int]:
        by_id = {node.node_id: node for node in graph_nodes}
        visiting: Set[int] = set()
        visited: Set[int] = set()
        cyclic: Set[int] = set()

        def visit(node_id: int, stack: Tuple[int, ...] = ()) -> None:
            if node_id in visiting:
                start = stack.index(node_id) if node_id in stack else 0
                cyclic.update(stack[start:])
                return
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in by_id[node_id].dependencies:
                visit(dependency, stack + (node_id,))
            visiting.remove(node_id)
            visited.add(node_id)

        for node in graph_nodes:
            visit(node.node_id)
        return cyclic

    algebraic_components = initial_components
    nodes = make_nodes(algebraic_components)
    # A component is non-convex exactly when its collapsed projection
    # participates in a cycle.  Split that component into its individual
    # operations and rebuild; this preserves every semantic edge while making
    # the generic-mediated boundary explicit.
    while True:
        cyclic = cycle_nodes(nodes)
        split = [
            node
            for node in nodes
            if node.node_id in cyclic
            and node.color == "bilinear"
            and len(node.members) > 1
        ]
        if not split:
            break
        split_components = {node.members for node in split}
        algebraic_components = [
            member
            for component in algebraic_components
            for member in (
                ((item,) for item in component)
                if component in split_components
                else (component,)
            )
        ]
        nodes = make_nodes(algebraic_components)
    if cycle_nodes(nodes):
        raise ValueError("colored barrier DAG contains a cycle")
    node_by_id = {node.node_id: node for node in nodes}
    # Longest path in the now-validated topological DAG.
    depths: Dict[int, int] = {}

    def depth(node_id: int) -> int:
        if node_id in depths:
            return depths[node_id]
        result = 1 + max((depth(dep) for dep in node_by_id[node_id].dependencies), default=0)
        depths[node_id] = result
        return result

    for node in nodes:
        depth(node.node_id)
    critical_path = max(depths.values(), default=0)

    unscheduled = set(node.node_id for node in nodes)
    scheduled: Set[int] = set()
    schedule: List[Tuple[int, ...]] = []
    previous_color = None
    switches = 0
    while unscheduled:
        ready = [
            node for node in nodes
            if node.node_id in unscheduled
            and all(dep in scheduled for dep in node.dependencies)
        ]
        if not ready:
            raise ValueError("colored barrier DAG cannot be scheduled: cycle")
        colors = {node.color for node in ready}
        color = previous_color if previous_color in colors else min(
            colors, key=lambda value: min(
                node.node_id for node in ready if node.color == value
            )
        )
        batch = tuple(node.node_id for node in ready if node.color == color)
        schedule.append(batch)
        scheduled.update(batch)
        unscheduled.difference_update(batch)
        if previous_color is not None and color != previous_color:
            switches += 1
        previous_color = color
    return ColoredBarrierDag(
        nodes=nodes,
        critical_path=critical_path,
        schedule=tuple(schedule),
        color_switches=switches,
        lower_bound=critical_path,
        theoretically_attainable_le_50=critical_path <= 50,
    )


def analyze_colored_barrier_dag(function: Function) -> ColoredBarrierDag:
    """Return compiler-only colored barrier metrics for ``function``."""
    semantic_dag = _build_semantic_dag(function)
    return _colored_barrier_dag(semantic_dag, function.tape)


build_colored_barrier_dag = analyze_colored_barrier_dag



def _semantic_waves(
    tape, nodes: Tuple[int, ...], dependencies: Dict[int, Tuple[int, ...]]
) -> Tuple[_SemanticWave, ...]:
    """Emit deterministic nonlinear strata with algebraic prerequisites first.

    Algebraic nodes retain their current expression frontier within a stratum;
    generic nodes advance to the next nonlinear level.  Views are aliases, so
    they inherit the level of their source rather than introducing a layer.
    """
    algebraic_operations = {
        OP.ADD, OP.SUB, OP.MUL, OP.DOT, OP.MATMUL, OP.CROSS
    }
    view_operations = (ReshapeOP, ConcatenateOP)
    required = set(nodes)
    inputs = set(tape.input_indicies)
    constants = {
        index for index in required
        if index not in inputs and tape.nodes[index][0] == OP.VALUE
    }
    levels: Dict[int, int] = {
        index: 0 for index in inputs | constants
    }

    def is_view(index: int) -> bool:
        return (
            index not in inputs
            and isinstance(tape.nodes[index][0], view_operations)
            or index not in inputs
            and tape.nodes[index][0] is OP.TRANSPOSE
        )

    visiting: Set[int] = set()

    def level(index: int) -> int:
        if index in levels:
            return levels[index]
        if index in visiting:
            raise ValueError("cyclic or unresolved execution dependencies")
        visiting.add(index)
        operation = tape.nodes[index][0]
        dependency_level = max(
            (level(dependency) for dependency in dependencies.get(index, ())),
            default=0,
        )
        # A view is transparent to scheduling.  Every other non-algebraic
        # operation is a nonlinear boundary and advances the stratum.
        if is_view(index):
            result = dependency_level
        elif operation in algebraic_operations:
            result = dependency_level
        else:
            result = dependency_level + 1
        visiting.remove(index)
        levels[index] = result
        return result

    for index in sorted(required):
        level(index)

    waves: List[_SemanticWave] = []
    ordinal = 0
    initial = tuple(sorted(inputs | constants))
    if initial:
        waves.append(_SemanticWave(ordinal, initial))
        ordinal += 1

    # A stratum consists of all algebraic prerequisites followed by generic
    # nodes.  The latter are emitted as ready antichains, retaining a safety
    # boundary if an unusual direct dependency ever shares their level.
    # Process each level as a small topological schedule.  Algebraic nodes
    # ready at the start of a stratum are emitted first, then all independent
    # generic nodes; algebraic nodes that consume those generic results follow
    # in the same stratum.
    scheduled = set(inputs | constants)
    max_level = max(levels.values(), default=0)
    for stratum in range(max_level + 1):
        unscheduled = {
            index for index in required
            if index not in scheduled and levels[index] == stratum
        }
        while unscheduled:
            ready = [
                index for index in sorted(unscheduled)
                if all(
                    dependency in scheduled
                    for dependency in dependencies.get(index, ())
                )
            ]
            if not ready:
                raise ValueError("cyclic or unresolved execution dependencies")
            algebraic_ready = tuple(
                index for index in ready
                if tape.nodes[index][0] in algebraic_operations or is_view(index)
            )
            if algebraic_ready:
                waves.append(_SemanticWave(ordinal, algebraic_ready))
                ordinal += 1
                unscheduled.difference_update(algebraic_ready)
                scheduled.update(algebraic_ready)
                continue
            # All generic nodes ready at this point are independent under the
            # direct execution DAG and can safely share one strict RAW layer.
            generic_ready = tuple(ready)
            waves.append(_SemanticWave(ordinal, generic_ready))
            ordinal += 1
            unscheduled.difference_update(generic_ready)
            scheduled.update(generic_ready)
    return tuple(waves)


def _build_semantic_dag(function: Function) -> _SemanticDag:
    """Collect required producers and retain both dependency projections."""
    tape = function.tape
    required: Set[int] = {
        output.index for output in function.output if output is not None
    }
    pending = list(required)
    while pending:
        node_index = pending.pop()
        if node_index in tape.input_indicies:
            continue
        operation, *arguments = tape.nodes[node_index]
        if operation == OP.VALUE:
            continue
        for argument in arguments:
            if isinstance(argument, Tracer) and argument.index not in required:
                required.add(argument.index)
                pending.append(argument.index)

    view_operations = (ReshapeOP, ConcatenateOP)

    def source_nodes(node_index: int) -> Tuple[int, ...]:
        if node_index in tape.input_indicies:
            return (node_index,)
        operation, *arguments = tape.nodes[node_index]
        if not (isinstance(operation, view_operations) or operation is OP.TRANSPOSE):
            return (node_index,)
        sources: Set[int] = set()
        for argument in arguments:
            if isinstance(argument, Tracer) and argument.index in required:
                sources.update(source_nodes(argument.index))
        return tuple(sorted(sources))

    execution_dependencies: Dict[int, Tuple[int, ...]] = {}
    materialized_dependencies: Dict[int, Tuple[int, ...]] = {}
    for node_index in sorted(required):
        if node_index in tape.input_indicies or tape.nodes[node_index][0] == OP.VALUE:
            continue
        direct = tuple(sorted({
            argument.index for argument in tape.nodes[node_index][1:]
            if isinstance(argument, Tracer) and argument.index in required
        }))
        execution_dependencies[node_index] = direct
        materialized_dependencies[node_index] = tuple(sorted({
            source for argument in tape.nodes[node_index][1:]
            if isinstance(argument, Tracer) and argument.index in required
            for source in source_nodes(argument.index)
        }))
    nodes = tuple(sorted(required))
    waves = _semantic_waves(tape, nodes, execution_dependencies)
    order = tuple(index for wave in waves for index in wave.nodes)
    return _SemanticDag(
        nodes=nodes,
        execution_dependencies=execution_dependencies,
        materialized_dependencies=materialized_dependencies,
        waves=waves,
        order=order,
    )




def _semantic_lowering_order(
    tape, semantic_dag: _SemanticDag
) -> Tuple[int, ...]:
    """Return the exact deterministic emitted wave order."""
    return tuple(
        node_index
        for wave in semantic_dag.waves
        for node_index in wave.nodes
    )


@dataclass(frozen=True)
class ViewValue:
    """Deferred alias resolved from current source nodes at a consumer boundary."""

    operation: object
    source_node_ids: Tuple[int, ...]
    shape: Tuple[int, ...]


def _schedule_residual_slots(function: Function) -> Dict[int, ValueLifetime]:
    """Assign deterministic reusable workspace slots to required tape values.

    Inputs retain their ABI order.  Every other materialized value takes the
    lowest-address free range large enough for its flattened width; a value's
    range becomes reusable immediately after its final graph/output use.
    Constants remain immediate operands and therefore consume no slot.
    """
    tape = function.tape

    # Reuse is enabled only when no scheduled frontier can retain a borrowed
    # view past the producer batch.  The conservative default handles the
    # mixed generic/algebraic case; pure residual DAGs may opt in below.
    reuse_slots = True
    semantic_dag = _build_semantic_dag(function)
    required = set(semantic_dag.nodes)
    emitted_ordinal = {
        node_index: wave.ordinal
        for wave in semantic_dag.waves
        for node_index in wave.nodes
    }
    final_uses = dict(emitted_ordinal)
    for consumer, producers in semantic_dag.materialized_dependencies.items():
        consumer_ordinal = emitted_ordinal[consumer]
        for producer in producers:
            final_uses[producer] = max(
                final_uses.get(producer, consumer_ordinal),
                consumer_ordinal,
            )
    algebraic_ops = {OP.ADD, OP.SUB, OP.MUL}
    algebraic_nodes = {
        node_index
        for node_index in required
        if node_index not in tape.input_indicies
        and tape.nodes[node_index][0] in algebraic_ops
    }
    # Keep the ordinary reverse-use result for every materialized value.  A
    # frontier is emitted later than its tape consumers, however, so reserve
    # only the roots read by that frontier through its common closure point.
    # Intermediate algebraic nodes are not roots and remain eligible for
    # residual reuse.
    frontier_roots = set()
    for algebraic_node in sorted(algebraic_nodes):
        pending_ancestors = list(tape.nodes[algebraic_node][1:])
        seen_ancestors = set()
        while pending_ancestors:
            argument = pending_ancestors.pop()
            if not isinstance(argument, Tracer):
                continue
            ancestor = argument.index
            if ancestor in seen_ancestors:
                continue
            seen_ancestors.add(ancestor)
            if ancestor in tape.input_indicies:
                frontier_roots.add(ancestor)
                continue
            operation = tape.nodes[ancestor][0]
            if operation not in algebraic_ops:
                frontier_roots.add(ancestor)
                continue
            pending_ancestors.extend(tape.nodes[ancestor][1:])
    frontier_closure = max(
        (emitted_ordinal[node_index] for node_index in algebraic_nodes),
        default=-1,
    )
    for root in frontier_roots:
        final_uses[root] = max(
            final_uses.get(root, 0),
            frontier_closure,
        )
    output_final_use = len(semantic_dag.order)
    for consumer, producers in semantic_dag.materialized_dependencies.items():
        consumer_ordinal = emitted_ordinal[consumer]
        for producer in producers:
            final_uses[producer] = max(
                final_uses.get(producer, consumer_ordinal),
                consumer_ordinal,
            )
    for output in function.output:
        if output is not None:
            final_uses[output.index] = output_final_use
    for node_index in sorted(required, reverse=True):
        if node_index in tape.input_indicies:
            continue
        operation, *arguments = tape.nodes[node_index]
        if operation == OP.VALUE:
            continue
        if not (
            isinstance(operation, (ReshapeOP, ConcatenateOP))
            or operation is OP.TRANSPOSE
        ):
            continue
        view_use = final_uses.get(
            node_index, emitted_ordinal[node_index]
        )
        for argument in arguments:
            if isinstance(argument, Tracer) and argument.index in required:
                final_uses[argument.index] = max(
                    final_uses.get(argument.index, 0),
                    view_use,
                )

    free_ranges: List[SlotRange] = []
    active: Dict[int, ValueLifetime] = {}
    lifetimes: Dict[int, ValueLifetime] = {}
    next_slot = 0

    def release(slot: SlotRange) -> None:
        free_ranges.append(slot)
        free_ranges.sort(key=lambda candidate: candidate.start)
        merged: List[SlotRange] = []
        for candidate in free_ranges:
            if (
                merged
                and merged[-1].start + merged[-1].length == candidate.start
            ):
                previous = merged.pop()
                merged.append(
                    SlotRange(
                        previous.start, previous.length + candidate.length
                    )
                )
            else:
                merged.append(candidate)
        free_ranges[:] = merged

    def allocate(width: int) -> SlotRange:
        nonlocal next_slot
        candidate_index = next(
            (
                index
                for index, slot in enumerate(free_ranges)
                if slot.length >= width
            ),
            None,
        )
        if candidate_index is None:
            slot = SlotRange(next_slot, width)
            next_slot += width
            return slot
        slot = free_ranges.pop(candidate_index)
        if slot.length > width:
            free_ranges.append(
                SlotRange(slot.start + width, slot.length - width)
            )
            free_ranges.sort(key=lambda candidate: candidate.start)
        return SlotRange(slot.start, width)

    for _tape_ordinal, node_index in enumerate(semantic_dag.order):
        # Multiple nodes can share an emitted wave; they are simultaneous for
        # lifetime purposes and must not recycle one another's ranges.
        definition_ordinal = emitted_ordinal[node_index]
        for value, lifetime in tuple(active.items()):
            if reuse_slots and lifetime.final_use < definition_ordinal:
                release(lifetime.slot)
                del active[value]
        if node_index in tape.input_indicies:
            operation = None
        else:
            operation = tape.nodes[node_index][0]
        if (
            isinstance(operation, (ReshapeOP, ConcatenateOP))
            or operation is OP.TRANSPOSE
        ):
            continue
        if node_index not in tape.input_indicies and operation == OP.VALUE:
            continue
        width = tape.dim[node_index].flat()
        lifetime = ValueLifetime(
            value=node_index,
            width=width,
            first_definition=definition_ordinal,
            final_use=final_uses[node_index],
            slot=allocate(width),
        )
        lifetimes[node_index] = lifetime
        active[node_index] = lifetime
    return lifetimes


class _FunctionTableBuilder:
    def __init__(self):
        self._function_ids_by_identity: Dict[int, int] = {}
        self._graphs_by_id: Dict[int, SparseNet | None] = {}

    def build(self, function: Function) -> FunctionTable:
        entry_function_id, _graph = self.get_or_build(function)
        ordered_graphs = [
            self._graphs_by_id[function_id]
            for function_id in range(len(self._graphs_by_id))
        ]
        assert all(graph_item is not None for graph_item in ordered_graphs)
        return FunctionTable(list(ordered_graphs), entry_function_id)

    def get_or_build(self, function: Function) -> Tuple[int, SparseNet]:
        function_identity = id(function)
        if function_identity in self._function_ids_by_identity:
            function_id = self._function_ids_by_identity[function_identity]
            existing_graph = self._graphs_by_id[function_id]
            if existing_graph is None:
                raise NotImplementedError(
                    "Recursive function evaluation is not supported"
                )
            return function_id, existing_graph

        function_id = len(self._function_ids_by_identity)
        self._function_ids_by_identity[function_identity] = function_id
        self._graphs_by_id[function_id] = None
        graph = _create_opgraph(function, self)
        self._graphs_by_id[function_id] = graph
        return function_id, graph


def _create_residual_opgraph(
    function: Function, function_table_builder: _FunctionTableBuilder
) -> SparseNet:
    """Lower a tape directly to deterministic stable-slot residual stages.

    The temporary ``BilinearWeights`` values below are only a convenient
    compiler algebra; emitted records contain absolute slots and no legacy
    layers.  Each tape node is a separate stage, which mechanically prevents
    intra-stage RAW hazards and preserves tape order.
    """
    tape = function.tape
    semantic_dag = _build_semantic_dag(function)
    lifetimes = _schedule_residual_slots(function)
    workspace_size = max(
        (life.slot.start + life.slot.length for life in lifetimes.values()),
        default=0,
    )
    input_bindings = []
    for index in tape.input_indicies:
        life = lifetimes[index]
        input_bindings.append(
            InputBinding(tuple(range(life.slot.start, life.slot.start + life.width)))
        )
    input_map = InputMap(tuple(input_bindings))
    values: Dict[int, object] = {}
    refs: Dict[int, Tuple[int, ...]] = {}
    stages = []
    compiler_memory = MemorySpec(0, workspace_size)
    for value_index, (value_operation, *value_arguments) in enumerate(tape.nodes):
        if value_operation is OP.VALUE:
            values[value_index] = _as_numpy_value(value_arguments[0])

    for index in tape.input_indicies:
        life = lifetimes[index]
        refs[index] = tuple(life.slot.start + i for i in range(life.width))
        values[index] = BilinearWeights.project(
            compiler_memory,
            MemorySpec(life.slot.start, life.width),
            _node_shape(tape.dim[index]),
        )

    def scalar_constant(value):
        return RetainedExpression(
            (), constant=float(np.asarray(value).reshape(-1)[0])
        )

    def operand(value, index, row=0):
        if index in refs:
            slots = refs[index]
            return SlotOperand(slots[min(row, len(slots) - 1)])
        if isinstance(value, (int, float, np.number)):
            return scalar_constant(value)
        array = np.asarray(value).reshape(-1)
        return scalar_constant(array[min(row, len(array) - 1)])

    def emit_bilinear(index, value):
        life = lifetimes[index]
        output_shape = _node_shape(tape.dim[index])
        rows = []
        for row in range(life.width):
            terms = []
            constant = value.constant.keys.get((row,), 0.0)
            if constant:
                terms.append(BilinearTerm(None, None, float(constant)))
            for key, coefficient in value.linear.keys.items():
                if key[0] == row and coefficient:
                    terms.append(BilinearTerm(None, key[1], float(coefficient)))
            for key, coefficient in value.quadratic.keys.items():
                if key[0] == row and coefficient:
                    terms.append(BilinearTerm(key[1], key[2], float(coefficient)))
            absolute = life.slot.start + row
            translated = []
            for term in terms:
                left = None if term.left is None else term.left
                right = None if term.right is None else term.right
                translated.append(BilinearTerm(left, right, term.coefficient))
            rows.append(BilinearRow(absolute, tuple(sorted(
                translated,
                key=lambda term: (-1 if term.left is None else term.left,
                                 -1 if term.right is None else term.right),
            ))))
        values[index] = value
        if rows:
            stages.append(BilinearStage(tuple(rows)))
        refs[index] = tuple(life.slot.start + i for i in range(life.width))

    def view_refs(index):
        value = values[index]
        if isinstance(value, tuple):
            return value
        return refs[index]

    for index in sorted(semantic_dag.nodes):
        if index in tape.input_indicies:
            life = lifetimes[index]
            refs[index] = tuple(life.slot.start + i for i in range(life.width))
            values[index] = BilinearWeights.project(
                compiler_memory,
                MemorySpec(life.slot.start, life.width),
                _node_shape(tape.dim[index]),
            )
            continue
        operation, *arguments = tape.nodes[index]
        if operation is OP.VALUE:
            values[index] = _as_numpy_value(arguments[0])
            continue
        if isinstance(operation, (ReshapeOP, ConcatenateOP)) or operation is OP.TRANSPOSE:
            source_refs = []
            for argument in arguments:
                source_refs.extend(view_refs(argument.index))
            if isinstance(operation, ReshapeOP):
                source_refs = np.reshape(
                    np.asarray(source_refs), _node_shape(tape.dim[index]),
                    order=operation.order,
                ).reshape(-1).tolist()
            elif operation is OP.TRANSPOSE:
                source_shape = _node_shape(tape.dim[arguments[0].index])
                source_refs = np.transpose(
                    np.asarray(source_refs).reshape(source_shape)
                ).reshape(-1).tolist()
            refs[index] = tuple(int(slot) for slot in source_refs)
            values[index] = refs[index]
            continue
        if operation is OP.EVALUATE and isinstance(values.get(arguments[0].index), Function):
            callee = values[arguments[0].index]
            callee_id, callee_graph = function_table_builder.get_or_build(callee)
            input_slots = tuple(
                view_refs(argument.index) for argument in arguments[1:]
                if isinstance(argument, Tracer)
            )
            life = lifetimes[index]
            output_slots = tuple(life.slot.start + i for i in range(life.width))
            stages.append(CallStage(callee_graph, input_slots, output_slots))
            refs[index] = output_slots
            values[index] = output_slots
            continue
        operands = []
        for argument in arguments:
            if isinstance(argument, Tracer):
                operand_value = values[argument.index]
                if argument.index in refs:
                    operands.append(BilinearWeights.project(
                        compiler_memory,
                        MemorySpec(refs[argument.index][0], len(refs[argument.index])),
                        _node_shape(tape.dim[argument.index]),
                    ))
                else:
                    operands.append(operand_value)
            else:
                operands.append(argument)
        if operation in {OP.ADD, OP.SUB, OP.MUL} and any(
            isinstance(item, BilinearWeights) for item in operands
        ):
            normalized = []
            shape = _node_shape(tape.dim[index])
            for item in operands:
                if isinstance(item, BilinearWeights):
                    normalized.append(item)
                else:
                    normalized.append(_constant_to_bw(
                        compiler_memory,
                        np.full(shape, item) if np.isscalar(item) else item,
                        shape,
                    ))
            result = compiler_binary({OP.ADD: "add", OP.SUB: "sub", OP.MUL: "mul"}[operation], *normalized)
            if result is not None:
                emit_bilinear(index, result)
                continue
        life = lifetimes[index]
        rows = []
        for row in range(life.width):
            args = [
                operand(
                    values[arg.index] if isinstance(arg, Tracer) else arg,
                    arg.index if isinstance(arg, Tracer) else -1,
                    row,
                )
                for arg in arguments
            ]
            rows.append(NonlinearOperation(
                life.slot.start + row, operation, args[0],
                args[1] if len(args) > 1 else None,
                args[2] if len(args) > 2 else None,
            ))
        stages.append(NonlinearStage(tuple(rows)))
        refs[index] = tuple(life.slot.start + i for i in range(life.width))
        values[index] = refs[index]

    output_bindings = []
    for output in function.output:
        if output is None:
            continue
        output_refs = view_refs(output.index)
        output_bindings.append(OutputBinding(tuple(output_refs), tuple(_node_shape(output.dim))))
    graph = SparseNet(
        workspace_size,
        input_map,
        OutputMap(tuple(output_bindings)),
        residual_stages=tuple(stages),
    )
    graph.residual_lifetimes = lifetimes
    graph.residual_workspace_size = workspace_size
    return graph


def _create_legacy_opgraph(
    function: Function,
    function_table_builder: _FunctionTableBuilder,
):
    tape = function.tape
    semantic_dag = _build_semantic_dag(function)
    wave_boundary_inputs: Dict[int, Tuple[int, ...]] = {}
    for wave in semantic_dag.waves:
        inputs: Set[int] = set()
        for node_index in wave.nodes:
            if node_index in tape.input_indicies:
                continue
            operation, *arguments = tape.nodes[node_index]
            if operation in {
                OP.VALUE, OP.ADD, OP.SUB, OP.MUL, OP.DOT, OP.MATMUL, OP.CROSS
            }:
                continue
            inputs.update(
                argument.index
                for argument in arguments
                if isinstance(argument, Tracer)
            )
        for node_index in wave.nodes:
            wave_boundary_inputs[node_index] = tuple(sorted(inputs))
    residual_lifetimes = _schedule_residual_slots(function)
    numpy_backend = get_backend_by_name("numpy", set_current=False)
    required_nodes = set(semantic_dag.nodes)
    wave_ordinals = {
        node_index: wave.ordinal
        for wave in semantic_dag.waves
        for node_index in wave.nodes
    }
    algebraic_operations = {OP.ADD, OP.SUB, OP.MUL}
    symbolic_dependencies = semantic_dag.materialized_dependencies
    earliest_consumer_wave: Dict[int, int] = {}
    for consumer, predecessors in symbolic_dependencies.items():
        operation = tape.nodes[consumer][0]
        if operation in algebraic_operations or (
            isinstance(operation, (ReshapeOP, ConcatenateOP))
            or operation is OP.TRANSPOSE
        ):
            continue
        consumer_wave = wave_ordinals[consumer]
        for predecessor in predecessors:
            earliest_consumer_wave[predecessor] = min(
                earliest_consumer_wave.get(predecessor, consumer_wave),
                consumer_wave,
            )
    output_consumer_wave = len(semantic_dag.waves)
    for output in function.output:
        if output is not None:
            earliest_consumer_wave[output.index] = min(
                earliest_consumer_wave.get(output.index, output_consumer_wave),
                output_consumer_wave,
            )

    output_indices = {
        output.index for output in function.output if output is not None
    }
    remaining_uses = _stable_use_counts(tape, required_nodes, function.output)
    input_layer = InputLayer()
    node_values = {}
    node_specs: Dict[int, MemorySpec] = {}
    frontier = _EpochFrontier(
        tuple(sorted(tape.input_indicies)),
        {},
        dict(remaining_uses),
    )
    frontier_metadata = []
    for input_index in tape.input_indicies:
        input_position = input_layer.add_input(tape.dim[input_index])
        _abi_spec, _shape = input_layer.input_specs[input_position]
        node_specs[input_index] = MemorySpec(
            residual_lifetimes[input_index].slot.start,
            residual_lifetimes[input_index].slot.length,
        )
    residual_workspace_size = max(
        (
            lifetime.slot.start + lifetime.slot.length
            for lifetime in residual_lifetimes.values()
        ),
        default=input_layer.dimension,
    )
    current_memory = MemorySpec(0, residual_workspace_size)
    current_size = current_memory.count

    def resolve_view(value):
        """Resolve a deferred view against current source-node values."""
        if not isinstance(value, ViewValue):
            return value
        operands = tuple(
            resolve_view(node_values[source_index])
            for source_index in value.source_node_ids
        )
        operation = value.operation
        if isinstance(operation, ReshapeOP):
            operand = operands[0]
            if isinstance(operand, BilinearWeights):
                return operand.reshape(operation.newshape, order=operation.order)
            return np.reshape(operand, operation.newshape, order=operation.order)
        if operation == OP.TRANSPOSE:
            operand = operands[0]
            if isinstance(operand, BilinearWeights):
                return operand.transpose()
            return np.transpose(operand)
        if isinstance(operation, ConcatenateOP):
            if any(isinstance(operand, BilinearWeights) for operand in operands):
                bilinear_memories = [
                    operand.memory for operand in operands
                    if isinstance(operand, BilinearWeights)
                ]
                base_memory = max(
                    bilinear_memories,
                    key=lambda memory: (memory.count, memory.location),
                )
                memory = MemorySpec(
                    base_memory.location,
                    max(memory.count for memory in bilinear_memories),
                )
                aligned_operands = [
                    operand.extend_memory(memory)
                    if isinstance(operand, BilinearWeights)
                    and operand.memory is not memory
                    else operand
                    for operand in operands
                ]
                return _concatenate_bilinear_operands(
                    memory, aligned_operands, axis=operation.axis
                )
            return np.concatenate(operands, axis=operation.axis)
        raise TypeError(f"Unsupported compiler view {operation!r}")
    for input_index in tape.input_indicies:
        node_values[input_index] = BilinearWeights.project(
            current_memory,
            node_specs[input_index],
            _node_shape(tape.dim[input_index]),
        )
    layers = []
    pending_generic = None
    pending_generic_nodes = set()
    generic_nodes = set()
    pending_bilinear_nodes: List[int] = []
    generic_flushes: List[dict] = []
    def flush_generic(cause: str):
        nonlocal pending_generic, pending_generic_nodes
        if pending_generic is None:
            return
        flush_nodes = tuple(sorted(pending_generic_nodes))
        flush_waves = tuple(sorted({
            wave_ordinals[node_index]
            for node_index in flush_nodes
        }))
        generic_flushes.append({
            "wave": flush_waves[-1] if flush_waves else None,
            "waves": flush_waves,
            "cause": cause,
            "nodes": flush_nodes,
        })
        if pending_generic is None:
            return
        epoch_memory = pending_generic["memory_in"]
        constant_values = pending_generic["constants"]
        generic_memory = epoch_memory
        if constant_values:
            generic_memory = MemorySpec(
                epoch_memory.location,
                epoch_memory.count + len(constant_values),
            )
            layers.append(
                BilinearWorkspaceLayer(
                    epoch_memory,
                    generic_memory,
                    _constant_extension_weights(
                        epoch_memory,
                        epoch_memory.count,
                        list(constant_values),
                    ),
                    destination_rows=range(
                        epoch_memory.count, generic_memory.count
                    ),
            )
            )
        layers.append(
            GenericVectorLayer(
                generic_memory,
                generic_memory,
                pending_generic["ops"],
                opaque_programs=pending_generic["opaque_programs"],
            )
        )
        pending_generic = None
        pending_generic_nodes.clear()

    def pending_nodes():
        return set(pending_bilinear_nodes)

    def pending_for_boundary(node_indices):
        """Select pending expressions directly required by this boundary."""
        return {
            node_index
            for node_index in node_indices
            if node_index in frontier.values and node_index not in node_specs
        }

    def view_refs(value):
        """Return workspace references when a view is row-addressable."""
        if isinstance(value, BilinearWeights):
            if value.constant.keys or value.quadratic.keys:
                return None
            refs = []
            for key in np.ndindex(value.shape):
                terms = [
                    (index[-1], coefficient)
                    for index, coefficient in value.linear.keys.items()
                    if index[:-1] == key
                ]
                if len(terms) != 1 or terms[0][1] != 1:
                    return None
                refs.append(terms[0][0])
            return refs
        if not isinstance(value, ViewValue):
            return None
        operand_refs = [
            view_refs(node_values[source_index])
            for source_index in value.source_node_ids
        ]
        if any(refs is None for refs in operand_refs):
            return None
        operation = value.operation
        if isinstance(operation, ReshapeOP):
            return np.reshape(
                np.asarray(operand_refs[0]), value.shape, order=operation.order
            ).reshape(-1).tolist()
        if operation == OP.TRANSPOSE:
            source_shape = _view_shape(node_values[value.source_node_ids[0]])
            return np.transpose(
                np.asarray(operand_refs[0]).reshape(source_shape)
            ).reshape(-1).tolist()
        if isinstance(operation, ConcatenateOP):
            arrays = [
                np.asarray(refs).reshape(_view_shape(node_values[source_index]))
                for refs, source_index in zip(
                    operand_refs, value.source_node_ids
                )
            ]
            return np.concatenate(arrays, axis=operation.axis).reshape(-1).tolist()
        return None

    def queue_non_linear_view_sources(node_index: int):
        """Queue nonlinear sources before a view reaches a bilinear consumer."""
        value = node_values[node_index]
        if isinstance(value, ViewValue):
            for source_index in value.source_node_ids:
                queue_non_linear_view_sources(source_index)
            return
        if (
            isinstance(value, BilinearWeights)
            and value.quadratic.keys
            and node_index in residual_lifetimes
            and node_index not in node_specs
        ):
            queue_bilinear(node_index, value)

    def _view_shape(value):
        if isinstance(value, ViewValue):
            return value.shape
        if isinstance(value, BilinearWeights):
            return value.shape
        return np.asarray(value).shape

    def extend_node_values(new_memory: MemorySpec):
        for node_index, value in list(node_values.items()):
            if (
                isinstance(value, BilinearWeights)
                and value.memory != new_memory
            ):
                node_values[node_index] = value.extend_memory(new_memory)
    def queue_bilinear(node_index: int, value: BilinearWeights):
        node_values[node_index] = value
        degree = 2 if value.quadratic.keys else (1 if value.linear.keys else 0)
        frontier.track(
            node_index,
            value,
            degree,
            symbolic_dependencies.get(node_index, ()),
            earliest_consumer_wave.get(node_index),
        )
        if node_index not in pending_bilinear_nodes and node_index not in node_specs:
            pending_bilinear_nodes.append(node_index)
            pending_bilinear_nodes.sort()
    boundary_reasons: List[str] = []

    def materialize_boundary(reason: str, node_indices=None):
        """Materialize only expressions consumed by this boundary."""
        nonlocal current_memory, current_size
        frontier.close(reason)

        def pending_closure(indices) -> set[int]:
            selected: set[int] = set()
            pending = list(indices)
            while pending:
                node_index = pending.pop()
                if node_index in selected:
                    continue
                value = node_values.get(node_index)
                if isinstance(value, ViewValue):
                    pending.extend(value.source_node_ids)
                    continue
                if node_index in pending_bilinear_nodes:
                    selected.add(node_index)
            return selected

        selected = (
            set(pending_bilinear_nodes)
            if node_indices is None
            else pending_closure(node_indices)
        )
        flush_generic(f"boundary:{reason}")
        boundary_reasons.append(reason)
        if not selected:
            return
        frontier_metadata.append(
            {
                "reason": reason,
                "selected": tuple(sorted(selected)),
                "values": {
                    index: frontier.values[index]
                    for index in sorted(selected)
                    if index in frontier.values
                },
                "remaining": tuple(sorted(
                    index for index in frontier.values if index not in selected
                )),
            }
        )
        previous_memory = current_memory
        output_rows = set()
        pending_specs = []
        for node_index in sorted(selected):
            shape = _node_shape(tape.dim[node_index])
            if node_index in residual_lifetimes:
                lifetime = residual_lifetimes[node_index]
                spec = MemorySpec(lifetime.slot.start, lifetime.slot.length)
            else:
                spec = MemorySpec(current_memory.count, tape.dim[node_index].flat())
            pending_specs.append((node_index, node_values[node_index], shape, spec))
            output_rows.update(range(spec.location, spec.location + spec.count))
        new_memory = MemorySpec(
            0,
            max(
                residual_workspace_size,
                previous_memory.count,
                *(spec.location + spec.count for _, _, _, spec in pending_specs),
            ),
        )
        constant = dok_ndarray((new_memory.count,))
        linear = dok_ndarray((new_memory.count, previous_memory.count))
        quadratic = dok_ndarray((new_memory.count, previous_memory.count, previous_memory.count))
        for _node_index, value, output_shape, spec in pending_specs:
            _append_bilinear_value(constant, linear, quadratic, spec.location, value, output_shape)
        layer = BilinearWorkspaceLayer(
            previous_memory, new_memory,
            BilinearWeights(previous_memory, (new_memory.count,),
                            constant=constant, linear=linear, quadratic=quadratic),
            destination_rows=output_rows,
        )
        # Keep semantic frontier boundaries authoritative: the post-pass may
        # coalesce independent rows within one boundary, but must not erase a
        # closure that separates two epochs.
        layer._frontier_id = len(frontier_metadata)
        layers.append(layer)
        current_memory = new_memory
        current_size = new_memory.count
        for node_index, _value, shape, spec in pending_specs:
            node_specs[node_index] = spec
            node_values[node_index] = BilinearWeights.project(new_memory, spec, shape)
            frontier.values.pop(node_index, None)
        pending_bilinear_nodes[:] = [
            node_index
            for node_index in pending_bilinear_nodes
            if node_index not in selected
        ]
        extend_node_values(new_memory)



    def lower_function_evaluation(node_index: int, arguments):
        """Emit a nested graph call after materializing consumed expressions."""
        nonlocal current_memory, current_size
        materialize_boundary("nested-call-input")

        function_value = node_values[arguments[0].index]
        if not isinstance(function_value, Function):
            raise NotImplementedError(
                "Function evaluation requires a statically known Function"
            )
        callee_function_id, callee_graph = function_table_builder.get_or_build(
            function_value
        )
        input_bindings = [
            _build_opaque_operand(
                node_values[argument.index],
                node_specs.get(argument.index),
                _node_shape(tape.dim[argument.index]),
            )
            for argument in arguments[1:]
        ]
        output_shape = _node_shape(tape.dim[node_index])
        output_count = tape.dim[node_index].flat()
        lifetime = residual_lifetimes[node_index]
        output_spec = MemorySpec(lifetime.slot.start, output_count)
        layers.append(
            FunctionEvaluationLayer(
                current_memory,
                MemorySpec(
                    0,
                    max(current_memory.count, output_spec.location + output_count),
                ),
                input_bindings,
                [output_spec],
                callee_graph,
                callee_function_id,
            )
        )
        current_memory = layers[-1].memory_out
        current_size = current_memory.count
        node_specs[node_index] = output_spec
        node_values[node_index] = BilinearWeights.project(
            current_memory, output_spec, output_shape
        )
    def lower_generic(
        node_index: int, operation, arguments, boundary_inputs
    ):
        nonlocal current_memory, current_size, pending_generic
        if operation == OP.CASE:
            # CASE is a generic barrier: every algebraic branch must have a
            # stable slot before its row references are captured.  In
            # particular, do not let a branch remain only in the pending
            # frontier while the condition and branches are added to one
            # generic batch.
            for argument in arguments:
                if not isinstance(argument, Tracer):
                    continue
                value = node_values[argument.index]
                if (
                    isinstance(value, BilinearWeights)
                    and argument.index in residual_lifetimes
                    and argument.index not in node_specs
                ):
                    queue_bilinear(argument.index, value)
        def depends_on_generic(index: int, seen=None) -> bool:
            """Whether an argument reads an output of this generic batch."""
            if seen is None:
                seen = set()
            if index in seen:
                return False
            if index in pending_generic_nodes:
                return True
            value = node_values.get(index)
            if isinstance(value, ViewValue):
                return any(
                    depends_on_generic(source_index, seen)
                    for source_index in value.source_node_ids
                )
            return False

        # The runtime validator intentionally rejects a later row reading or
        # rewriting an earlier row's output in the same scheduled layer.  A
        # generic chain therefore needs a layer boundary even when the DAG
        # scheduler made all of its nodes part of one nonlinear wave.
        if pending_generic and any(
            isinstance(argument, Tracer)
            and depends_on_generic(argument.index)
            for argument in arguments
        ):
            flush_generic("direct-dependency")
        pending_generic_arguments = [
            argument.index
            for argument in arguments
            if isinstance(argument, Tracer)
            and argument.index in pending_nodes()
        ]
        if pending_generic_arguments:
            materialize_boundary("generic-consumer")
        if pending_generic is None:
            pending_generic = {
                "memory_in": current_memory,
                "ops": [],
                "constants": [],
                "opaque_programs": [],
            }
        output_shape = _node_shape(tape.dim[node_index])
        output_count = tape.dim[node_index].flat()

        def reserve_constant_rows(value, shape: Tuple[int, ...]):
            nonlocal current_memory, current_size
            rows = _flatten_constant_rows(value, shape)
            if rows and all(row == rows[0] for row in rows[1:]):
                rows = rows[:1]
            start = (
                pending_generic["memory_in"].count
                + len(pending_generic["constants"])
            )
            pending_generic["constants"].extend(rows)
            current_memory_ref = MemorySpec(
                pending_generic["memory_in"].location,
                pending_generic["memory_in"].count
                + len(pending_generic["constants"]),
            )
            current_memory = current_memory_ref
            current_size = current_memory_ref.count
            if len(rows) == 1:
                return start
            return [start + offset for offset in range(len(rows))]

        def refs_for_arg(argument):
            nonlocal current_memory, current_size
            argument_shape = _node_shape(tape.dim[argument.index])
            if (
                argument.index in node_specs
                and isinstance(node_values[argument.index], BilinearWeights)
            ):
                spec = node_specs[argument.index]
                if spec.count == 1:
                    return spec.location
                return [
                    spec.location + offset for offset in range(spec.count)
                ]
            view_value = node_values[argument.index]
            view_references = view_refs(view_value)
            if view_references is not None:
                return view_references
            if isinstance(view_value, ViewValue):
                # Nonlinear views are closed above and then resolve through
                # the source's stable workspace rows.  A view-local copy would
                # violate aliasing and inflate the frontier layer.
                raise ValueError(
                    f"unsupported non-bilinear dynamic view at node {argument.index}"
                )
            if (
                argument.index in residual_lifetimes
                and isinstance(node_values[argument.index], BilinearWeights)
            ):
                lifetime = residual_lifetimes[argument.index]
                spec = MemorySpec(lifetime.slot.start, lifetime.slot.length)
                node_specs[argument.index] = spec
                if spec.count == 1:
                    return spec.location
                return [
                    spec.location + offset for offset in range(spec.count)
                ]
            return reserve_constant_rows(
                node_values[argument.index], argument_shape
            )

        def row_ref(refs, row: int):
            if isinstance(refs, int):
                return refs
            if len(refs) == 1:
                return refs[0]
            return refs[row]

        appended_operations = []
        scalar_lowered = False
        if operation in {
            OP.SIN,
            OP.COS,
            OP.TAN,
            OP.EXP,
            OP.SQRT,
            OP.LOG,
            OP.NEG,
            OP.ABS,
        }:
            (argument,) = arguments
            refs = refs_for_arg(argument)
            appended_operations.extend(
                (operation, row_ref(refs, row), UNUSED_REF, UNUSED_REF)
                for row in range(output_count)
            )
            scalar_lowered = True
        elif operation in {
            OP.ADD,
            OP.SUB,
            OP.MUL,
            OP.DIV,
            OP.PWR,
            OP.INT_PWR,
            OP.ARCTAN2,
            OP.EQUAL,
            OP.LESS_THAN,
            OP.LESS_EQUAL,
        }:
            left_refs = refs_for_arg(arguments[0])
            right_refs = refs_for_arg(arguments[1])
            appended_operations.extend(
                (
                    operation,
                    row_ref(left_refs, row),
                    row_ref(right_refs, row),
                    UNUSED_REF,
                )
                for row in range(output_count)
            )
            scalar_lowered = True
        elif operation == OP.CASE:
            condition_refs = refs_for_arg(arguments[0])
            true_refs = refs_for_arg(arguments[1])
            false_refs = refs_for_arg(arguments[2])
            appended_operations.extend(
                (
                    operation,
                    row_ref(condition_refs, 0),
                    row_ref(true_refs, row),
                    row_ref(false_refs, row),
                )
                for row in range(output_count)
            )
            scalar_lowered = True


        lifetime = residual_lifetimes[node_index]
        output_spec = MemorySpec(lifetime.slot.start, lifetime.slot.length)
        layer_operations = [
            (output_spec.location + offset, *operation_row)
            for offset, operation_row in enumerate(appended_operations)
        ]
        opaque_programs: List[OpaqueProgram] = []
        if not scalar_lowered:
            operand_specs = []
            for argument in arguments:
                spec = node_specs.get(argument.index)
                operand_specs.append(
                    _build_opaque_operand(
                        node_values[argument.index],
                        spec,
                        _node_shape(tape.dim[argument.index]),
                    )
                )
            opaque_programs.append(
                OpaqueProgram(
                    output_spec.location,
                    output_shape,
                    operation,
                    tuple(operand_specs),
                )
            )
            for row in range(output_count):
                layer_operations.append(
                    (
                        output_spec.location + row,
                        OPAQUE_OP,
                        0,
                        row,
                        UNUSED_REF,
                    )
                )
        pending_generic["ops"].extend(layer_operations)
        pending_generic["opaque_programs"].extend(opaque_programs)
        # Generic outputs still own their stable residual slot.  Recording
        # that binding prevents later CASE rows from deriving references from
        node_specs[node_index] = output_spec
        node_values[node_index] = BilinearWeights.project(
            current_memory, output_spec, output_shape
        )
        generic_nodes.add(node_index)
        pending_generic_nodes.add(node_index)
    def release_arguments(arguments):
        for argument in arguments:
            if not isinstance(argument, Tracer):
                continue
            remaining_uses[argument.index] -= 1
    # The semantic DAG is the sole source of dependency order. Views have
    # already been resolved to their materialized producers in its edges.
    lowering_order = _semantic_lowering_order(tape, semantic_dag)

    for node_index in lowering_order:
        if node_index in tape.input_indicies:
            continue
        operation, *arguments = tape.nodes[node_index]
        if operation == OP.VALUE:
            (constant_value,) = arguments
            node_values[node_index] = _as_numpy_value(constant_value)
            continue
        if node_index not in required_nodes:
            continue
        try:
            if (
                isinstance(operation, (ReshapeOP, ConcatenateOP))
                or operation is OP.TRANSPOSE
            ):
                node_values[node_index] = ViewValue(
                    operation,
                    tuple(argument.index for argument in arguments),
                    tuple(_node_shape(tape.dim[node_index])),
                )
                continue
            operands = [
                resolve_view(node_values[argument.index])
                for argument in arguments
            ]
            if operation not in {OP.DOT, OP.CROSS} and all(
                not isinstance(operand, BilinearWeights)
                for operand in operands
            ):
                node_values[node_index] = numpy_backend.call(
                    operation, *operands
                )
                continue

            if operation == OP.EVALUATE and isinstance(
                node_values[arguments[0].index], Function
            ):
                lower_function_evaluation(node_index, arguments)
                continue
            if operation in {OP.ADD, OP.SUB, OP.MUL}:
                if pending_nodes() and any(
                    isinstance(argument, Tracer)
                    and argument.index in generic_nodes
                    for argument in arguments
                ) and any(
                    isinstance(argument, Tracer)
                    and argument.index in pending_nodes()
                    for argument in arguments
                ):
                    materialize_boundary("nonlinear-scalar-consumer")
            if operation in {OP.ADD, OP.SUB, OP.MUL} and any(
                isinstance(operand, BilinearWeights) for operand in operands
            ) and all(
                not isinstance(operand, BilinearWeights)
                or operand.shape == _node_shape(tape.dim[node_index])
                for operand in operands
            ):
                output_shape = _node_shape(tape.dim[node_index])

                def algebra_operands():
                    normalized = []
                    for operand in operands:
                        if isinstance(operand, BilinearWeights):
                            value = operand
                        else:
                            value = _constant_to_bw(
                                current_memory,
                                np.full(output_shape, operand)
                                if np.isscalar(operand)
                                else operand,
                                output_shape,
                            )
                        if value.memory is not current_memory:
                            value = value.extend_memory(current_memory)
                        normalized.append(value)
                    return normalized

                algebra_values = algebra_operands()
                result = compiler_binary(
                    {OP.ADD: "add", OP.SUB: "sub", OP.MUL: "mul"}[operation],
                    *algebra_values,
                )
                if result is None:
                    materialize_boundary("bilinear-degree")
                    operands = [
                        resolve_view(node_values[argument.index])
                        for argument in arguments
                    ]
                    algebra_values = algebra_operands()
                    result = compiler_binary(
                        {OP.ADD: "add", OP.SUB: "sub", OP.MUL: "mul"}[operation],
                        *algebra_values,
                    )
                if result is None:
                    raise TypeError(
                        f"compiler algebra could not lower {operation} "
                        f"at tape node {node_index}"
                    )
                queue_bilinear(node_index, result)
                continue
            elif operation in {OP.DOT, OP.MATMUL, OP.CROSS}:
                for argument in arguments:
                    if isinstance(argument, Tracer):
                        queue_non_linear_view_sources(argument.index)
                operands = [
                    resolve_view(node_values[argument.index])
                    for argument in arguments
                ]
                degrees = [
                    2 if isinstance(operand, BilinearWeights)
                    and operand.quadratic.keys
                    else (
                        1 if isinstance(operand, BilinearWeights)
                        and operand.linear.keys
                        else 0
                    )
                    for operand in operands
                ]
                if (
                    any(degree >= 2 for degree in degrees)
                    and sum(degree > 0 for degree in degrees) >= 2
                ):
                    materialize_boundary("bilinear-degree")
                    operands = [
                        resolve_view(node_values[argument.index])
                        for argument in arguments
                    ]
                required_memory = max(
                    (
                        operand.memory.count
                        for operand in operands
                        if isinstance(operand, BilinearWeights)
                    ),
                    default=current_memory.count,
                )
                required_memory = max(required_memory, current_memory.count)
                if required_memory > current_memory.count:
                    current_memory = MemorySpec(
                        current_memory.location, required_memory
                    )
                    current_size = required_memory
                operands = [
                    (
                        _constant_to_bw(
                            current_memory,
                            operand,
                            _node_shape(tape.dim[argument.index]),
                        )
                        if not isinstance(operand, BilinearWeights)
                        else operand
                    )
                    for operand, argument in zip(operands, arguments)
                ]
                operands = [
                    (
                        operand.extend_memory(current_memory)
                        if isinstance(operand, BilinearWeights)
                        and operand.memory is not current_memory
                        else operand
                    )
                    for operand in operands
                ]
                memories = {
                    (operand.memory.location, operand.memory.count)
                    for operand in operands
                    if isinstance(operand, BilinearWeights)
                }
                if len(memories) <= 1:
                    try:
                        result = ops[operation](*operands)
                        if not isinstance(result, BilinearWeights):
                            result = _constant_to_bw(
                                current_memory,
                                result,
                                _node_shape(tape.dim[node_index]),
                            )
                        queue_bilinear(node_index, result)
                        continue
                    except (
                        AssertionError,
                        TypeError,
                        NotImplementedError,
                    ) as error:
                        raise TypeError(
                            f"scheduled {operation} lowering failed at tape node {node_index}"
                        ) from error

            lower_generic(
                node_index,
                operation,
                arguments,
                wave_boundary_inputs[node_index],
            )

        finally:
            release_arguments(arguments)
    pure_view_outputs = (
        not layers
        and all(
            output is None
            or (
                isinstance(node_values[output.index], ViewValue)
                and view_refs(node_values[output.index]) is not None
            )
            for output in function.output
        )
    )
    for output in function.output:
        if output is None:
            continue
        raw_value = node_values[output.index]
        value = resolve_view(raw_value)
        view_references = view_refs(raw_value)
        if (
            pure_view_outputs
            and isinstance(raw_value, ViewValue)
            and view_references is not None
        ):
            # ``refs`` describes the logical output independently of the
            # backing range.  Keep the range within the ABI workspace so the
            # archive validator does not mistake a four-element view of a
            # two-element input for an out-of-bounds output.
            node_specs[output.index] = MemorySpec(0, input_layer.dimension)
            continue
        if (
            view_references is not None
            and view_references
            == list(
                range(view_references[0], view_references[0] + len(view_references))
            )
        ):
            start = view_references[0]
            node_specs[output.index] = MemorySpec(start, len(view_references))
            node_values[output.index] = BilinearWeights.project(
                current_memory,
                node_specs[output.index],
                _node_shape(output.dim),
            )
            continue
        if isinstance(raw_value, ViewValue) and view_references is not None:
            value = resolve_view(raw_value)
            if isinstance(value, BilinearWeights):
                output_count = output.dim.flat()
                output_spec = MemorySpec(current_memory.count, output_count)
                next_memory = MemorySpec(
                    0, current_memory.count + output_count
                )
                value = value.extend_memory(current_memory)
                constant = dok_ndarray((next_memory.count,))
                linear = dok_ndarray((next_memory.count, current_memory.count))
                quadratic = dok_ndarray(
                    (next_memory.count, current_memory.count, current_memory.count)
                )
                _append_bilinear_value(
                    constant, linear, quadratic, output_spec.location, value,
                    _node_shape(output.dim),
                )
                layers.append(
                    BilinearWorkspaceLayer(
                        current_memory, next_memory,
                        BilinearWeights(
                            current_memory, (next_memory.count,),
                            constant=constant, linear=linear,
                            quadratic=quadratic,
                        ),
                        destination_rows=range(
                            output_spec.location,
                            output_spec.location + output_count,
                        ),
                    )
                )
                current_memory = next_memory
                current_size = next_memory.count
                node_specs[output.index] = output_spec
                node_values[output.index] = BilinearWeights.project(
                    current_memory, output_spec, _node_shape(output.dim)
                )
                continue
            # A final view may resolve to nonlinear BilinearWeights without a
            # residual lifetime of its own.  Give it a declared ABI slot and
            # materialize the resolved expression directly into that slot,
            # rather than queueing an un-lifetimed node for the frontier.
            if pending_nodes() or pending_generic is not None:
                materialize_boundary("output-abi")
            if output.index in node_specs:
                continue
            if not isinstance(value, BilinearWeights):
                value = _constant_to_bw(
                    current_memory, value, _node_shape(output.dim)
                )
            elif value.memory is not current_memory:
                value = value.extend_memory(current_memory)
            output_count = output.dim.flat()
            output_spec = MemorySpec(current_memory.count, output_count)
            next_memory = MemorySpec(
                0, current_memory.count + output_count
            )
            constant = dok_ndarray((next_memory.count,))
            linear = dok_ndarray((next_memory.count, current_memory.count))
            quadratic = dok_ndarray(
                (next_memory.count, current_memory.count, current_memory.count)
            )
            _append_bilinear_value(
                constant,
                linear,
                quadratic,
                output_spec.location,
                value,
                _node_shape(output.dim),
            )
            layers.append(
                BilinearWorkspaceLayer(

                    current_memory,
                    next_memory,
                    BilinearWeights(
                        current_memory,
                        (next_memory.count,),
                        constant=constant,
                        linear=linear,
                        quadratic=quadratic,
                    ),
                    destination_rows=range(
                        output_spec.location,
                        output_spec.location + output_count,
                    ),
                )
            )
            current_memory = next_memory
            current_size = next_memory.count
            node_specs[output.index] = output_spec
            node_values[output.index] = BilinearWeights.project(
                current_memory, output_spec, _node_shape(output.dim)
            )
            continue
        if output.index not in node_specs:
            queue_bilinear(output.index, value)
            materialize_boundary("output-abi")
            if output.index in node_specs:
                continue
        node_values[output.index] = value

    materialize_boundary("output-abi")

    output_layer = OutputLayer()
    output_weights = []
    for output in function.output:
        if output is None:
            continue
        output_value = node_values.get(output.index)
        output_refs = view_refs(output_value)
        output_layer.add_output(
            node_specs[output.index],
            output.dim,
            refs=output_refs if pure_view_outputs else None,
        )
        output_weights.append(output_value)

    merged_layers = []
    for layer in layers:
        previous_frontier = (
            getattr(merged_layers[-1], "_frontier_id", None)
            if merged_layers
            else None
        )
        generic_merge = (
            merged_layers
            and isinstance(merged_layers[-1], GenericVectorLayer)
            and isinstance(layer, GenericVectorLayer)
            and not merged_layers[-1].opaque_programs
            and not layer.opaque_programs
            and not merged_layers[-1].constants
            and not layer.constants
            and merged_layers[-1].memory_out == layer.memory_in
        )
        if generic_merge:
            previous = merged_layers[-1]
            previous_outputs = {operation[0] for operation in previous.ops}
            generic_hazard = any(
                reference >= 0 and reference in previous_outputs
                for _output, _op, first, second, third in layer.ops
                for reference in (first, second, third)
            )
            if not generic_hazard:
                merged_layers[-1] = GenericVectorLayer(
                    previous.memory_in,
                    layer.memory_out,
                    previous.ops + layer.ops,
                )
                continue
        layer_frontier = getattr(layer, "_frontier_id", None)
        frontier_reasons = tuple(boundary_reasons)
        nonlinear_boundary = lambda frontier_id: (
            frontier_id is not None
            and 0 < frontier_id <= len(frontier_reasons)
            and "nonlinear-scalar" in frontier_reasons[frontier_id - 1]
        )
        if (
            merged_layers
            and isinstance(merged_layers[-1], BilinearWorkspaceLayer)
            and isinstance(layer, BilinearWorkspaceLayer)
            and (
                previous_frontier == layer_frontier
                or previous_frontier is None
                or layer_frontier is None
                or not nonlinear_boundary(previous_frontier)
                and not nonlinear_boundary(layer_frontier)
            )
        ):
            previous = merged_layers[-1]
            previous_rows = set(previous._destination_rows)
            reads_previous = (
                any(
                    int(key[-1]) in previous_rows
                    or (
                        len(key) >= 2
                        and int(key[-2]) in previous_rows
                    )
                    for sparse in (layer.weights.linear, layer.weights.quadratic)
                    for key in sparse.keys.keys()
                )
            )
            if not reads_previous:
                common = MemorySpec(
                    0,
                    max(previous.memory_in.count, layer.memory_in.count),
                )
                combined = compiler_binary(
                    "add",
                    previous.weights.extend_memory(common),
                    layer.weights.extend_memory(common),
                )
                if combined is not None:
                    merged_layers[-1] = BilinearWorkspaceLayer(
                        common,
                        MemorySpec(
                            0,
                            max(previous.memory_out.count, layer.memory_out.count),
                        ),
                        combined,
                        destination_rows=(
                            previous_rows | set(layer._destination_rows)
                        ),
                    )
                    continue
        merged_layers.append(layer)
    layers = merged_layers
    workspace_count = max(
        [
            input_layer.dimension,
            current_memory.count,
            *(
                max(layer.memory_in.count, layer.memory_out.count)
                for layer in layers
            ),
        ]
    )

    graph = SparseNet(
        MemorySpec(0, workspace_count),
        input_layer,
        output_layer,
        layers,
    )
    graph.residual_lifetimes = residual_lifetimes
    graph.residual_workspace_size = max(
        (
            lifetime.slot.start + lifetime.slot.length
            for lifetime in residual_lifetimes.values()
        ),
        default=0,
    )
    graph.frontier_metadata = tuple(frontier_metadata)
    graph.output_weights = output_weights
    graph.materialization_boundary_reasons = tuple(boundary_reasons)
    graph.algebraic_frontier_count = sum(
        isinstance(layer, BilinearWorkspaceLayer) for layer in layers
    )
    graph.nonlinear_batch_count = sum(
        isinstance(layer, GenericVectorLayer) for layer in layers
    )
    graph.generic_flushes = tuple(generic_flushes)
    graph.generic_flush_cause_histogram = {
        cause: sum(record["cause"] == cause for record in generic_flushes)
        for cause in sorted({record["cause"] for record in generic_flushes})
    }
    graph.generic_flush_wave_histogram = {
        wave: sum(record["wave"] == wave for record in generic_flushes)
        for wave in sorted({
            record["wave"] for record in generic_flushes
            if record["wave"] is not None
        })
    }
    colored_barrier_dag = _colored_barrier_dag(semantic_dag, tape)
    graph.colored_barrier_dag = colored_barrier_dag
    graph.colored_barrier_lower_bound = colored_barrier_dag.lower_bound
    graph.scheduled_color_switches = colored_barrier_dag.color_switches
    graph.colored_barrier_theoretically_attainable_le_50 = (
        colored_barrier_dag.theoretically_attainable_le_50
    )
    graph.frontier_closure_reasons = tuple(boundary_reasons)
    # Frontier emission writes only algebraic destination rows.  It never
    # emits relocation, clear, or copy rows.
    graph.frontier_copy_rows = 0
    return graph


def _create_opgraph(
    function: Function,
    function_table_builder: _FunctionTableBuilder,
) -> SparseNet:
    return _create_residual_opgraph(function, function_table_builder)

def build_function_table(function: Function) -> FunctionTable:
    """Build the ordinary lowering module and own all nested programs."""
    return _FunctionTableBuilder().build(function)


def build_unfused_opgraph(function: Function) -> SparseNet:
    """Build the ordinary graph-lowering path for compatibility callers."""
    return build_function_table(function).entry
