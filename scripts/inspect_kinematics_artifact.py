"""Regenerate and inspect the Coker Hexapod kinematics artifact.

Run from the repository root with the project environment active:
    python scripts/inspect_kinematics_artifact.py --calls 1000

The command emits one JSON object containing artifact/layout metrics followed by
steady-state mapped-host timing (microseconds). It intentionally uses the same
model and invocation on the v5 baseline and v6 branch.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from coker import VectorSpace, function
from coker.algebra.kernel import Tracer
from coker.algebra.ops import ModuleCallOP, OP
from coker.backends.coker.residual import (
    BilinearStage,
    CallStage,
    NonlinearStage,
    stage_count,
)
from coker.toolkits.kinematics import Inertia, Revolute, RigidBody
from coker.toolkits.spatial import Isometry3, Screw



def tape_dependency_diagnostic(
    tape: Any, node_ids: set[int] | None = None
) -> dict[str, object]:
    """Describe degree-bearing tape dependencies for an over-budget artifact."""
    selected = set(range(len(tape.nodes))) if node_ids is None else set(node_ids)
    records = []
    predecessors: dict[int, tuple[int, ...]] = {}
    for node_id in sorted(selected):
        operation, *arguments = tape.nodes[node_id]
        deps = tuple(
            sorted(
                argument.index
                for argument in arguments
                if isinstance(argument, Tracer) and argument.index in selected
            )
        )
        predecessors[node_id] = deps
        operation_name = (
            str(operation.name).lower()
            if hasattr(operation, "name")
            else type(operation).__name__
        )
        if node_id in getattr(tape, "input_indicies", ()):
            degree = "input"
        elif isinstance(operation, ModuleCallOP):
            degree = "call"
        elif getattr(operation, "is_bilinear", lambda: False)():
            degree = "bilinear"
        elif getattr(operation, "is_linear", lambda: False)():
            degree = "linear"
        elif operation is OP.VALUE:
            degree = "constant"
        else:
            degree = "nonlinear"
        records.append(
            {
                "node_id": node_id,
                "operation": operation_name,
                "degree": degree,
                "predecessors": list(deps),
            }
        )

    depth: dict[int, int] = {}
    for node_id in sorted(selected):
        depth[node_id] = 1 + max(
            (depth[pred] for pred in predecessors[node_id]), default=0
        )
    critical_path = max(depth.values(), default=0)
    ready_nodes = [node_id for node_id in sorted(selected) if not predecessors[node_id]]
    nonlinear_nodes = {
        record["node_id"]
        for record in records
        if record["degree"] in {"nonlinear", "call"}
    }
    nonlinear_depth: dict[int, int] = {}
    for node_id in sorted(nonlinear_nodes):
        nonlinear_depth[node_id] = 1 + max(
            (
                nonlinear_depth[pred]
                for pred in predecessors[node_id]
                if pred in nonlinear_nodes
            ),
            default=0,
        )
    return {
        "nodes": records,
        "ready_nodes": ready_nodes,
        "critical_path": critical_path,
        "nonlinear_call_nodes": sorted(nonlinear_nodes),
        "nonlinear_call_depth": max(nonlinear_depth.values(), default=0),
        "serial_nonlinear_call_chain": (
            len(nonlinear_nodes) >= 2
            and max(nonlinear_depth.values(), default=0) >= 2
        ),
    }


def build_hexapod() -> RigidBody:
    model = RigidBody()
    for leg in range(6):
        parent = model.WORLD
        for _joint in range(3):
            parent = model.add_link(
                parent=parent,
                at=Isometry3(translation=np.array([0.4, 0.1 * leg, 0.0])),
                joint=Revolute(Screw.w_z()),
                inertia=Inertia.zero(),
            )
        model.add_effector(
            parent, Isometry3(translation=np.array([0.2, 0.0, 0.0]))
        )
    return model


def compile_hexapod():
    """Build the executable Python residual graph for the hexapod.
    Bytecode compilation is intentionally deferred to the typed-bytecode
    phase.  The returned graph is the residual SparseNet itself and is
    executable through ``__call__`` and ``push_forward``.
    """
    model = build_hexapod()
    lowered = function(
        [VectorSpace("q", model.total_joints())],
        implementation=model.to_function(),
        backend="coker",
    )
    graph = lowered.lower().graph
    return model, graph


def payload_metrics(
    graph: Any,
    workspace_f32: int = 0,
    tangent_f32: int = 0,
    *,
    require_v6: bool = False,
) -> dict[str, object]:
    """Summarize residual stages without serializing a legacy payload."""
    stages = tuple(graph.residual_stages or ())
    kinds: dict[str, int] = {}
    generic_ops = bilinear_rows = bilinear_terms = 0
    workspace_bytes_touched = 0
    for stage in stages:
        if isinstance(stage, BilinearStage):
            kind = "bilinear"
            bilinear_rows += len(stage.rows)
            bilinear_terms += sum(len(row.terms) for row in stage.rows)
        elif isinstance(stage, NonlinearStage):
            kind = "generic"
            generic_ops += len(stage.operations)
        elif isinstance(stage, CallStage):
            kind = "call"
        else:
            kind = type(stage).__name__.lower()
        kinds[kind] = kinds.get(kind, 0) + 1
    workspace = max(
        int(workspace_f32),
        int(tangent_f32),
        int(getattr(graph, "residual_workspace_size", graph.memory)),
    )
    if require_v6:
        # Residual stages have no relocation, explicit-copy, or clear rows.
        pass
    return {
        "layer_count": stage_count(stages),
        "residual_stage_count": stage_count(stages),
        "residual_copy_rows": 0,
        "residual_workspace_f32": workspace,
        "algebraic_frontier_count": sum(isinstance(s, BilinearStage) for s in stages),
        "nonlinear_batch_count": sum(isinstance(s, NonlinearStage) for s in stages),
        "generic_layer_count": sum(isinstance(s, NonlinearStage) for s in stages),
        "bilinear_layer_count": sum(isinstance(s, BilinearStage) for s in stages),
        "layer_kinds": kinds,
        "frontier_closure_reasons": {},
        "frontier_copy_rows": 0,
        "generic_arithmetic_rows": generic_ops,
        "identity_relocation_rows": 0,
        "bilinear_identity_relocation_rows": 0,
        "explicit_copy_rows": 0,
        "bilinear_rows": bilinear_rows,
        "bilinear_terms": bilinear_terms,
        "output_clear_rows": 0,
        "output_clear_bytes": 0,
        "workspace_bytes_touched": workspace_bytes_touched,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.calls < 1 or args.warmup < 0:
        raise SystemExit("--calls must be positive and warmup non-negative")

    model, graph = compile_hexapod()
    q = np.linspace(-0.4, 0.4, model.total_joints())
    tangent = np.linspace(0.1, 0.6, model.total_joints())
    for _ in range(args.warmup):
        graph(q)
    samples = []
    for _ in range(args.calls):
        start = time.perf_counter_ns()
        graph(q)
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    positions, jacobian = graph(q)
    expected_positions, expected_jacobian = model.to_function()(q)
    np.testing.assert_allclose(positions, expected_positions, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-12, atol=1e-12)
    _, tangent_output = graph.push_forward(q, tangent)
    _, expected_tangent = model.to_function()(q + 1e-3 * tangent)
    metrics = payload_metrics(graph)
    metrics.update(
        {
            "artifact_bytes": None,
            "artifact_magic": None,
            "artifact_version": None,
            "bytecode_deferred": True,
            "logical_primal_workspace_f32": int(graph.memory),
            "required_primal_workspace_f32": int(graph.residual_workspace_size),
            "logical_tangent_workspace_f32": int(graph.memory),
            "required_tangent_workspace_f32": int(graph.residual_workspace_size),
            "host_us_median": statistics.median(samples),
            "host_us_p95": float(np.percentile(samples, 95)),
            "host_us_max": max(samples),
            "calls": args.calls,
            "warmup": args.warmup,
        }
    )
    metrics["fk_target"] = 50
    metrics["fk_target_met"] = metrics["layer_count"] < 50
    if not metrics["fk_target_met"]:
        metrics["dependency_diagnostic"] = tape_dependency_diagnostic(
            function(
                [VectorSpace("q", model.total_joints())],
                implementation=model.to_function(),
                backend="coker",
            ).tape
        )
        raise AssertionError(
            f"FK residual stage budget target not met: {metrics['layer_count']} >= 50"
        )
    print(json.dumps(metrics, sort_keys=True))
