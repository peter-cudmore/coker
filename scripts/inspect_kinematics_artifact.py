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
        model.add_effector(parent, Isometry3(translation=np.array([0.2, 0.0, 0.0])))
    return model


def compile_hexapod():
    model = build_hexapod()
    lowered = function(
        [VectorSpace("q", model.total_joints())],
        implementation=model.to_function(),
        backend="coker",
    ).lower()
    payload = lowered.export_payload()
    return model, lowered, payload


def payload_metrics(payload: dict[str, object], workspace_f32: int, tangent_f32: int, *, require_v6: bool) -> dict[str, object]:
    programs = payload.get("functions", [])
    entry = programs[0]["program"]
    layers = entry.get("intermediate_layers", [])
    residual_stages = entry.get("residual_stages", ())
    kinds: dict[str, int] = {}
    generic_ops = bilinear_rows = bilinear_terms = explicit_copies = 0
    sparse_entries = identity_rows = bilinear_identity_rows = 0
    identity_groups = []
    output_clear_bytes = workspace_bytes_touched = output_clear_rows = 0
    frontier_closure_reasons: dict[str, int] = {}
    generic_layers = bilinear_layers = 0
    algebraic_frontier_count = nonlinear_batch_count = 0
    for layer_index, layer in enumerate(layers):
        kind = str(layer["kind"])
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind in {"generic", "scheduled_generic"}:
            generic_layers += 1
            nonlinear_batch_count += 1
        elif kind in {"bilinear", "scheduled_bilinear"}:
            bilinear_layers += 1
            algebraic_frontier_count += 1
        reason = layer.get("frontier_closure_reason")
        if reason:
            reason = str(reason)
            frontier_closure_reasons[reason] = frontier_closure_reasons.get(reason, 0) + 1
        memory_in = layer.get("memory_in", {})
        memory_out = layer.get("memory_out", {})
        workspace_bytes_touched += 4 * (
            int(memory_in.get("count", 0)) + int(memory_out.get("count", 0))
        )
        if kind in {"generic", "scheduled_generic"}:
            ops = layer.get("ops", [])
            generic_ops += len(ops)
            identity_rows += sum(
                (
                    (op.get("op", {}).get("value") if isinstance(op.get("op"), dict) else op.get("op"))
                    == "identity"
                )
                for op in ops
            )
        elif kind in {"bilinear", "scheduled_bilinear"}:
            if kind == "bilinear":
                entries = layer.get("weights", {}).get("entries", [])
                bilinear_rows += len({int(entry["index"][0]) for entry in entries})
                sparse_entries += len(entries)
                for entry in entries:
                    row, left, right = entry["index"]
                    if (
                        right == 0
                        and left
                        and float(entry["value"]) == 1.0
                        and int(memory_out.get("location", 0)) + int(row)
                        == int(memory_in.get("location", 0)) + int(left) - 1
                    ):
                        bilinear_identity_rows += 1
            else:
                rows, terms = layer.get("rows", []), layer.get("terms", [])
                bilinear_rows += len(rows)
                bilinear_terms += len(terms)
                layer_identity = 0
                for row in rows:
                    selected = terms[
                        int(row["term_start"]): int(row["term_start"]) + int(row["term_count"])
                    ]
                    if (
                        len(selected) == 1
                        and int(selected[0].get("right", 0)) == 0
                        and int(selected[0].get("left", 0)) == int(row["output"]) + 1
                        and float(selected[0].get("value", 0.0)) == 1.0
                    ):
                        bilinear_identity_rows += 1
                        layer_identity += 1
                        identity_groups.append((layer_index, kind, row, selected[0]))
                identity_rows += layer_identity
        elif kind in {"copy", "explicit_copy"}:
            explicit_copies += int(layer.get("count", 1))
        output_clear_rows += int(layer.get("output_clear_rows", layer.get("clear_rows", 0)))
        output_clear_bytes += 4 * int(layer.get("output_clear_bytes", 0))
    exported_reasons = entry.get("frontier_closure_reasons", ())
    if not frontier_closure_reasons and isinstance(exported_reasons, dict):
        for reason, count in exported_reasons.items():
            frontier_closure_reasons[str(reason)] = int(count)
    elif not frontier_closure_reasons:
        for reason in exported_reasons:
            reason = str(reason)
            frontier_closure_reasons[reason] = frontier_closure_reasons.get(reason, 0) + 1
    if require_v6 and (identity_rows or explicit_copies or output_clear_rows):
        raise AssertionError(
            "forbidden relocation/copy/output-clear rows: "
            f"identity={identity_rows}, copies={explicit_copies}, output_clear={output_clear_rows}"
        )
    return {
        "layer_count": len(layers),
        "residual_stage_count": len(residual_stages),
        "residual_copy_rows": sum(
            int(stage.get("copy_rows", stage.get("copies", 0)))
            for stage in residual_stages
            if isinstance(stage, dict)
        ),
        "residual_workspace_f32": max(int(workspace_f32), int(tangent_f32)),
        "algebraic_frontier_count": algebraic_frontier_count,
        "nonlinear_batch_count": nonlinear_batch_count,
        "generic_layer_count": generic_layers,
        "bilinear_layer_count": bilinear_layers,
        "layer_kinds": kinds,
        "frontier_closure_reasons": frontier_closure_reasons,
        "frontier_copy_rows": explicit_copies,
        "generic_arithmetic_rows": generic_ops,
        "identity_relocation_rows": identity_rows,
        "bilinear_identity_relocation_rows": bilinear_identity_rows,
        "explicit_copy_rows": explicit_copies,
        "bilinear_rows": bilinear_rows,
        "bilinear_terms": bilinear_terms + sparse_entries,
        "output_clear_rows": output_clear_rows,
        "output_clear_bytes": output_clear_bytes,
        "workspace_bytes_touched": workspace_bytes_touched,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.calls < 1 or args.warmup < 0:
        raise SystemExit("--calls must be positive and --warmup non-negative")

    model, lowered, payload = compile_hexapod()
    import coker._coker_runtime as runtime

    if hasattr(lowered, "compile_artifact"):
        artifact = lowered.compile_artifact(name="hexapod_kinematics", version="6")
    else:
        artifact = SimpleNamespace(data=bytes(lowered.compile_bytecode()))
    program = runtime.load_program(artifact.data)
    info = dict(program.info())
    q = np.linspace(-0.4, 0.4, model.total_joints(), dtype=np.float32)
    outputs = np.empty(sum(info["output_specs"]), dtype=np.float32)
    inputs = [q]
    execute = program.execute_into
    for _ in range(args.warmup):
        execute(inputs, outputs)
    samples = []
    for _ in range(args.calls):
        start = time.perf_counter_ns()
        execute(inputs, outputs)
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    if not np.all(np.isfinite(outputs)):
        raise RuntimeError("mapped artifact produced non-finite output")
    expected = np.concatenate(
        [np.asarray(value).reshape(-1) for value in model.to_function()(q)]
    ).astype(np.float32, copy=False)
    if not np.allclose(outputs, expected, rtol=1e-5, atol=1e-5):
        diff = np.abs(outputs - expected)
        max_index = int(np.argmax(diff))
        raise AssertionError(
            f"mapped artifact output differs: max_abs={float(diff[max_index])}, "
            f"max_rel={float(diff[max_index] / max(abs(float(expected[max_index])), 1e-6))}, "
            f"max_index={max_index}, actual={float(outputs[max_index])}, "
            f"expected={float(expected[max_index])}, output_specs={info['output_specs']}, "
            f"actual_nonzero={int(np.count_nonzero(outputs))}, "
            f"expected_nonzero={int(np.count_nonzero(expected))}, "
            f"output_shape={outputs.shape}, expected_shape={expected.shape}"
        )
    tangent = np.linspace(0.1, 0.6, model.total_joints(), dtype=np.float32)
    tangent_outputs = np.empty_like(outputs)
    if hasattr(program, "push_forward_into"):
        program.push_forward_into(inputs, [tangent], outputs, tangent_outputs)
        epsilon = 1e-3
        plus = np.concatenate(
            [
                np.asarray(value).reshape(-1)
                for value in model.to_function()(q + epsilon * tangent)
            ]
        )
        minus = np.concatenate(
            [
                np.asarray(value).reshape(-1)
                for value in model.to_function()(q - epsilon * tangent)
            ]
        )
        expected_tangent = ((plus - minus) / (2 * epsilon)).astype(np.float32)
        if not np.allclose(tangent_outputs, expected_tangent, rtol=2e-3, atol=2e-3):
            raise AssertionError("mapped artifact tangent differs from Python graph")

    metrics = payload_metrics(
        payload,
        int(info["required_workspace_size"]),
        int(info["required_workspace_size"]),
        require_v6=artifact.data[:8] == b"COKERB04",
    )
    metrics.update(
        {
            "artifact_bytes": len(artifact.data),
            "artifact_magic": artifact.data[:8].decode("ascii", errors="replace"),
            "artifact_version": int.from_bytes(artifact.data[8:10], "little"),
            "logical_primal_workspace_f32": int(info["workspace_size"]),
            "required_primal_workspace_f32": int(info["required_workspace_size"]),
            "logical_tangent_workspace_f32": int(info["workspace_size"]),
            "required_tangent_workspace_f32": int(info["required_workspace_size"]),
            "host_us_median": statistics.median(samples),
            "host_us_p95": float(np.percentile(samples, 95)),
            "host_us_max": max(samples),
            "calls": args.calls,
            "warmup": args.warmup,
            "command": "python scripts/inspect_kinematics_artifact.py --calls %d --warmup %d" % (args.calls, args.warmup),
        }
    )
    if artifact.data[:8] == b"COKERB04":
        metrics["fk_target"] = 50
        metrics["fk_target_met"] = metrics["layer_count"] < 50
        if not metrics["fk_target_met"]:
            metrics["dependency_diagnostic"] = tape_dependency_diagnostic(
                lowered.function.tape
            )
            metrics["interim_target"] = "under-100 (evidence only)"
            print(json.dumps(metrics, sort_keys=True))
            raise AssertionError(
                "FK scheduled layer budget target not met: "
                f"{metrics['layer_count']} >= 50; dependency diagnostic included"
            )
        if any(
            metrics[name]
            for name in (
                "identity_relocation_rows",
                "explicit_copy_rows",
                "output_clear_rows",
            )
        ):
            raise AssertionError("FK artifact contains forbidden relocation/copy/clear rows")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
