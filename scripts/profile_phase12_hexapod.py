"""Reproducible desktop baseline for Phase 12 hexapod kinematics."""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEXAPOD_PY = Path(r"C:\projects\hexapod\hexapod_py")
sys.path[:0] = [str(ROOT / "src"), str(HEXAPOD_PY)]

from coker import VectorSpace, function  # noqa: E402
from coker_backend import CokerBackend  # noqa: E402
from hexapod.model import build_hexapod_model  # noqa: E402

REPETITIONS = 10
SAMPLES = 3


def make_functions():
    model = build_hexapod_model()
    angles = VectorSpace("angles", model.total_joints())

    def forward_kinematics(q):
        return np.concatenate(
            [x.translation for x in model.forward_kinematics(q)]
        )

    def forward_spatial_jacobian(q):
        return np.concatenate(model.spatial_manipulator_jacobian(q), axis=0)

    return (
        (
            "forward_kinematics",
            function([angles], forward_kinematics, backend="coker"),
        ),
        (
            "forward_spatial_jacobian",
            function([angles], forward_spatial_jacobian, backend="coker"),
        ),
    )


def profile(name, source, input_value):
    compiled = CokerBackend().lower(source)
    compile_start = time.perf_counter_ns()
    compiled(input_value)
    first_call_ns = time.perf_counter_ns() - compile_start
    artifact = compiled._artifact
    sample_ns = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        for _ in range(REPETITIONS):
            compiled(input_value)
        sample_ns.append((time.perf_counter_ns() - start) / REPETITIONS)
    return {
        "name": name,
        "inputs": [int(value) for value in artifact.info()["input_specs"]],
        "outputs": [int(value) for value in artifact.info()["output_specs"]],
        "workspace_f32": int(artifact.info()["workspace_size"]),
        "archive_bytes": len(artifact.to_bytes()),
        "compile_and_first_call_ns": first_call_ns,
        "repeated_call_ns_median": statistics.median(sample_ns),
        "repeated_call_ns_min": min(sample_ns),
        "repetitions_per_sample": REPETITIONS,
        "samples": SAMPLES,
    }


def main():
    input_value = np.zeros(24, dtype=np.float32)
    results = [
        profile(name, source, input_value) for name, source in make_functions()
    ]
    print(
        json.dumps(
            {
                "machine": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                },
                "workload": {
                    "model": str(HEXAPOD_PY / "hexapod" / "model.py"),
                    "input_value": (
                        "24 f32 zeros "
                        "(free-body pose plus six three-joint legs)"
                    ),
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
