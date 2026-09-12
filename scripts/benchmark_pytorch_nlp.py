"""Opt-in CUDA benchmark corpus for the PyTorch nonlinear-program backend.

This script is intentionally not run by the test suite.  Run it explicitly on
CUDA hardware with ``python scripts/benchmark_pytorch_nlp.py --run``.
"""

from __future__ import annotations

import argparse
import time

import torch
from coker.backends.pytorch import PytorchNLPSolverOptions
from coker.toolkits.codesign import Minimise, ProblemBuilder
from coker.algebra.dimensions import Scalar, VectorSpace
from coker.dynamics import BoundedVariable, VariationalProblem
from coker.dynamics.system import create_autonomous_ode


def _problem(kind: str):
    if kind == "unconstrained":
        with ProblemBuilder(
            solver_options=PytorchNLPSolverOptions(warm_start=True)
        ) as builder:
            x = builder.new_variable("x", initial_value=3.0)
            builder.objective = Minimise((x - 1.0) ** 2)
            builder.outputs = [x]
            return builder.build("pytorch"), None
    if kind == "inequality":
        with ProblemBuilder() as builder:
            x = builder.new_variable("x", initial_value=0.2)
            builder.objective = Minimise((x - 1.0) ** 2)
            builder.constraints = [x * x <= 0.25]
            builder.outputs = [x]
            return builder.build("pytorch"), None
    if kind == "equality":
        with ProblemBuilder() as builder:
            x = builder.new_variable("x", initial_value=0.0)
            builder.objective = Minimise((x - 3.0) ** 2)
            builder.constraints = [x == 1.0]
            builder.outputs = [x]
            return builder.build("pytorch"), None
    if kind == "parameterized":
        with ProblemBuilder(arguments=[Scalar("target")]) as builder:
            target = builder.arguments[0]
            x = builder.new_variable("x", initial_value=0.0)
            builder.objective = Minimise((x - target) ** 2)
            builder.outputs = [x]
            return builder.build("pytorch"), (
                torch.tensor(2.0, device="cuda"),
            )
    raise ValueError(f"Unknown benchmark kind: {kind}")


def _variational_problem():
    parameters = VectorSpace("p", 1)
    system = create_autonomous_ode(
        x0=1.0,
        xdot=lambda x, p: p[0] * x,
        parameters=parameters,
        backend="pytorch",
    )
    rate = BoundedVariable("rate", -2.0, 2.0, guess=0.0)
    target = float(torch.exp(torch.tensor(0.7)).item())
    return VariationalProblem(
        loss=lambda solution, p: (solution(1.0, p) - target) ** 2,
        t_final=1.0,
        system=system,
        parameters=[rate],
        backend="pytorch",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run on CUDA")
    args = parser.parse_args()
    if not args.run:
        print("Benchmark not run; pass --run on a CUDA-capable host.")
        return
    if not torch.cuda.is_available():
        print("Benchmark not run: CUDA is unavailable.")
        return
    for kind in ("unconstrained", "inequality", "equality", "parameterized"):
        program, arguments = _problem(kind)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = program(*(arguments or ()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        memory = torch.cuda.max_memory_allocated()
        info = program.solve_info
        print(
            {
                "problem": kind,
                "objective": result[0],
                "constraint_violation": "see solve status",
                "stationarity": "see solve status",
                "status": getattr(info, "status", None),
                "time_seconds": elapsed,
                "peak_memory_bytes": memory,
            }
        )

    problem = _variational_problem()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    solution = problem.get_solver("pytorch").solve()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        {
            "problem": "neural_ode_variational",
            "fitted_rate": solution.parameter_solutions["rate"],
            "objective": solution.cost,
            "time_seconds": elapsed,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }
    )


if __name__ == "__main__":
    main()
