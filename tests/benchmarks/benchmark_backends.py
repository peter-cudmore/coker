"""
Backend performance benchmarks.

Benchmark 1 — Function evaluation: chain of matmul + nonlinear ops across backends
Benchmark 2 — ODE integration: Van der Pol oscillator across backends and solvers
Benchmark 3 — Variational problem: parameter fitting (CasADi only)

Run with:
    python tests/benchmarks/benchmark_backends.py
"""

import time
import numpy as np

import coker
from coker import function, VectorSpace, Scalar
from coker.dynamics import create_autonomous_ode
from coker.dynamics.types import VariationalProblem, BoundedVariable
from coker.backends.backend import get_backend_by_name
from coker.backends.numpy.core import NumpySolverParameters, Solver


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def timeit(fn, n_warmup=2, n_calls=50):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fn()
    return (time.perf_counter() - t0) / n_calls


def fmt_time(s):
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def print_results(title, subtitle, results):
    results = sorted(results, key=lambda r: r[1])
    fastest = results[0][1]

    name_w = max(len(r[0]) for r in results) + 2
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'-' * 60}")
    print(f"  {'Rank':<6}{'Backend':<{name_w}}{'Time':<12}{'vs fastest'}")
    print(f"  {'-'*4}  {'-'*(name_w-2)}  {'-'*10}  {'-'*10}")
    for rank, (name, t, note) in enumerate(results, 1):
        ratio = f"{t / fastest:.1f}×" if rank > 1 else "—"
        note_str = f"  ({note})" if note else ""
        print(f"  {rank:<6}{name:<{name_w}}{fmt_time(t):<12}{ratio}{note_str}")


# ---------------------------------------------------------------------------
# Benchmark 1: Function evaluation
# ---------------------------------------------------------------------------

N_STATES = 16
N_LAYERS = 24  # produces ~170 tape nodes; nonlinear every 4 layers


def bench_function_eval():
    np.random.seed(42)
    weights = [
        np.random.randn(N_STATES, N_STATES) * 0.1 for _ in range(N_LAYERS)
    ]
    biases = [np.random.randn(N_STATES) * 0.1 for _ in range(N_LAYERS)]

    def impl(x):
        for i, (W, b) in enumerate(zip(weights, biases)):
            x = W @ x + b
            if i % 4 == 0:
                x = np.sin(x)
        return x

    x0 = np.random.randn(N_STATES)
    startup_results = []
    call_results = []

    for backend_name in ("numpy", "casadi", "coker"):
        try:
            # Startup: time from fresh Function object to first result
            t0 = time.perf_counter()
            f = function(
                [VectorSpace("x", N_STATES)], impl, backend=backend_name
            )
            f(x0)
            startup = time.perf_counter() - t0
            startup_results.append((backend_name, startup, None))

            # Repeat calls: already compiled, measure steady-state throughput
            t = timeit(lambda: f(x0), n_warmup=5, n_calls=1000)
            call_results.append((backend_name, t, None))
        except Exception as e:
            msg = str(e)[:40]
            startup_results.append((backend_name, float("inf"), msg))
            call_results.append((backend_name, float("inf"), msg))

    subtitle = (
        f"{N_STATES}-state, {N_LAYERS} layers (matmul + bias + sin every 4)"
    )
    print_results(
        "Benchmark 1a: Function evaluation — startup cost",
        subtitle,
        startup_results,
    )
    print_results(
        "Benchmark 1b: Function evaluation — repeat calls (1000)",
        subtitle,
        call_results,
    )


# ---------------------------------------------------------------------------
# Benchmark 2: ODE integration — Van der Pol oscillator
# ---------------------------------------------------------------------------

MU = 2.0  # mild stiffness; makes LSODA/Radau relevant
T_FINAL = 10.0
N_ODE_CALLS = 20


def bench_ode():
    x0_val = np.array([2.0, 0.0])
    p_val = np.array([MU])

    def xdot(x, p):
        return np.array([x[1], p[0] * (1.0 - x[0] ** 2) * x[1] - x[0]])

    # -- numpy variants --------------------------------------------------------
    system_np = create_autonomous_ode(
        x0=x0_val,
        xdot=xdot,
        parameters=VectorSpace("p", 1),
        backend="numpy",
    )
    np_backend = get_backend_by_name("numpy")
    x0, z0 = system_np.x0(0, None, p_val)

    results = []
    for solver in (Solver.RK45, Solver.LSODA, Solver.Radau):
        sp = NumpySolverParameters(solver)
        try:
            np_backend.evaluate_integrals(
                [system_np.dxdt, system_np.g, system_np.dqdt],
                [x0, z0, None],
                T_FINAL,
                [None, p_val],
                solver_parameters=sp,
            )
            t = timeit(
                lambda sp=sp: np_backend.evaluate_integrals(
                    [system_np.dxdt, system_np.g, system_np.dqdt],
                    [x0, z0, None],
                    T_FINAL,
                    [None, p_val],
                    solver_parameters=sp,
                ),
                n_warmup=2,
                n_calls=N_ODE_CALLS,
            )
            results.append((f"numpy / {solver.value}", t, None))
        except Exception as e:
            results.append(
                (f"numpy / {solver.value}", float("inf"), str(e)[:40])
            )

    # -- casadi ----------------------------------------------------------------
    try:
        system_ca = create_autonomous_ode(
            x0=x0_val,
            xdot=xdot,
            parameters=VectorSpace("p", 1),
            backend="casadi",
        )
        system_ca(T_FINAL, p_val)  # warm-up (builds ca.Function on first call)
        t = timeit(
            lambda: system_ca(T_FINAL, p_val),
            n_warmup=2,
            n_calls=N_ODE_CALLS,
        )
        results.append(("casadi / IDAS", t, None))
    except Exception as e:
        results.append(("casadi / IDAS", float("inf"), str(e)[:40]))

    print_results(
        "Benchmark 2: ODE integration",
        f"Van der Pol µ={MU}, t=0 to {T_FINAL}, {N_ODE_CALLS} solves",
        results,
    )


# ---------------------------------------------------------------------------
# Benchmark 3: VariationalProblem — exponential decay parameter fitting
# ---------------------------------------------------------------------------

N_VAR_CALLS = 5


def bench_variational():
    a_true, c_true = 1.5, 3.0
    t_data = np.linspace(0, 2, 12)
    y_data = c_true * np.exp(-a_true * t_data)

    def x0(p):
        return p[1:2]

    def xdot(x, p):
        return -p[0:1] * x

    system = create_autonomous_ode(
        x0=x0,
        xdot=xdot,
        parameters=VectorSpace("p", 2),
        backend="numpy",
    )

    def loss(f, p):
        return sum((f(t_i, p) - y_i) ** 2 for t_i, y_i in zip(t_data, y_data))

    problem = VariationalProblem(
        loss=loss,
        system=system,
        parameters=[
            BoundedVariable("a", lower_bound=0.1, upper_bound=5.0, guess=1.0),
            BoundedVariable("c", lower_bound=0.1, upper_bound=10.0, guess=1.0),
        ],
        t_final=2.0,
        backend="casadi",
    )

    results = []
    try:
        sol = problem()  # warm-up
        t = timeit(lambda: problem(), n_warmup=0, n_calls=N_VAR_CALLS)
        results.append(
            (
                "casadi",
                t,
                f"a={sol.parameter_solutions['a']:.3f} (true {a_true}), "
                f"c={sol.parameter_solutions['c']:.3f} (true {c_true})",
            )
        )
    except Exception as e:
        results.append(("casadi", float("inf"), str(e)[:60]))

    print_results(
        "Benchmark 3: VariationalProblem",
        f"Exponential decay fitting, {N_VAR_CALLS} solves  "
        f"(CasADi is the only supported backend)",
        results,
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nCoker backend benchmarks  —  NumPy {np.__version__}")
    bench_function_eval()
    bench_ode()
    bench_variational()
    print()
