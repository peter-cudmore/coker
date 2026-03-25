import time
import numpy as np
from coker import function, VectorSpace

N_STATES = 10
N_OPS = 30  # produces ~100 tape nodes (matmul + add per iteration)


def make_dynamics_fn():
    """Synthetic ODE rhs with ~100 tape nodes."""
    weights = [np.random.randn(N_STATES, N_STATES) * 0.1 for _ in range(N_OPS)]
    biases = [np.random.randn(N_STATES) * 0.1 for _ in range(N_OPS)]

    def impl(x):
        for W, b in zip(weights, biases):
            x = W @ x + b
        return x

    return function([VectorSpace("x", N_STATES)], impl, backend="numpy")


def test_evaluate_inner_speed():
    """
    Once traced, calling a compiled function must NOT re-interpret the tape
    node-by-node on each invocation. 1000 calls to a ~100-node function
    should complete in < 100ms total.

    Currently FAILS: the interpreted loop takes ~1-5ms/call due to per-node
    isinstance checks and dict dispatch in evaluate_inner.
    """
    f = make_dynamics_fn()
    x0 = np.zeros(N_STATES)

    N_CALLS = 1000
    # warm-up
    f(x0)

    start = time.perf_counter()
    for _ in range(N_CALLS):
        f(x0)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1, (
        f"evaluate_inner too slow: {elapsed:.3f}s for {N_CALLS} calls "
        f"({elapsed / N_CALLS * 1000:.3f}ms/call). "
        f"Tape has {len(f.tape.nodes)} nodes. "
        f"Expected < 0.1ms/call once tape is compiled to a lambda."
    )
