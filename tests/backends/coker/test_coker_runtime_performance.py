import time

import numpy as np
import pytest

from coker import VectorSpace, function

N_STATES = 8
N_LAYERS = 20
N_CALLS = 2000


rng = np.random.default_rng(0)
WEIGHTS = [
    rng.normal(size=(N_STATES, N_STATES)) * 0.1 for _ in range(N_LAYERS)
]
BIASES = [rng.normal(size=(N_STATES,)) * 0.1 for _ in range(N_LAYERS)]


def make_compiled_runtime_graph():
    symbolic_function = function(
        [VectorSpace("x", N_STATES)],
        implementation=lambda x: x,
        backend="coker",
    )
    return symbolic_function.lower()


@pytest.mark.perf
def test_runtime_inference_speed():
    compiled_graph = make_compiled_runtime_graph()
    input_vector = np.zeros(N_STATES)

    compiled_graph(input_vector)

    start = time.perf_counter()
    for _ in range(N_CALLS):
        compiled_graph(input_vector)
    elapsed = time.perf_counter() - start

    per_call_ms = elapsed / N_CALLS * 1000
    print(
        f"desktop Rust ordinary runtime: {elapsed:.3f}s for {N_CALLS} calls "
        f"({per_call_ms:.3f}ms/call)"
    )
    assert elapsed < 0.25, (
        f"runtime inference too slow: {elapsed:.3f}s for {N_CALLS} calls "
        f"({per_call_ms:.3f}ms/call)."
    )
