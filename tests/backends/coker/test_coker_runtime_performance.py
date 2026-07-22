import time

import numpy as np
import pytest

from coker import VectorSpace, function
from coker.backends.coker.core import create_opgraph
from coker.backends.coker.runtime import CompiledGraph

N_STATES = 8
N_LAYERS = 20
N_CALLS = 2000


rng = np.random.default_rng(0)
WEIGHTS = [rng.normal(size=(N_STATES, N_STATES)) * 0.1 for _ in range(N_LAYERS)]
BIASES = [rng.normal(size=(N_STATES,)) * 0.1 for _ in range(N_LAYERS)]


def make_compiled_runtime_graph():
    def implementation(x):
        for weight_matrix, bias_vector in zip(WEIGHTS, BIASES):
            x = weight_matrix @ x + bias_vector
        return x

    symbolic_function = function(  # pyright: ignore[reportArgumentType]
        [VectorSpace("x", N_STATES)], implementation=implementation
    )
    return CompiledGraph.compile(create_opgraph(symbolic_function))


@pytest.mark.perf
def test_runtime_inference_speed():
    compiled_graph = make_compiled_runtime_graph()
    input_vector = np.zeros(N_STATES)

    compiled_graph(input_vector)

    start = time.perf_counter()
    for _ in range(N_CALLS):
        compiled_graph(input_vector)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.25, (
        f"runtime inference too slow: {elapsed:.3f}s for {N_CALLS} calls "
        f"({elapsed / N_CALLS * 1000:.3f}ms/call). "
        f"Expected < 0.125ms/call on the compiled runtime path."
    )
