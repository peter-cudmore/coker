import numpy as np

from coker import Scalar, VectorSpace, function
from coker.backends.coker.core import create_opgraph
from coker.backends.coker.runtime import CompiledGraph


def test_scalar_vector_bilinear_products_preserve_output_rows():
    symbolic_function = function(
        [
            VectorSpace("lift_off", 3),
            VectorSpace("touchdown", 3),
            Scalar("phase"),
            Scalar("offset"),
        ],
        implementation=lambda lift_off, touchdown, phase, offset: (
            (1.0 - phase) * lift_off
            + phase * touchdown
            + offset * np.array([0.0, 0.0, 1.0])
        ),
    )
    args = (
        np.array([0.2, -0.1, -0.3]),
        np.array([0.8, 0.4, -0.2]),
        0.5,
        0.15,
    )
    expected = symbolic_function(*args)
    graph = create_opgraph(symbolic_function)

    assert graph.intermediate_layers[-1].memory_out.count == 3
    np.testing.assert_allclose(graph(*args), expected)
    np.testing.assert_allclose(CompiledGraph.compile(graph)(*args), expected)
