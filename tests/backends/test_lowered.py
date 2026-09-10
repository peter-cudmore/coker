import numpy as np
import pytest

from coker import VectorSpace, function
from coker.backends.lowered import LoweredFunction


@pytest.mark.parametrize("backend_name", ["numpy", "casadi", "pytorch"])
def test_lowered_function_has_packed_and_positional_abis(backend_name):
    if backend_name == "casadi":
        pytest.importorskip("casadi")
    if backend_name == "pytorch":
        pytest.importorskip("torch")
    compiled = function(
        [VectorSpace("x", 2)],
        lambda x: (x + 1, x * 2),
        backend=backend_name,
    )
    lowered = compiled.lower()
    value = np.array([2.0, 3.0])

    assert isinstance(lowered, LoweredFunction)
    assert lowered.backend_name == backend_name
    assert [spec.name for spec in lowered.signature.inputs] == ["x"]
    assert [spec.name for spec in lowered.signature.outputs] == [
        "output_0",
        "output_1",
    ]
    assert lowered is compiled.lower()

    packed = lowered.execute([value])
    positional = lowered(value)
    assert isinstance(packed, tuple)
    assert isinstance(positional, tuple)
    np.testing.assert_allclose(packed[0], [3.0, 4.0])
    np.testing.assert_allclose(packed[1], [4.0, 6.0])
    np.testing.assert_allclose(positional[0], packed[0])
    np.testing.assert_allclose(positional[1], packed[1])

    public_result = compiled(value)
    assert isinstance(public_result, list)
    np.testing.assert_allclose(public_result[0], packed[0])
    np.testing.assert_allclose(public_result[1], packed[1])
