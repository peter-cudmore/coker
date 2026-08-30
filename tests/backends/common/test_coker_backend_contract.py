import numpy as np

from coker import Scalar, function
from coker.backends.backend import get_backend_by_name
from coker_backend import CokerBackend, CokerFunction


def test_import_registers_coker_backend(coker_backend):
    assert isinstance(coker_backend, CokerBackend)
    assert get_backend_by_name("coker", set_current=False) is coker_backend


def test_coker_backend_converts_and_reshapes_arrays(coker_backend):
    values = np.array([1.0, 2.0], dtype=np.float32)

    backend_values = coker_backend.to_backend_array(values)

    assert np.array_equal(coker_backend.to_numpy_array(backend_values), values)
    assert np.array_equal(coker_backend.reshape(values, (2, 1)), [[1.0], [2.0]])
    assert np.ndarray in coker_backend.native_types()


def test_coker_backend_lowers_a_function(coker_backend):
    identity = function(
        arguments=[Scalar("x")],
        implementation=lambda x: x,
        backend="coker",
    )

    assert isinstance(coker_backend.lower(identity), CokerFunction)
