import numpy as np
import pytest
from coker.toolkits.spatial import Isometry3, Rotation3


def is_close(a, b, tolerance=1e-8):

    if isinstance(a, Isometry3) and isinstance(b, Isometry3):
        return is_close(a.translation, b.translation, tolerance) and is_close(
            a.rotation, b.rotation, tolerance
        )

    if isinstance(a, Rotation3) and isinstance(b, Rotation3):
        return is_close(a.as_vector(), b.as_vector(), tolerance)

    if not isinstance(a, np.ndarray) and not isinstance(b, np.ndarray):
        return abs(a - b) < tolerance

    assert a.shape == b.shape, "shapes don't match"

    try:
        return np.linalg.norm(a - b, ord=np.inf) < tolerance
    except ValueError:
        pass

    return abs(float(a) - float(b)) < tolerance


def validate_symbolic_call(name, f, arguments, test_set, backend):
    from coker.algebra.kernel import function

    f_test = function(implementation=f, arguments=arguments, backend=backend)

    for i, item in enumerate(test_set):
        expected = f(*item)
        result = f_test(*item)
        try:
            are_equal = all(
                is_close(e, r, 1e-6) for e, r in zip(expected, result)
            )
        except ValueError as ex:
            ex.add_note(f"Expected: {expected}\n Result: {result}")
            raise ex

        assert (
            are_equal
        ), f"Test {name} failed on item {i}: {item}\n Expected: {expected}\nGot: {result}"


def validate_symbolic_call_throws(
    f, arguments, value_exception_pairs, backend
):
    from coker.algebra.kernel import function

    f_test = function(implementation=f, arguments=arguments, backend=backend)

    for item, exception in value_exception_pairs:
        with pytest.raises(exception) as ex:
            _ = f_test(*item)
