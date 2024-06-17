import numpy as np
import pytest

def is_close(a, b, tolerance=1e-16):
    assert a.shape == b.shape, "shapes don't match"
    return np.linalg.norm(a-b, ord=np.inf) < tolerance


def validate_symbolic_call(name, function, arguments, test_set):
    from coker.algebra.kernel import kernel

    f_test = kernel(implementation=function, arguments=arguments)

    for i, item in enumerate(test_set):
        expected = function(*item)
        result = f_test(*item)

        are_equal = all(is_close(e, r, 1e-6) for e, r in zip(expected, result))

        assert are_equal, f"Test {name} failed on item {i}: {item}"


def validate_symbolic_call_throws(function, arguments, value_exception_pairs):
    from coker.algebra.kernel import kernel

    f_test = kernel(implementation=function, arguments=arguments)

    for item, exception in value_exception_pairs:
        with pytest.raises(exception) as ex:
            _ = f_test(*item)