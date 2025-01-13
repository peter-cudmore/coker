import numpy as np
import pytest


def is_close(a, b, tolerance=1e-16):

    if not isinstance(a, np.ndarray) and not isinstance(b, np.ndarray):
        return abs(a - b) < tolerance

    assert a.shape == b.shape, "shapes don't match"


    try:
        return np.linalg.norm(a - b, ord=np.inf) < tolerance
    except ValueError:
        pass





def validate_symbolic_call(name, function, arguments, test_set, backend):
    from coker.algebra.kernel import kernel

    f_test = kernel(implementation=function, arguments=arguments, backend=backend)

    for i, item in enumerate(test_set):
        expected = function(*item)
        result = f_test(*item)
        try:
            are_equal = all(is_close(e, r, 1e-6) for e, r in zip(expected, result))
        except ValueError as ex:
            ex.add_note(
                f"Expected: {expected}\n Result: {result}"
            )
            raise ex

        assert are_equal, f"Test {name} failed on item {i}: {item}\n Expected: {expected}\nGot: {result}"


def validate_symbolic_call_throws(function, arguments, value_exception_pairs, backend):
    from coker.algebra.kernel import kernel

    f_test = kernel(implementation=function, arguments=arguments, backend=backend)

    for item, exception in value_exception_pairs:
        with pytest.raises(exception) as ex:
            _ = f_test(*item)