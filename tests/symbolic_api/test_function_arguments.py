import pytest

from coker import Scalar, function


@pytest.mark.parametrize("arguments", [(), (1.0, 2.0)])
def test_call_rejects_wrong_argument_count(arguments):
    identity = function([Scalar("x")], lambda x: x, backend="numpy")

    with pytest.raises(TypeError, match="Expected 1 arguments"):
        identity(*arguments)


def test_traced_call_rejects_wrong_argument_count():
    identity = function([Scalar("x")], lambda x: x, backend="numpy")

    with pytest.raises(TypeError, match="Expected 1 arguments"):
        function([Scalar("x")], lambda x: identity(x, x), backend="numpy")


def test_factory_rejects_invalid_argument_space():
    with pytest.raises(TypeError, match="Invalid input type"):
        function([object()], lambda x: x, backend="numpy")
