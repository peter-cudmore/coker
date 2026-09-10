from importlib.metadata import entry_points
import pytest


def test_bundled_backends_are_declared_as_plugins():
    plugins = entry_points(group="coker.backends")

    assert {plugin.name for plugin in plugins} >= {
        "numpy",
        "jax",
        "pytorch",
        "casadi",
        "sympy",
    }


def test_unknown_backend_has_stable_error():
    from coker.backends.backend import get_backend_by_name

    with pytest.raises(NotImplementedError, match="Unknown backend 'missing'"):
        get_backend_by_name("missing", set_current=False)
