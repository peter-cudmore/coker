from importlib.metadata import entry_points


def test_bundled_backends_are_declared_as_plugins():
    plugins = entry_points(group="coker.backends")

    assert {plugin.name for plugin in plugins} >= {
        "numpy",
        "jax",
        "pytorch",
        "casadi",
        "sympy",
    }
