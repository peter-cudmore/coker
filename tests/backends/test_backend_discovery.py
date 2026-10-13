from importlib.metadata import entry_points
import pytest


def test_unknown_backend_has_stable_error():
    from coker.backends.backend import get_backend_by_name

    with pytest.raises(NotImplementedError, match="Unknown backend 'missing'"):
        get_backend_by_name("missing", set_current=False)


def test_default_backend_works_with_only_base_numpy_plugin(monkeypatch):
    import coker.backends.backend as backend_module

    numpy_plugins = tuple(
        plugin
        for plugin in entry_points(group="coker.backends")
        if plugin.name == "numpy"
    )
    monkeypatch.setattr(
        backend_module, "entry_points", lambda **kwargs: numpy_plugins
    )
    monkeypatch.setattr(backend_module, "__registered_backends", {})
    monkeypatch.setattr(backend_module, "__backends", {})
    monkeypatch.setattr(backend_module, "__current_backend", None)

    assert backend_module.get_current_backend().name == "numpy"


def test_backend_lookup_only_selects_when_requested(monkeypatch):
    import coker.backends.backend as backend_module

    monkeypatch.setattr(backend_module, "__current_backend", None)
    numpy_backend = backend_module.get_backend_by_name(
        "numpy", set_current=True
    )

    sympy_backend = backend_module.get_backend_by_name("sympy")
    assert backend_module.get_current_backend() is numpy_backend

    assert backend_module.get_backend_by_name(
        "sympy", set_current=True
    ) is sympy_backend
    assert backend_module.get_current_backend() is sympy_backend
