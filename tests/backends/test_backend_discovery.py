import pytest

from coker.backends import get_backend_by_name, get_current_backend


def test_unknown_backend_has_stable_error():
    with pytest.raises(NotImplementedError, match="Unknown backend 'missing'"):
        get_backend_by_name("missing", set_current=False)


def test_backend_lookup_only_selects_when_requested(backend):
    original = get_current_backend()
    try:
        selected = get_backend_by_name("numpy", set_current=True)

        resolved = get_backend_by_name(backend)
        assert get_current_backend() is selected

        assert get_backend_by_name(backend, set_current=True) is resolved
        assert get_current_backend() is resolved
    finally:
        get_backend_by_name(original.name, set_current=True)
