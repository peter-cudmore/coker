import pytest


@pytest.fixture
def coker_backend():
    import coker_backend
    from coker.backends.backend import get_backend_by_name

    return get_backend_by_name("coker", set_current=False)
