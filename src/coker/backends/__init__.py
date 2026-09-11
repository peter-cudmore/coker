"""Backend registry and lazy interpreter entry point."""

from coker.backends.backend import (
    get_backend_by_name,
    get_current_backend,
    register_backend,
)


def evaluate(*args, **kwargs):
    """Evaluate through the interpreter without eager interpreter imports."""
    from coker.backends.evaluator import evaluate as evaluate_function

    return evaluate_function(*args, **kwargs)


__all__ = [
    "evaluate",
    "get_backend_by_name",
    "get_current_backend",
    "register_backend",
]
