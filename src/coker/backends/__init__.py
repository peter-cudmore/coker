"""Coker Backends

Supported
- Numpy/scipy (default)

To add:
- Jax
- Casadi
- Pytorch

"""

from coker.backends.evaluator import evaluate
from coker.backends.backend import get_backend_by_name, get_current_backend
