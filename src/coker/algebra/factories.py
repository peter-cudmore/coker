import numpy as np
from coker.algebra.kernel import TraceContext


def zeros(shape: tuple):
    """Create a zero-filled array node on the active tape.

    Must be called inside a tracing context (i.e. within an ``implementation``
    passed to :func:`~coker.algebra.kernel.function`).  The returned array
    behaves like a numpy array and supports index assignment of symbolic values.

    Args:
        shape: Shape of the array as a tuple of ints.

    Returns:
        A numpy array (or symbolic tensor) of zeros with the given shape.
    """
    tape = TraceContext.get_local_tape()
    assert tape is not None
    return tape.insert_value(np.zeros(shape))
