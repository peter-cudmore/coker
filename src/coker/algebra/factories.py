import numpy as np
from coker.algebra.kernel import TraceContext


def zeros(shape: tuple):
    tape = TraceContext.get_local_tape()
    assert tape is not None
    return tape.insert_value(np.zeros(shape))
