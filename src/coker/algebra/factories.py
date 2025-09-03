import numpy as np
from coker.algebra.kernel import Tape


def zeros(shape: tuple):
    return Tape().insert_value(np.zeros(shape),)