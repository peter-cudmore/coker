import casadi as ca
import numpy as np
from typing import Tuple, Type

from coker import Tensor
from coker.backends.backend import Backend, ArrayLike


class CasadiBackend(Backend):
    def to_native(self, array: Tensor) -> ArrayLike:
        return ca.MX(array)

    def from_native(self, array: ArrayLike) -> Tensor:
        return Tensor.from_array(array)

    def call(self, op, *args) -> ArrayLike:
        pass

    def native_types(self) -> Tuple[Type]:
        pass

    def reshape(self, array: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
        pass
