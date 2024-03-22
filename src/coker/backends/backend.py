from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Type
from coker.algebra import Tensor

ArrayLike = Any


class Backend(metaclass=ABCMeta):

    @abstractmethod
    def native_types(self) -> Tuple[Type]:
        pass

    @abstractmethod
    def to_native(self, array: Tensor) -> ArrayLike:
        pass

    @abstractmethod
    def from_native(self, array: ArrayLike) -> Tensor:
        pass

    @abstractmethod
    def reshape(self, array: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
        pass

    @abstractmethod
    def call(self, op, *args) -> ArrayLike:
        pass


def get_backend_by_name(name: str) -> Backend:

    if name == 'numpy':
        try:
            return __backends['numpy']
        except KeyError:
            pass

        from coker.backends.numpy import NumpyBackend
        impl = NumpyBackend()
        __backends['numpy'] = impl
        return impl

    raise NotImplementedError(f"Unknown backend {name}")


__backends = {}


def get_current_backend() -> Backend:
    return get_backend_by_name('numpy')

