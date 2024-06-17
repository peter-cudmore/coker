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

    def evaluate(self, graph, inputs: ArrayLike, outputs: ArrayLike):
        from coker.backends.evaluator import evaluate_inner
        workspace = {}
        return evaluate_inner(graph, inputs, outputs, self, workspace)



__known_backends = {}

def coker_backend(name: str):
    def func(backend: Backend):
        __known_backends[name] = backend
        return backend
    return func

def instantiate_backend(name: str):
    if name == 'numpy':
        import coker.backends.numpy.core
        backend = coker.backends.numpy.core.NumpyBackend()
    elif name == 'jax':
        import coker.backends.jax
        backend = coker.backends.jax.JaxBackend()
    else:
        raise ValueError(f'Unknown backend: {name}')


    __backends[name] = backend
    return backend


def get_backend_by_name(name: str) -> Backend:

    try:
        return __backends[name]
    except KeyError:
        pass
    try:
        return instantiate_backend(name)
    except KeyError:
        pass

    raise NotImplementedError(f"Unknown backend {name}")


__backends = {}


def get_current_backend() -> Backend:
    return get_backend_by_name('numpy')

