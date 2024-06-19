from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Type
from coker import Tensor, Kernel

ArrayLike = Any


class Backend(metaclass=ABCMeta):

    @abstractmethod
    def native_types(self) -> Tuple[Type]:
        pass

    @abstractmethod
    def to_native(self, array: Tensor) -> ArrayLike:
        """Cast array from backend to numpy type."""
        pass

    @abstractmethod
    def from_native(self, array: ArrayLike) -> Tensor:
        """Cast array from native python (numpy) to backend type."""
        pass

    @abstractmethod
    def reshape(self, array: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
        pass

    @abstractmethod
    def call(self, op, *args) -> ArrayLike:
        pass

    def evaluate(self, kernel:Kernel, inputs: ArrayLike):
        from coker.backends.evaluator import evaluate_inner
        workspace = {}
        return evaluate_inner(kernel.tape, inputs, kernel.output, self, workspace)


__known_backends = {}


def instantiate_backend(name: str):
    if name == 'numpy':
        import coker.backends.numpy.core
        backend = coker.backends.numpy.core.NumpyBackend()
    elif name == 'coker':
        import coker.backends.coker
        backend = coker.backends.coker.CokerBackend()
    elif name == 'jax':
        import coker.backends.jax
        backend = coker.backends.jax.JaxBackend()
    elif name == 'casadi':
        import coker.backends.casadi
        backend = coker.backends.casadi.CasadiBackend()
    else:
        raise ValueError(f'Unknown backend: {name}')

    __backends[name] = backend
    return backend



def get_backend_by_name(name: str, set_current=True) -> Backend:
    try:
        b = __backends[name]
        if set_current:
            __current_backend = b
        return b
    except KeyError:
        pass
    try:
        b = instantiate_backend(name)
        if set_current:
            __current_backend = b
        return b
    except KeyError:
        pass

    raise NotImplementedError(f"Unknown backend {name}")


__backends = {}

default_backend = 'coker'
__current_backend = None


def get_current_backend() -> Backend:
    if __current_backend is None:
        return get_backend_by_name(default_backend)
    else:
        return __current_backend
