from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Type
from coker import Function
from coker import Tracer
from typing import List, Dict

from coker.dynamics import (
    VariationalProblem,
    create_autonomous_ode,
    DynamicsSpec,
)

ArrayLike = Any


class Backend(metaclass=ABCMeta):

    @abstractmethod
    def native_types(self) -> Tuple[Type]:
        pass

    @abstractmethod
    def to_numpy_array(self, array) -> ArrayLike:
        """Cast array from backend to numpy type."""
        pass

    @abstractmethod
    def to_backend_array(self, array: ArrayLike):
        """Cast array from native python (numpy) to backend type."""
        pass

    @abstractmethod
    def reshape(self, array: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
        pass

    @abstractmethod
    def call(self, op, *args) -> ArrayLike:
        pass

    @abstractmethod
    def build_optimisation_problem(
        self,
        cost: Tracer,  # cost
        constraints: List[Tracer],
        parameters: List[Tracer],
        outputs: List[Tracer],
        initial_conditions: Dict[int, ArrayLike],
    ):
        raise NotImplementedError

    def create_variational_solver(self, problem: VariationalProblem):
        raise NotImplementedError

    def evaluate(self, function: Function, inputs: ArrayLike):
        from coker.backends.evaluator import evaluate_inner

        workspace = {}
        return evaluate_inner(
            function.tape, inputs, function.output, self, workspace
        )

    def evaluate_integrals(
        self,
        functions,
        initial_conditions,
        end_point: float,
        inputs,
        solver_parameters=None,
    ):
        raise NotImplementedError(
            "Evaluating integrals is not implemented for this backend"
        )

    def lower(self, function: Function):
        raise NotImplementedError(
            "lowering is not implemented for this backend"
        )


__known_backends = {}


def instantiate_backend(name: str):
    if name == "numpy":
        import coker.backends.numpy.core

        backend = coker.backends.numpy.core.NumpyBackend()
    elif name == "coker":
        import coker.backends.coker

        backend = coker.backends.coker.CokerBackend()
    elif name == "jax":
        import coker.backends.jax

        backend = coker.backends.jax.JaxBackend()
    elif name == "casadi":
        import coker.backends.casadi

        backend = coker.backends.casadi.CasadiBackend()
    elif name == "sympy":
        import coker.backends.sympy

        backend = coker.backends.sympy.SympyBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")

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

default_backend = "coker"
__current_backend = None


def get_current_backend() -> Backend:
    if __current_backend is None:
        return get_backend_by_name(default_backend)
    else:
        return __current_backend
