from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Type
from coker.algebra.kernel import Function, Tracer
from typing import List, Dict

from coker.dynamics import (
    VariationalProblem,
    create_autonomous_ode,
    DynamicsSpec,
)

ArrayLike = Any


class SolverParameters(metaclass=ABCMeta):
    """Interface for backend-specific ODE solver configuration."""

    pass


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

    def resolve_fn(self, op):
        """Return a callable for op, resolved at plan-build time.

        The default wraps backend.call; backends can override to return the
        underlying function directly and avoid the per-call dispatch overhead.
        """
        _op = op
        return lambda *args: self.call(_op, *args)

    def resolve_post_fn(self, dim):
        """Return the post-processing function applied to a step's output.

        The default handles the general case (Tracer passthrough + reshape).
        Backends can override to return an identity function where safe, removing
        the isinstance check and reshape call from the hot loop.
        """
        _dim = dim
        _reshape = self.reshape

        def post(value):
            if not isinstance(value, Tracer):
                return _reshape(value, _dim)
            return value

        return post

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
        """Compile function to a callable for repeated numerical evaluation.

        Called once on the first concrete (non-Tracer) invocation of
        Function.__call__. Returns a callable f(inputs) -> outputs where
        inputs and outputs are lists matching the function's signature.

        The default wraps evaluate_inner; backends override this to return an
        optimised compiled callable (e.g. a plan-based closure for numpy, a
        ca.Function for casadi).
        """
        backend = self
        tape = function.tape
        outputs = function.output

        def compiled(inputs):
            from coker.backends.evaluator import evaluate_inner

            workspace = {}
            return evaluate_inner(tape, inputs, outputs, backend, workspace)

        return compiled


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
