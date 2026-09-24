from __future__ import annotations


from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from coker.algebra.function import Function
    from coker.dynamics.variational.problem import VariationalProblem
from coker.backends.evaluator import Evaluator
from coker.backends.lowered import LoweredFunction, LoweringOptions
from coker.algebra.graph import Tape, Tracer
from coker.algebra.ops import Noop, OP, SelectOP
from coker.algebra.dimensions import Dimension, FunctionSpace
from coker.interfaces import SolverParameters

ArrayLike = Any


def split_function_parameter_values(declaration: Any, flat_values: Any):
    """Split flat backend values into the declaration's concrete blocks."""
    from coker.parameters import BoundedVariable, UnboundedVariable

    parameters = []
    offset = 0
    for concrete in declaration.list_concrete_parameters():
        is_scalar = isinstance(concrete, (BoundedVariable, UnboundedVariable))
        size = 1 if is_scalar else concrete.size
        block = flat_values[offset : offset + size]
        if block.shape[0] != size:
            raise ValueError(
                "solver decisions do not match function parameter declarations"
            )
        parameters.append(
            block[0] if is_scalar else block.reshape(concrete.shape)
        )
        offset += size
    if offset != flat_values.shape[0]:
        raise ValueError(
            "solver decisions do not match function parameter declarations"
        )
    return tuple(parameters)


class Backend(metaclass=ABCMeta):
    name: str

    @abstractmethod
    def to_numpy_array(self, array) -> ArrayLike:
        """Cast array from backend to numpy type."""
        pass

    @abstractmethod
    def to_backend_array(self, array: ArrayLike):
        """Cast array from native python (numpy) to backend type."""
        pass

    @abstractmethod
    def reshape(self, array: ArrayLike, shape: Dimension) -> ArrayLike:
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
        *,
        options=None,
    ):
        raise NotImplementedError

    def make_optimisation_module(self, implementation):
        """Wrap a backend solver for use as a numerical program module."""
        return implementation

    @abstractmethod
    def fit_function_parameter(
        self, declaration: Any, target: Any, values: ArrayLike
    ) -> Any:
        """Materialize a fitted function from backend decision values."""

    def create_variational_solver(
        self, problem: VariationalProblem
    ) -> VariationalSolver:
        raise NotImplementedError

    @abstractmethod
    def get_evaluator(self) -> Evaluator:
        """Return this backend's compiled-plan evaluator."""

    def evaluate(
        self, function: Function, inputs: Sequence[Any]
    ) -> list[Any | None]:
        from coker.backends.evaluator import evaluate_inner

        workspace: dict[int, Any] = {}
        return evaluate_inner(
            function.tape, inputs, function.output, self, workspace
        )

    def compose(
        self,
        function: Function,
        inputs: Sequence[Any],
        outer_tape: Tape,
    ) -> list[Tracer | None]:
        """Record a Coker function-table call on ``outer_tape``."""
        if len(inputs) != len(function.tape.input_indicies):
            raise TypeError(
                f"Expected {len(function.tape.input_indicies)} inputs, got "
                f"{len(inputs)}"
            )
        if (
            function._native_callable is not None
            and function.backend != self.name
        ):
            raise RuntimeError(
                "Cannot compose native callable for backend "
                f"{function.backend!r} into {self.name!r} trace"
            )

        from coker.algebra.function import BoundCallable, Function

        arguments = []
        for value, spec in zip(inputs, function.signature.inputs):
            expected_space = spec.space
            if expected_space is None or isinstance(expected_space, Noop):
                continue
            if isinstance(value, BoundCallable):
                arguments.append(outer_tape._create_function_reference(value))
                continue
            if not isinstance(value, Function):
                arguments.append(value)
                continue
            if isinstance(expected_space, FunctionSpace) and (
                len(expected_space.arguments)
                == sum(
                    input_spec.space is not None
                    and not isinstance(input_spec.space, Noop)
                    for input_spec in value.signature.inputs
                )
            ):
                value = BoundCallable(value, expected_space, ())
            arguments.append(outer_tape._create_function_reference(value))
        arguments = tuple(arguments)
        reference = outer_tape._create_function_reference(function)
        bundle = Tracer(
            outer_tape,
            outer_tape.append(OP.EVALUATE, reference, *arguments),
        )
        present_output_count = sum(
            output.shape is not None for output in function.signature.outputs
        )
        result: list[Tracer | None] = []
        output_index = 0
        for output in function.signature.outputs:
            if output.shape is None:
                result.append(None)
            else:
                result.append(
                    bundle
                    if present_output_count == 1
                    else Tracer(
                        outer_tape,
                        outer_tape.append(SelectOP(output_index), bundle),
                    )
                )
                output_index += 1
        return result

    def evaluate_integrals(
        self,
        functions,
        initial_conditions,
        end_point: float,
        inputs,
        solver_parameters: SolverParameters | None = None,
    ):
        raise NotImplementedError(
            "Evaluating integrals is not implemented for this backend"
        )

    @abstractmethod
    def lower(
        self,
        function: Function,
        options: LoweringOptions | None = None,
    ) -> LoweredFunction:
        """Return this backend's concrete lowered execution handle."""


class VariationalSolver:
    """Interface definition for variational solvers."""

    @property
    def parameters(self) -> List[str]:
        """List the free optimisation parameter names."""
        raise NotImplementedError(
            "Subclasses must implement parameters property"
        )

    def solve(self, **kwargs):
        """Solve the variational problem.

        Optionally fix parameters by name.
        """
        raise NotImplementedError("Subclasses must implement solve")

    def __call__(self, **kwargs):
        return self.solve(**kwargs)


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """Registers an importable backend factory under ``name``.

    Registering the same factory more than once is harmless. Registering a
    different factory for an existing name fails so backend selection cannot
    depend on import order.
    """
    existing = __registered_backends.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(f"Backend {name!r} is already registered")
    __registered_backends[name] = factory


__registered_backends: dict[str, Callable[[], Backend]] = {}


def _entry_points_for_backend(name: str):
    return tuple(
        entry_point
        for entry_point in entry_points(group="coker.backends")
        if entry_point.name == name
    )


def _discover_backend(name: str) -> None:
    plugins = _entry_points_for_backend(name)
    if not plugins:
        raise NotImplementedError(f"Unknown backend {name!r}")
    if len(plugins) > 1:
        raise ValueError(f"Backend {name!r} has multiple plugin registrations")

    register_backend(name, plugins[0].load())


def instantiate_backend(name: str):
    factory = __registered_backends.get(name)
    if factory is None:
        _discover_backend(name)
        factory = __registered_backends[name]
    backend = factory()

    # Backend identity is part of the callable/lowering contract.
    backend.name = name
    __backends[name] = backend
    return backend


def get_backend_by_name(name: str, set_current=True) -> Backend:
    global __current_backend

    backend = __backends.get(name)
    if backend is None:
        backend = instantiate_backend(name)

    if set_current:
        __current_backend = backend
    return backend


__backends = {}

default_backend = "coker"
__current_backend = None


def get_current_backend() -> Backend:
    if __current_backend is None:
        return get_backend_by_name(default_backend)
    else:
        return __current_backend
