from __future__ import annotations


from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from coker.algebra.function import Function
    from coker.dynamics.variational.problem import VariationalProblem
from coker.backends.evaluator import Evaluator
from coker.backends.lowered import (
    FunctionOutputSpec,
    FunctionSignature,
    LoweredFunction,
    LoweringOptions,
)
from coker.algebra.graph import SymbolEntry, Tape, TraceContext, Tracer
from coker.algebra.ops import OP, SelectOP

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    ResultBundleDimension,
    Scalar,
    VectorSpace,
)
from coker.interfaces import SolverParameters

ArrayLike = Any


def create_native_symbol_entry(
    target: Callable[..., Any],
    function_space: FunctionSpace,
    result_dimension: (
        Dimension | FunctionSpace | ResultBundleDimension | None
    ) = None,
    *,
    name: str | None = None,
) -> SymbolEntry:
    """Describe a backend-native target for opaque tape interning."""
    if result_dimension is None:
        output_dimensions = function_space.output_dimensions()
        if len(output_dimensions) != 1:
            raise ValueError(
                "Native symbols require one result or an explicit result "
                "dimension"
            )
        (result_dimension,) = output_dimensions
    return SymbolEntry(
        target,
        function_space,
        result_dimension,
        name,
        (
            "native",
            id(target),
            id(function_space),
            id(result_dimension),
            name,
        ),
    )


def append_native_outputs(
    tape: Tape,
    native: Callable[..., Any],
    backend: str,
    input_spaces: Sequence[Scalar | VectorSpace | FunctionSpace],
    output_specs: Sequence[FunctionOutputSpec],
    args: Sequence[Any],
    name: str | None = None,
) -> list[Tracer | None]:
    """Emit one backend-native evaluation and select its public outputs."""

    def result_output_dimension(
        shape: Dimension | FunctionSpace | Scalar | VectorSpace | None,
    ) -> Dimension | FunctionSpace | None:
        if shape is None or isinstance(shape, (Dimension, FunctionSpace)):
            return shape
        if isinstance(shape, Scalar):
            return Dimension.scalar()
        if isinstance(shape, VectorSpace):
            return Dimension(shape.dimension)
        raise TypeError(f"Unsupported native output shape {shape!r}")

    result_dimension = ResultBundleDimension(
        tuple(
            result_output_dimension(output_spec.shape)
            for output_spec in output_specs
        )
    )
    output_spaces = [
        (
            output_spec.shape.to_space(output_spec.name)
            if isinstance(output_spec.shape, Dimension)
            else output_spec.shape
        )
        for output_spec in output_specs
        if output_spec.shape is not None
    ]
    function_space = FunctionSpace(
        f"{backend}_native",
        arguments=list(input_spaces),
        output=output_spaces,
    )
    symbol = tape.intern_symbol(
        create_native_symbol_entry(
            native,
            function_space,
            result_dimension,
            name=name,
        )
    )
    bundle = Tracer(tape, tape.append(OP.EVALUATE, symbol, *args))
    return [
        (
            None
            if output_spec.shape is None
            else Tracer(tape, tape.append(SelectOP(output_index), bundle))
        )
        for output_index, output_spec in enumerate(output_specs)
    ]


def import_native_function(
    native: Callable[..., Any],
    signature: FunctionSignature,
    *,
    backend: str,
    name: str | None = None,
):
    """Build a traceable Coker function around a backend-native callable."""
    from coker.algebra.function import Function

    if not isinstance(signature, FunctionSignature):
        raise TypeError("signature must be a FunctionSignature")

    input_spaces = [spec.space for spec in signature.inputs]
    with TraceContext(backend=backend) as tape:
        args = [tape.input(space) for space in input_spaces]
        outputs = append_native_outputs(
            tape,
            native,
            backend,
            input_spaces,
            signature.outputs,
            args,
            name=name,
        )

    result = Function(
        tape,
        outputs[0] if len(outputs) == 1 else outputs,
        backend=backend,
        name=name,
        signature=signature,
    )
    result._native_callable = native
    return result


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

    def materialize_parameter(
        self,
        target: Scalar | VectorSpace | FunctionSpace,
        declaration: Any,
        blocks: tuple[ArrayLike, ...],
    ) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot materialize parameters"
        )

    def fit_function_parameter(
        self, declaration: Any, target: Any, values: ArrayLike
    ) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot materialize function parameters"
        )

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
        return self.get_evaluator().evaluate(function, inputs)

    def append_native_call(
        self,
        function: Function,
        inputs: Sequence[Any],
        outer_tape: Tape,
    ) -> list[Tracer | None] | None:
        """Append a backend-specific native call, if this backend owns it."""
        if (
            function._native_callable is not None
            and function.backend != self.name
        ):
            raise RuntimeError(
                "Cannot compose native callable for backend "
                f"{function.backend!r} into {self.name!r} trace"
            )
        return None

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
