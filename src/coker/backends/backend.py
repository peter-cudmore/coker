from __future__ import annotations


from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from importlib.metadata import entry_points
from importlib.util import find_spec
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
    Scalar,
    VectorSpace,
)
from coker.interfaces import SolverParameters

ArrayLike = Any


def create_native_symbol_entry(
    target: Callable[..., Any],
    function_space: FunctionSpace,
    *,
    name: str | None = None,
) -> SymbolEntry:
    """Describe a backend-native target for opaque tape interning."""
    return SymbolEntry(
        target,
        function_space,
        name,
        (
            "native",
            id(target),
            id(function_space),
            name,
        ),
    )


def _restore_output_slots(
    tape: Tape,
    result: Tracer,
    output_specs: Sequence[FunctionOutputSpec],
) -> list[Tracer | None]:
    """Expand compact graph outputs into their declared ABI slots."""
    output_count = sum(output_spec.shape is not None for output_spec in output_specs)
    present_results = (
        (result,)
        if output_count == 1
        else tuple(
            Tracer(tape, tape.append(SelectOP(index), result))
            for index in range(output_count)
        )
    )
    present_results_iter = iter(present_results)
    return [
        (None if output_spec.shape is None else next(present_results_iter))
        for output_spec in output_specs
    ]


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

    output_indices = tuple(
        index
        for index, output_spec in enumerate(output_specs)
        if output_spec.shape is not None
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

    def compact_native(*native_args: Any) -> Any:
        native_outputs = native(*native_args)
        values = (native_outputs,) if len(output_specs) == 1 else tuple(native_outputs)
        if len(values) != len(output_specs):
            raise ValueError(
                f"Native callable returned {len(values)} results; expected "
                f"{len(output_specs)}"
            )
        present_outputs = tuple(values[index] for index in output_indices)
        return present_outputs[0] if len(present_outputs) == 1 else present_outputs

    function_space = FunctionSpace(
        f"{backend}_native",
        arguments=list(input_spaces),
        output=output_spaces,
    )
    symbol = tape.intern_symbol(
        create_native_symbol_entry(
            compact_native,
            function_space,
            name=name,
        )
    )
    result = Tracer(tape, tape.append(OP.EVALUATE, symbol, *args))
    return _restore_output_slots(tape, result, output_specs)


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
        parameters.append(block[0] if is_scalar else block.reshape(concrete.shape))
        offset += size
    if offset != flat_values.shape[0]:
        raise ValueError(
            "solver decisions do not match function parameter declarations"
        )
    return tuple(parameters)


class Backend(metaclass=ABCMeta):
    """Define conversion, lowering and advertised solver extension hooks.

    Backends are constructed lazily by the public registry. Implementations
    share this interface, not a universal numeric value or solver-result type.
    Optional solve paths must raise ``NotImplementedError`` when unavailable.

    Example:
        >>> import numpy as np
        >>> from coker.backends import get_backend_by_name
        >>> backend = get_backend_by_name("numpy")
        >>> np.testing.assert_allclose(
        ...     backend.to_numpy_array(backend.to_backend_array([1., 2.])),
        ...     [1., 2.],
        ... )
    """

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
        """Convert a value to the declared numeric dimension.

        Args:
            array: Backend-compatible scalar or array.
            shape: Required Coker dimension; scalar and length-one vector
                dimensions are distinct.

        Returns:
            A backend value with the declared shape. Conversion and invalid
            size/type errors are backend-specific.
        """
        pass

    @abstractmethod
    def call(self, op, *args) -> ArrayLike:
        """Apply a primitive operation to backend values.

        Args:
            op: Compiler operation recognised by the backend.
            *args: Its ordered native operands.

        Returns:
            The backend-native operation result.
        """
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
        """Compile a nonlinear program without solving it.

        Args:
            cost: Scalar symbolic objective.
            constraints: Symbolic comparisons or bounded residuals.
            parameters: Ordered runtime input tracers, excluding decisions.
            outputs: Objective-first requested outputs and private captures.
            initial_conditions: Decision-index-to-initial-value mapping.
            options: Backend-compatible solver configuration, or ``None``.

        Returns:
            A backend implementation accepted by ``make_optimisation_module``.

        Raises:
            NotImplementedError: Nonlinear-program construction is unsupported.
        """
        raise NotImplementedError

    def make_optimisation_module(self, implementation):
        """Wrap a backend solver for use as a numerical program module."""
        return implementation

    def _materialize_parameter(
        self,
        target: Scalar | VectorSpace | FunctionSpace,
        declaration: Any,
        blocks: tuple[ArrayLike, ...],
    ) -> Any:
        """Reconstruct one public parameter from native solver blocks."""
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot materialize parameters"
        )

    def create_variational_solver(
        self, problem: VariationalProblem
    ) -> VariationalSolver:
        """Create this backend's solver for a variational problem."""
        raise NotImplementedError

    @abstractmethod
    def get_evaluator(self) -> Evaluator:
        """Return this backend's compiled-plan evaluator."""

    def evaluate(self, function: Function, inputs: Sequence[Any]) -> list[Any | None]:
        """Evaluate a graph through this backend's interpreter.

        Args:
            function: Coker function whose graph supplies input/output metadata.
            inputs: Values in declared input order.

        Returns:
            An ordered list of public outputs, retaining absent ``None`` slots.
        """
        return self.get_evaluator().evaluate(function, inputs)

    def append_native_call(
        self,
        function: Function,
        inputs: Sequence[Any],
        outer_tape: Tape,
    ) -> list[Tracer | None] | None:
        """Append a backend-specific native call, if this backend owns it."""
        if function._native_callable is not None and function.backend != self.name:
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
        """Integrate a system through a backend-specific initial-value solver.

        Args:
            functions: Ordered derivative, algebraic residual and quadrature
                callables.
            initial_conditions: Ordered differential, algebraic and quadrature
                initial values, using ``None`` for absent channels.
            end_point: Final physical time, integrating from zero.
            inputs: Ordered control-function and parameter values.
            solver_parameters: Compatible backend solver settings, or ``None``.

        Returns:
            Differential, algebraic and quadrature endpoint values. Native
            representation and supported channels depend on the backend.

        Raises:
            NotImplementedError: The backend has no initial-value integrator.
        """
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
        raise NotImplementedError("Subclasses must implement parameters property")

    def solve(self, **kwargs):
        """Solve the variational problem.

        Optionally fix parameters by name.
        """
        raise NotImplementedError("Subclasses must implement solve")

    def __call__(self, **kwargs):
        return self.solve(**kwargs)


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """Register an extension backend factory under an extensible string name.

    Args:
        name: Registry/plugin name used by ``function`` and solvers.
        factory: Zero-argument callable returning a ``Backend`` instance.

    Raises:
        ValueError: Another factory is already registered under this name.

    Note:
        Re-registering the same factory is harmless. Registration does not
        instantiate a backend or select it as the ambient backend.
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

    plugin = plugins[0]
    try:
        factory = plugin.load()
    except ModuleNotFoundError as error:
        dependency = {
            "casadi": "casadi",
            "jax": "jax",
            "pytorch": "torch",
        }.get(name)
        # Only diagnose an absent top-level dependency of a built-in plugin.
        # Installed but broken packages and third-party plugins keep their
        # original import diagnostics.
        if (
            dependency is None
            or error.name != dependency
            or plugin.dist is None
            or plugin.dist.name != "coker"
            or find_spec(dependency) is not None
        ):
            raise
        requirement = " (requires Python >=3.11)" if name == "jax" else ""
        raise ImportError(
            f"Backend {name!r} requires optional dependency {dependency!r}. "
            f'Install it with pip install "coker[{name}]"{requirement}.'
        ) from error
    register_backend(name, factory)


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


def get_backend_by_name(name: str, set_current: bool = False) -> Backend:
    """Resolve and cache a backend without changing the ambient selection.

    Args:
        name: Registered name or installed ``coker.backends`` entry-point name.
        set_current: Select this instance as the ambient backend when true.
            Defaults to false.

    Returns:
        The process-wide cached backend instance. Mutating backend device,
        dtype, or other options affects future consumers of this instance.

    Raises:
        NotImplementedError: No registry factory or plugin has this name.
        ValueError: Multiple plugins register the same name.
        ImportError: A selected plugin's optional dependency is unavailable.

    Example:
        Resolve NumPy with ``get_backend_by_name("numpy")``; select it for
        ambient consumers with ``get_backend_by_name("numpy", set_current=True)``.
    """
    global __current_backend

    backend = __backends.get(name)
    if backend is None:
        backend = instantiate_backend(name)

    if set_current:
        __current_backend = backend
    return backend


__backends = {}

default_backend = "numpy"
__current_backend = None


def get_current_backend() -> Backend:
    """Return the ambient backend, resolving the default on first use.

    Returns:
        The current cached backend, or the default ``"numpy"`` instance when
        no backend has been explicitly selected. Looking up or lowering a
        function for another backend does not change this selection.

    Example:
        Use ``get_current_backend().name`` to inspect the ambient selection.
    """
    if __current_backend is None:
        return get_backend_by_name(default_backend)
    else:
        return __current_backend
