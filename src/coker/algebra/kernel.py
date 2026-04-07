import weakref

import numpy as np
import scipy as sp
from typing import Callable, Union, Tuple, List, Optional, Set, Iterable, Any
from collections import defaultdict

from coker.algebra.dimensions import (
    Dimension,
    VectorSpace,
    Scalar,
    FunctionSpace,
    Element,
)
from coker.algebra.tensor import SymbolicVector
from coker.algebra.ops import OP, Noop, Operator

from coker.algebra.ops import numpy_atomics, numpy_composites

import threading

scalar_types = (
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    float,
    complex,
    int,
)


def get_basis(dimension: Dimension, i: int):
    return np.array([1 if j == i else 0 for j in range(dimension.dim[0])])


def get_projection(dimension: Dimension, slc: slice):
    if isinstance(dimension.dim, tuple):
        cols = dimension.dim[0]
    else:
        return 1

    indices = list(range(cols))[slc]
    rows = len(indices)
    proj = np.zeros((rows, cols), dtype=float)
    for row, col in enumerate(indices):
        proj[row, col] = 1
    return proj


def get_dim_by_class(arg):
    if isinstance(arg, scalar_types):
        return Dimension(None)
    try:

        d = Dimension(arg.shape)
        return d
    except (AttributeError, ValueError):
        pass

    raise NotImplementedError(f"Don't know the shape of {type(arg)}")


class DanglingTracerError(Exception):
    def __init__(self, *args, tracers: List["Tracer"]):
        super().__init__(*args)
        self.tracers = tracers


def _find_closure_tracers(fn) -> dict:
    """Return all Tracer objects captured in fn's closure.

    Returns a dict keyed by (tape_id, tracer_index) so duplicates are collapsed.
    """
    captured = {}
    if not (hasattr(fn, "__code__") and fn.__closure__):
        return captured
    for cell in fn.__closure__:
        try:
            val = cell.cell_contents
            if isinstance(val, Tracer):
                captured[(id(val.tape), val.index)] = val
        except ValueError:
            pass
    return captured


class TapeInner:
    INNER_REF = -1
    CONSTANT_REF = -2

    def __init__(self, tape_ref: "Tape"):
        self._nodes = []
        self._constants = []
        self._constant_hashmap = {}
        self.tape_ref = weakref.ref(tape_ref)
        assert self.INNER_REF not in OP.__members__.values()
        assert self.CONSTANT_REF not in OP.__members__.values()

    @staticmethod
    def constant_hash(value) -> int:
        if isinstance(value, (int, float)):
            return hash((0, 0, value))
        elif isinstance(value, np.ndarray):
            return hash((*value.shape, *value.flatten().tolist()))
        else:
            return hash(value)

    @staticmethod
    def try_sparsify(value: Any) -> Any:
        if not isinstance(value, np.ndarray):
            return value

        n_entries: int = value.flatten().shape[0]

        if len(value.shape) == 1 or n_entries <= 16:
            return value

        nnz: int = np.count_nonzero(value)
        density = nnz / n_entries
        threshold = 0.25

        if density > threshold:
            return value
        try:
            return sp.sparse.csc_array(value)
        except ValueError:
            return value

    def push_op(self, op: OP, *args) -> int:
        assert isinstance(op, OP) or isinstance(op, Operator)
        idx = len(self._nodes)

        if op == OP.VALUE:
            (value,) = args

            hsh = self.constant_hash(value)
            if hsh in self._constant_hashmap:
                value_idx = self._constant_hashmap[hsh]
            else:
                value_idx = len(self._constants)
                self._constant_hashmap[hsh] = value_idx
                value = self.try_sparsify(value)
                self._constants.append(value)

            self._nodes.append((self.CONSTANT_REF, value_idx))
        else:
            self._nodes.append((op, *args))
        return idx

    def define_input(self, size: int) -> int:
        idx = len(self._nodes)
        assert isinstance(size, int)
        self._nodes.append((self.INNER_REF, idx, size))
        return idx

    def __getitem__(self, item):
        op, *args = self._nodes[item]
        if op == self.INNER_REF:
            idx, size = args
            return Tracer(self.tape_ref(), idx)
        if op == self.CONSTANT_REF:
            (value_idx,) = args
            return OP.VALUE, self._constants[value_idx]
        return op, *args

    def __len__(self):
        return len(self._nodes)


class Tape:
    NONE = -1
    MAP_TO_NONE = -2

    def __init__(self):
        self._inner = TapeInner(self)
        self.dim = []
        self.input_indicies = []
        self.input_names = []
        self._substitutions: dict = {}

    def add_substitution(self, foreign: "Tracer", local: "Tracer"):
        """Register a rewrite rule: any occurrence of foreign in append args is replaced by local."""
        self._substitutions[(id(foreign.tape), foreign.index)] = local

    def op(self, i):
        return self.nodes[i][0]

    @property
    def nodes(self):
        return self._inner

    def find_dependents(self, tracer: "Tracer") -> Set[int]:
        if tracer is None or tracer is Noop():
            return set()

        index = tracer.index
        result = set()
        for i, inner in enumerate(self.nodes[tracer.index :]):
            if isinstance(inner, Tracer):
                continue
            op, *args = inner
            if op == OP.VALUE:
                continue
            for arg in args:
                if isinstance(arg, Tracer):
                    if arg.index == index or arg.index in result:
                        result.add(index + i)
        return result

    def inputs(self):
        for index in self.input_indicies:
            if index == Tape.NONE:
                yield None
            elif index == Tape.MAP_TO_NONE:
                yield Noop()
            else:
                yield Tracer(self, index)

    def __len__(self):
        return len(self.nodes)

    def __hash__(self):
        return id(self)

    def _compute_shape(self, op: OP, *args) -> Dimension:
        dims = []
        for arg in args:
            if arg is None:
                dims.append(None)
                continue

            assert isinstance(arg, Tracer)

            dims.append(arg.dim)

        return op.compute_shape(*dims)

    def append(self, op: OP, *args) -> int:
        args = [strip_symbols_from_array(a) for a in args]

        if self._substitutions:
            args = [
                (
                    self._substitutions.get((id(a.tape), a.index), a)
                    if isinstance(a, Tracer)
                    else a
                )
                for a in args
            ]

        invalid_tracers = [
            arg for arg in args if isinstance(arg, Tracer) and arg.tape != self
        ]
        if invalid_tracers:
            raise DanglingTracerError(tracers=invalid_tracers)

        args = [
            self.insert_value(a) if not isinstance(a, Tracer) else a.copy()
            for a in args
        ]

        out_dim = self._compute_shape(op, *args)
        index = len(self.dim)
        self.nodes.push_op(op, *args)
        self.dim.append(out_dim)
        return index

    def insert_value(self, arg):
        if arg is None:
            return None
        assert not isinstance(arg, Tracer)
        dim = get_dim_by_class(arg)
        idx = len(self.dim)
        self.nodes.push_op(OP.VALUE, arg)
        self.dim.append(dim)
        return Tracer(self, idx)

    def input(self, v: VectorSpace | Scalar):
        if v is None:
            self.input_indicies.append(Tape.NONE)
            self.input_names.append("None")
            return None

        if isinstance(v, Noop):
            self.input_indicies.append(Tape.MAP_TO_NONE)
            self.input_names.append("Noop")
            return v

        if isinstance(v, VectorSpace):
            dimension = Dimension(v.dimension)
            self.dim.append(dimension)
            size = dimension.flat()
        elif isinstance(v, Scalar):
            self.dim.append(Dimension(None))
            size = 1
        elif isinstance(v, FunctionSpace):
            self.dim.append(v)
            size = sum([d.flat() for d in v.output_dimensions()])
        else:
            assert False, f"Invalid input type {v}: of {type(v)} "

        index = self.nodes.define_input(size)
        tracer = Tracer(self, index)

        self.input_indicies.append(index)
        self.input_names.append(v.name)
        return tracer

    def list_inputs(self) -> Iterable[None | Noop | Scalar | VectorSpace]:
        for arg_idx, node_idx in enumerate(self.input_indicies):
            if node_idx == Tape.NONE:
                yield None
            elif node_idx == Tape.MAP_TO_NONE:
                yield Noop()
            elif isinstance(self.dim[node_idx], FunctionSpace):
                yield self.dim[node_idx]
            else:
                dim: Dimension = self.dim[node_idx]
                name = self.input_names[arg_idx]
                yield dim.to_space(name)

    def substitute(self, index, value):
        assert index in self.input_indicies

        self.input_indicies.remove(index)
        self.nodes[index] = value


def is_additive_identity(space: Dimension, arg) -> bool:
    if isinstance(arg, scalar_types) and arg == 0:
        return True
    try:
        return (space.dim == arg.shape) and (arg == 0).all()
    except:
        pass

    return False


class Tracer(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, tape: Tape, index: int):
        self._tape = weakref.ref(tape)
        self.index = index

    @property
    def tape(self):
        return self._tape()

    def _active_tape(self) -> "Tape":
        """Return the tape that operations on this Tracer should be recorded on.

        During tracing the current TraceContext tape is returned so that all ops
        land on the active tape (which may differ from self.tape when a closure
        captures a tracer from an enclosing trace). Falls back to self.tape when
        no TraceContext is active.
        """
        ctx = TraceContext.get_local_tape()
        return ctx if ctx is not None else self.tape

    def _emit(self, op: OP, *args) -> "Tracer":
        """Append op to the active tape and return the resulting Tracer."""
        tape = self._active_tape()
        return Tracer(tape, tape.append(op, *args))

    def copy(self):
        return Tracer(self.tape, self.index)

    def is_input(self):
        return self.index in self.tape.input_indicies

    def is_constant(self):
        if self.is_input():
            return False
        op, *args = self.tape.nodes[self.index]
        if op != OP.VALUE:
            return False
        return True

    def is_functional(self):
        if self.is_input():
            return False
        op, *args = self.tape.nodes[self.index]
        if op not in {OP.EVALUATE}:
            return False
        return True

    def as_halfplane_bound(self) -> Tuple["Tracer", float, float]:
        op, lhs, rhs = self.tape.nodes[self.index]
        bounds = {
            OP.EQUAL: (-1e-9, 1e-9),
            OP.LESS_THAN: (0, np.inf),
            OP.LESS_EQUAL: (-1e-9, np.inf),
        }
        return rhs - lhs, *bounds[op]

    def value(self):
        op, *args = self.tape.nodes[self.index]

        if op != OP.VALUE:
            return self.tape.nodes[self.index]

        (arg,) = args
        return arg

    def __hash__(self):
        return hash(hash(self.tape) + self.index)

    def __repr__(self):
        return f"Tracer({self.index})"

    @property
    def shape(self) -> Tuple:
        dim = self.tape.dim[self.index]
        if dim.is_scalar():
            raise ValueError("Scalars have no shape")
        return dim.dim

    @property
    def dim(self) -> Dimension:
        return self.tape.dim[self.index]

    def __str__(self):
        return f"Tape {self.tape}:{self.index}"

    def __mul__(self, other):
        return self._emit(OP.MUL, self, other)

    def __rmul__(self, other):
        return self._emit(OP.MUL, other, self)

    def __sub__(self, other):
        return self._emit(OP.SUB, self, other)

    def __matmul__(self, other):
        return self._emit(OP.MATMUL, self, other)

    def __rmatmul__(self, other):
        assert other.shape[0] > 0
        return self._emit(OP.MATMUL, other, self)

    def __add__(self, other):
        tape = self._active_tape()
        if is_additive_identity(other, self):
            return self
        if not isinstance(other, Tracer):
            other = tape.insert_value(other)
        if self.dim.is_scalar() and not other.dim.is_scalar():
            return self * np.ones(other.shape) + other
        elif not self.dim.is_scalar() and other.dim.is_scalar():
            return self + other * np.ones(self.shape)
        return self._emit(OP.ADD, self, other)

    def __radd__(self, other):
        tape = self._active_tape()
        if is_additive_identity(self.dim, other):
            return self
        if not isinstance(other, Tracer):
            other = tape.insert_value(other)
        return other + self

    def __pow__(self, power, modulo=None):
        if isinstance(power, float) and power == 0.5:
            return self._emit(OP.SQRT, self)
        if isinstance(power, int):
            return self._do_integer_power(power)
        return self._emit(OP.PWR, self, power)

    @property
    def T(self):
        return self._emit(OP.TRANSPOSE, self)

    def _do_integer_power(self, power):
        if power <= 0:
            raise NotImplementedError("Negative power")
        result = self
        for _ in range(1, power):
            result = self * result

        return result

    def __getitem__(self, key):
        if self.is_constant():
            return self.value()[key]

        if isinstance(key, slice):
            dimension = self.tape.dim[self.index]
            if isinstance(dimension, FunctionSpace):
                assert len(dimension.output_dimensions()) == 1
                assert dimension.output_dimensions()[0].is_vector()
                p = get_projection(dimension.output_dimensions()[0], key)
            else:
                assert dimension.is_vector(), f"Tried to index a vector"
                p = get_projection(dimension, key)
            return self._emit(OP.MATMUL, p, self)

        if isinstance(key, int):
            dimension = self.tape.dim[self.index]
            assert dimension.is_vector(), f"Tried to index a vector"
            p = get_basis(dimension, key)
            return self._emit(OP.DOT, p, self)

        raise NotImplementedError(
            "Cannot get key {}, not yet implemented", key
        )

    def __setitem__(self, key, value):
        if len(key) != len(self.shape):
            raise ValueError(
                f"Cannot set item {key} = {value} on {self} with shape {self.shape}"
            )

        # when we set an item, we need to do 2 things.
        # 1. Store the operation in the tape
        # 2. Mutate this object so that it points to the new tracer
        assert (
            self[key].shape == value.shape
        ), f"Expected shape {self[key].shape } but got {value.shape} for {key} = {value} on {self} with shape {self.shape}"

        # if the value here is a constant that is not referenced by any other
        # tracers, we can just go ahead and mutate it.
        op, old_value = self.tape.nodes[self.index]
        if (
            op == OP.VALUE
            and not self.tape.find_dependents(self)
            and (
                (isinstance(value, Tracer) and value.is_constant())
                or isinstance(value, scalar_types)
                or isinstance(value, np.ndarray)
            )
        ):
            old_value.__setitem__(key, value)
            return

        # otherwise, we need to add the "SET" operation to the tape
        # and change this tracers index to point to that.

        raise NotImplementedError(
            f"Cannot set item. SET operation not implemented yet for {key} = {value} on {self} with shape {self.shape}"
        )

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def __le__(self, other):
        return self._emit(OP.LESS_EQUAL, self, other)

    def __ge__(self, other):
        return self._emit(OP.LESS_EQUAL, other, self)

    def __lt__(self, other):
        return self._emit(OP.LESS_THAN, self, other)

    def __gt__(self, other):
        return self._emit(OP.LESS_THAN, other, self)

    def __eq__(self, other):
        return self._emit(OP.EQUAL, self, other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if (
            ufunc == np.matmul
            and isinstance(inputs[0], np.ndarray)
            and inputs[0].shape[0] == 0
        ):
            tape = self._active_tape()
            return Tracer(tape, tape.NONE)
        if ufunc == np.less:
            lhs, rhs = inputs
            return rhs > lhs
        try:
            return self._emit(numpy_atomics[ufunc], *inputs)
        except KeyError:
            pass
        raise NotImplementedError(f"{ufunc} is not implemented")

    def __array_function__(self, func, types, args, kwargs):
        try:
            return self._emit(numpy_atomics[func], *args)
        except KeyError:
            pass
        try:
            op = numpy_composites[func](**kwargs)
            return self._emit(op, *op.pre_process(*args))
        except KeyError:
            pass
        raise NotImplementedError(f"{func} with {kwargs} is not implemented")

    def norm(self):
        assert (len(self.shape) == 1 and self.shape[0] > 1) or (
            len(self.shape) == 2 and self.shape[1] == 1
        )
        return np.sqrt(np.dot(self, self))

    def normalise(self):
        norm = self.norm()
        return self._emit(OP.CASE, norm == 0, self, self / norm)

    def __call__(self, *args):
        return self._emit(OP.EVALUATE, self, *args)


class Function:
    """A compiled Coker function.

    Created by :func:`function`.  Holds the traced computation graph and
    dispatches to a backend for evaluation.  Calling a ``Function`` with
    concrete numpy arrays returns the result; calling it inside a tracing
    context records the operations on the outer tape.

    Attributes:
        tape: The recorded computation graph.
        backend: Name of the backend used for concrete evaluation.
        output: List of output :class:`~coker.algebra.kernel.Tracer` nodes.
    """

    INLINE_SIZE = 10

    def __init__(
        self, tape: Tape, outputs: List[Tracer], backend="coker", name=None
    ):
        self.name = name
        self.tape = tape
        self.backend = backend
        self._compiled = None
        if isinstance(outputs, Tracer) or outputs is None:
            self.output = [outputs]
            self.is_single = True
        else:
            self.output = outputs
            self.is_single = False

    @property
    def arguments(self):
        return self.tape.input_names.copy()

    def __repr__(self):
        name = self.name if self.name else "<unknown>"
        try:
            out_shape = f"{self.output_shape()}"
        except AttributeError:
            out_shape = f"{self.output}"
        return f"{name}:{self.input_shape()} -> {out_shape}"

    def to_space(self, name):
        return FunctionSpace(
            name,
            arguments=[
                dim.to_space("input_{i}") if dim else None
                for i, dim in enumerate(self.input_shape())
            ],
            output=[
                dim.to_space("output_{i}") if dim else None
                for i, dim in enumerate(self.input_shape())
            ],
        )

    def input_spaces(self):
        """Return the argument spaces of this function as a list.

        Returns:
            A list of :class:`~coker.algebra.dimensions.Scalar`,
            :class:`~coker.algebra.dimensions.VectorSpace`, or
            :class:`~coker.algebra.dimensions.FunctionSpace` objects in
            argument order.
        """
        return list(self.tape.list_inputs())

    def input_shape(self) -> Tuple[Dimension, ...]:
        """Return the shape of each input argument as a tuple of :class:`~coker.algebra.dimensions.Dimension`."""
        special_inputs = {
            Tape.NONE: None,
            Tape.MAP_TO_NONE: Noop().cast_to_function_space(None),
        }

        return tuple(
            self.tape.dim[i] if i >= 0 else special_inputs[i]
            for i in self.tape.input_indicies
        )

    def output_shape(self) -> Tuple[Dimension, ...]:
        """Return the shape of each output as a tuple of :class:`~coker.algebra.dimensions.Dimension`."""
        return tuple(o.dim if o is not None else None for o in self.output)

    def _prepare_argument(self, arg, index):
        if index == Tape.MAP_TO_NONE:
            return Noop()
        elif index == Tape.NONE:
            return None

        elif isinstance(self.tape.dim[index], FunctionSpace):
            if isinstance(arg, Function):
                return arg

            try:
                return function(
                    self.tape.dim[index].arguments, arg, self.backend
                )
            except DanglingTracerError as ex:
                return self._lift_closure(
                    arg, self.tape.dim[index], ex
                )  # arg is a Python callable

        return arg

    def _lift_closure(
        self, fn, space: FunctionSpace, ex: DanglingTracerError
    ) -> "_BoundFunction":
        """Re-trace fn with captured outer-tape tracers promoted to extra inputs.

        Creates a new inner tape with extra inputs for each captured tracer, registers
        substitution rules so that uses of the outer tracers inside fn are
        transparently rewritten to the corresponding inner inputs during re-tracing,
        then wraps the result in a _BoundFunction that supplies the captured values
        at call time.
        """
        captured = _find_closure_tracers(fn)
        if not captured:
            raise NotImplementedError from ex

        unique_captured: List[Tracer] = list(captured.values())

        extra_spaces = []
        for i, t in enumerate(unique_captured):
            dim = t.dim
            if not isinstance(dim, Dimension):
                raise NotImplementedError(
                    "Capturing FunctionSpace-typed tracers is not supported"
                ) from ex
            extra_spaces.append(dim.to_space(f"_cap_{i}"))

        all_spaces = list(space.arguments) + extra_spaces
        inner_tape = Tape()
        all_inner_args = [inner_tape.input(v) for v in all_spaces]
        orig_inner_args = all_inner_args[: len(space.arguments)]
        cap_inner_args = all_inner_args[len(space.arguments) :]

        for outer_t, inner_t in zip(unique_captured, cap_inner_args):
            inner_tape.add_substitution(outer_t, inner_t)

        with TraceContext(inner_tape):
            output = fn(*orig_inner_args)
            result = _normalise_result(output, inner_tape)

        inner_fn = Function(inner_tape, result, self.backend)
        return _BoundFunction(inner_fn, unique_captured)

    def call_inline(self, *args) -> Tuple[Tracer]:
        """Evaluate this function symbolically inside an active tracing context.

        Unlike ``__call__``, which compiles to the configured backend, this
        always routes through the numpy interpreter so that the result is a
        :class:`~coker.algebra.kernel.Tracer` recorded on the enclosing tape.
        Use this when composing functions inside an ``implementation`` passed
        to :func:`function`.
        """
        from coker.backends import get_backend_by_name

        backend = get_backend_by_name("numpy", set_current=False)
        output = backend.evaluate(self, args)
        if self.is_single:
            return output[0]
        return output

    def __call__(self, *args):
        assert len(args) == len(
            self.tape.input_indicies
        ), f"Expected {len(self.tape.input_indicies)} arguments but got {len(args)}"

        args = [
            (self._prepare_argument(arg, idx))
            for idx, arg in zip(self.tape.input_indicies, args)
        ]

        from coker.backends import get_backend_by_name

        if any(isinstance(a, Tracer) for a in args):
            # Tracing context: interpret through numpy so ops are recorded on
            # the outer tape rather than evaluated numerically.
            backend = get_backend_by_name("numpy", set_current=False)
            output = backend.evaluate(self, args)
        else:
            # Concrete evaluation: compile on first call, reuse thereafter.
            if self._compiled is None:
                backend = get_backend_by_name(self.backend)
                self._compiled = backend.lower(self)
            output = self._compiled(args)

        if self.is_single:
            return output[0]
        return output

    def lower(self):
        from coker.backends import get_backend_by_name

        backend = get_backend_by_name(self.backend)
        return backend.lower(self)

    def compile(self, backend: str):
        raise NotImplementedError("Not yet implemented")

    def __le__(self, other: np.ndarray):
        # self < other
        assert len(self.output_shape()) == 1, "Cannot compare tensors"
        dim = self.output_shape()[0]
        assert dim.shape == get_dim_by_class(
            other
        ), "Arguments have different shapes"
        ones = np.ones_like(other)
        return InequalityExpression(self, -np.inf * ones, other, is_equal=True)

    def __ge__(self, other):
        # self => other
        assert len(self.output_shape()) == 1, "Cannot compare tensors"
        (dim,) = self.output_shape()
        assert dim == get_dim_by_class(
            other
        ), "Arguments have different shapes"
        ones = np.ones_like(other)
        return InequalityExpression(self, other, ones * np.inf, is_equal=True)

    def __lt__(self, other):
        # self <= other
        assert len(self.output_shape()) == 1, "Cannot compare tensors"
        (dim,) = self.output_shape()
        assert dim == get_dim_by_class(
            other
        ), "Arguments have different shapes"
        ones = np.ones_like(other)
        return InequalityExpression(
            self, -np.inf * ones, other, is_equal=False
        )

    def __gt__(self, other):
        # self > other
        assert len(self.output_shape()) == 1, "Cannot compare tensors"
        dim = self.output_shape()[0]
        assert dim.shape == other.shape, "Arguments have different shapes"
        ones = np.ones_like(other)
        return InequalityExpression(self, other, ones * np.inf, is_equal=False)


class _BoundFunction(Function):
    """A Function whose extra tail inputs are pre-bound to captured outer-tape tracers.

    Created by Function._lift_closure when a Python callable passed as a
    FunctionSpace argument closes over tracers from an enclosing trace.
    """

    def __init__(self, inner_fn: Function, captured: List[Tracer]):
        self.name = inner_fn.name
        self.tape = inner_fn.tape
        self.backend = inner_fn.backend
        self._compiled = None
        self.output = inner_fn.output
        self.is_single = inner_fn.is_single
        self._inner = inner_fn
        self._captured = captured

    def __call__(self, *args):
        return self._inner(*args, *self._captured)


class InequalityExpression:
    def __init__(
        self,
        value: Function,
        lower: np.ndarray,
        upper: np.ndarray,
        is_equal: bool = False,
    ):
        self.value = value
        self.lower = lower
        self.upper = upper


def _normalise_result(result, tape: Tape):
    """Normalise an implementation's return value into Tracer(s) on tape."""
    if isinstance(result, np.ndarray):
        result = strip_symbols_from_array(result)
    if isinstance(result, SymbolicVector):
        result = result.collapse()

    def wrap(v):
        if isinstance(v, np.ndarray):
            v = strip_symbols_from_array(v)
        if isinstance(v, SymbolicVector):
            v = v.collapse()
        return v if isinstance(v, Tracer) else tape.insert_value(v)

    if isinstance(result, (list, tuple)):
        result = [wrap(r) for r in result]
    else:
        result = wrap(result)

    if isinstance(result, Tracer) and result.index == tape.NONE:
        return None
    return result


def function(
    arguments: List[Scalar | VectorSpace | FunctionSpace],
    implementation: Callable[[Element, ...], Element],
    backend: str = "coker",
    name: Optional[str] = None,
) -> Function:
    """Compile a Python callable into a Coker :class:`Function`.

    Traces ``implementation`` by calling it with symbolic arguments derived
    from ``arguments``, records the resulting computation graph, and returns
    a :class:`Function` that evaluates it via the chosen backend.

    Args:
        arguments: Ordered list of argument spaces describing the domain of
            the function.  Each entry must be a :class:`Scalar`,
            :class:`VectorSpace`, or :class:`FunctionSpace`.
        implementation: A Python callable that defines the computation.  It
            will be called once during tracing with symbolic
            :class:`~coker.algebra.kernel.Tracer` arguments.
        backend: Name of the backend used for concrete evaluation.
            Built-in options are ``"numpy"`` (default for tracing),
            ``"casadi"``, ``"sympy"``, and ``"coker"``.
        name: Optional human-readable name attached to the returned
            :class:`Function`.

    Returns:
        A compiled :class:`Function` that can be called with concrete numpy
        arrays or symbolic tracers.

    Example:
        >>> import numpy as np
        >>> from coker import function, VectorSpace
        >>> A = np.eye(3)
        >>> f = function([VectorSpace("x", 3)], lambda x: A @ x, backend="numpy")
        >>> f(np.array([1.0, 0.0, 0.0]))
        array([1., 0., 0.])
    """
    with TraceContext() as tape:
        args = [tape.input(v) for v in arguments]
        output = implementation(*args)
        result = _normalise_result(output, tape)
        return Function(tape, result, backend, name)


def strip_symbols_from_array(array: np.ndarray, float_type=float):
    if not isinstance(array, np.ndarray):
        return array

    symbols = defaultdict(list)

    with np.nditer(
        array, flags=["refs_ok", "multi_index"], op_flags=[["readwrite"]]
    ) as it:
        for x in it:
            try:
                x[...] = float_type(x)
            except TypeError as e:

                value = x.tolist()
                assert isinstance(
                    value, Tracer
                ), "Unexpected object in array: {}".format(value)
                symbols[value].append(it.multi_index)
                x[...] = 0.0

    symbol_array = array.astype(float)

    for symbol, coords in symbols.items():
        basis = np.zeros_like(array)
        for c in coords:
            basis[c] = 1
        symbol_array = symbol_array + basis * symbol

    return symbol_array


def normalise(v: Union[np.ndarray, Tracer]):
    if isinstance(v, np.ndarray):
        if all(v_i == 0 for v_i in v):
            return np.zeros_like(v), 0
        else:
            r = np.linalg.norm(v)
            return v / r, r

    assert isinstance(v, Tracer), f"Expected Tracer got {type(v)}"

    unit_v = v.normalise()
    norm_v = v.norm()

    return unit_v, norm_v


_comparison_ops = frozenset({OP.EQUAL, OP.LESS_THAN, OP.LESS_EQUAL})


def if_then_else(expression, true_branch, false_branch):
    """Return one of two values based on a scalar boolean expression.

    Inside a tracing context, records an ``OP.CASE`` node on the tape so that
    the branch is preserved symbolically rather than evaluated eagerly.
    Outside a tracing context, evaluates ``bool(expression)`` immediately.

    Args:
        expression: A scalar boolean condition.  Inside a trace this must be
            the result of a comparison operator (``==``, ``<``, ``<=``) applied
            to a :class:`~coker.algebra.kernel.Tracer`.
        true_branch: Value returned when ``expression`` is ``True``.
        false_branch: Value returned when ``expression`` is ``False``.  Must
            have the same shape as ``true_branch``.

    Returns:
        ``true_branch`` or ``false_branch``, or a symbolic
        :class:`~coker.algebra.kernel.Tracer` representing the choice.

    Raises:
        TypeError: If ``expression`` is a :class:`~coker.algebra.kernel.Tracer`
            that was not produced by a comparison operator, or if it is a
            multi-element array that cannot be unambiguously cast to ``bool``.
        :class:`~coker.algebra.exceptions.InvalidShape`: If ``true_branch`` and
            ``false_branch`` have different shapes.

    Example:
        >>> from coker import function, Scalar, if_then_else
        >>> import numpy as np
        >>> f = function(
        ...     [Scalar("x")],
        ...     lambda x: if_then_else(x == 0, np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        ...     backend="numpy",
        ... )
        >>> f(0)
        array([1., 0.])
        >>> f(1)
        array([0., 1.])
    """
    if isinstance(expression, Tracer):
        node = expression.tape.nodes[expression.index]
        if isinstance(node, Tracer):
            # Raw input variable — not a comparison result.
            raise TypeError(
                "expression must result from a comparison operator (==, <, <=), "
                "got a raw input variable"
            )
        cond_op, *_ = node
        if cond_op not in _comparison_ops:
            raise TypeError(
                f"expression must result from a comparison operator (==, <, <=), got {cond_op}"
            )
        index = expression.tape.append(
            OP.CASE, expression, true_branch, false_branch
        )
        return Tracer(expression.tape, index)

    try:
        cond = bool(expression)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"expression cannot be unambiguously coerced to bool: {exc}"
        ) from exc

    return true_branch if cond else false_branch


_local = threading.local()
_local.trace = []


class TraceContext:
    def __init__(self, tape: "Tape | None" = None):
        self._tape = tape

    def __enter__(self):
        if self._tape is None:
            self._tape = Tape()
        _local.trace.append(self._tape)
        return self._tape

    def __exit__(self, exc_type, exc_val, exc_tb):
        _local.trace.pop()

    @staticmethod
    def get_local_tape() -> Tape | None:
        if not _local.trace:
            return None
        return _local.trace[-1]
