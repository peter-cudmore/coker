from __future__ import annotations

import dataclasses
import weakref
from collections import defaultdict
from types import FunctionType

import threading

import numpy as np
import scipy as sp
from typing import Any, Callable, Iterable, List, Set, Tuple

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    FunctionValueDimension,
    ResultBundleDimension,
    Scalar,
    VectorSpace,
)
from coker.algebra.ops import (
    OP,
    Noop,
    Operator,
    ReshapeOP,
    numpy_atomics,
    numpy_composites,
)

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
    if isinstance(arg, CallableReference):
        return arg.function_space
    try:
        d = Dimension(arg.shape)
        return d
    except (AttributeError, ValueError):
        pass

    raise NotImplementedError(f"Don't know the shape of {type(arg)}")


class DanglingTracerError(Exception):
    def __init__(self, *args, tracers: List[Tracer]):
        super().__init__(*args)
        self.tracers = tracers


def _find_closure_tracers(fn) -> dict:
    """Return all Tracer objects captured in fn's closure.

    Returns a dict keyed by ``(tape_id, tracer_index)`` so duplicates
    are collapsed.
    """
    captured = {}
    if not isinstance(fn, FunctionType) or fn.__closure__ is None:
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
    FUNCTION_REF = -3

    def __init__(self, tape_ref: Tape):
        self._nodes = []
        self._constants = []
        self._constant_hashmap = {}
        self._callable_archive = []
        self._callable_hashmap = {}
        self.tape_ref = weakref.ref(tape_ref)
        assert self.INNER_REF not in OP.__members__.values()
        assert self.CONSTANT_REF not in OP.__members__.values()
        assert self.FUNCTION_REF not in OP.__members__.values()

    @staticmethod
    def constant_hash(value) -> int:
        if isinstance(value, (int, float)):
            return hash((0, 0, value))
        if isinstance(value, np.ndarray):
            return hash((*value.shape, *value.flatten().tolist()))
        if sp.sparse.issparse(value):
            canonical = value.tocsc(copy=True)
            canonical.sum_duplicates()
            canonical.sort_indices()
            return hash(
                (
                    canonical.shape,
                    canonical.dtype.str,
                    canonical.indptr.tobytes(),
                    canonical.indices.tobytes(),
                    canonical.data.tobytes(),
                )
            )
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
        elif op == OP.FUNCTION_VALUE:
            (reference,) = args
            self._nodes.append((self.FUNCTION_REF, reference._archive_index))
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
        if op == self.FUNCTION_REF:
            (archive_index,) = args
            return OP.FUNCTION_VALUE, CallableReference(
                self.tape_ref(), archive_index
            )
        return op, *args

    def __len__(self):
        return len(self._nodes)


@dataclasses.dataclass
class _CallableArchiveEntry:
    """Native callable and its input/output declaration stored by a tape."""

    callable_value: Callable
    function_space: FunctionSpace
    result_dimension: Dimension | FunctionSpace | ResultBundleDimension


class CallableReference:
    """A reference to a callable entry owned by a tape."""

    def __init__(self, tape: Tape, archive_index: int):
        self._tape = weakref.ref(tape)
        self._archive_index = archive_index

    @property
    def _entry(self) -> _CallableArchiveEntry:
        return self._tape().nodes._callable_archive[self._archive_index]

    @property
    def function_space(self) -> FunctionSpace:
        return self._entry.function_space

    @property
    def result_dimension(
        self,
    ) -> Dimension | FunctionSpace | ResultBundleDimension:
        return self._entry.result_dimension

    def __call__(self, *args):
        return self._entry.callable_value(*args)


class Tape:
    NONE = -1
    MAP_TO_NONE = -2

    def __init__(self, backend: str | None = None):
        self._inner = TapeInner(self)
        self.dim = []
        self.input_indicies = []
        self.input_names = []
        self.backend = backend
        self._substitutions: dict = {}
        self._node_hashmap: dict[int, int] = {}

    def add_substitution(self, foreign: Tracer, local: Tracer):
        """Register a rewrite rule for substituting a traced value.

        Any occurrence of ``foreign`` in appended args is replaced by
        ``local``.
        """
        self._substitutions[(id(foreign.tape), foreign.index)] = local

    def op(self, i):
        return self.nodes[i][0]

    @property
    def nodes(self):
        return self._inner

    def __len__(self):
        return len(self.nodes)

    def _create_callable_reference(
        self,
        callable_value: Callable[..., Any],
        function_space: FunctionSpace,
        result_dimension: (
            Dimension | FunctionSpace | ResultBundleDimension | None
        ) = None,
    ) -> CallableReference:
        if result_dimension is None:
            output_dimensions = function_space.output_dimensions()
            if len(output_dimensions) != 1:
                raise ValueError(
                    "Callable references require one result or an explicit "
                    "result dimension"
                )
            (result_dimension,) = output_dimensions

        key = id(callable_value)
        archive_index = self._inner._callable_hashmap.get(key)
        if archive_index is None:
            archive_index = len(self._inner._callable_archive)
            self._inner._callable_hashmap[key] = archive_index
            self._inner._callable_archive.append(
                _CallableArchiveEntry(
                    callable_value,
                    function_space,
                    result_dimension,
                )
            )
        return CallableReference(self, archive_index)

    def find_dependents(self, tracer: Tracer) -> Set[int]:
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

    def depends_on(self, expression: Tracer, dependency: Tracer) -> bool:
        """Return whether ``expression`` is structurally derived from
        ``dependency``.

        The query follows tracer edges in this tape rather than relying on
        input names, operation names, or metadata maintained by callers.
        """
        if (
            not isinstance(expression, Tracer)
            or not isinstance(dependency, Tracer)
            or expression.tape is not self
            or dependency.tape is not self
        ):
            return False

        target = dependency.index
        pending = [expression.index]
        visited = set()
        while pending:
            index = pending.pop()
            if index == target:
                return True
            if index in visited or index < 0 or index >= len(self.nodes):
                continue
            visited.add(index)
            node = self.nodes[index]
            if isinstance(node, Tracer):
                continue
            if node[0] == OP.VALUE:
                continue
            pending.extend(
                argument.index
                for argument in node[1:]
                if isinstance(argument, Tracer) and argument.tape is self
            )
        return False

    def inputs(self):
        for index in self.input_indicies:
            if index == Tape.NONE:
                yield None
            elif index == Tape.MAP_TO_NONE:
                yield Noop()
            else:
                yield Tracer(self, index)

    def _compute_shape(self, op: OP | Operator, *args):
        dims = []
        for arg in args:
            if arg is None:
                dims.append(None)
            else:
                assert isinstance(arg, Tracer)
                dims.append(arg.dim)
        return op.compute_shape(*dims)

    def append(
        self,
        op: OP | Operator,
        *args,
    ) -> int:
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
            (
                a.copy()
                if isinstance(a, Tracer)
                else (
                    self.insert_function_value(a)
                    if isinstance(a, CallableReference)
                    else self.insert_value(a)
                )
            )
            for a in args
        ]

        node_hash = hash((op, *args))
        if node_hash in self._node_hashmap:
            return self._node_hashmap[node_hash]

        out_dim = self._compute_shape(op, *args)
        index = len(self.dim)
        self.nodes.push_op(op, *args)
        self.dim.append(out_dim)
        self._node_hashmap[node_hash] = index
        return index

    def insert_value(self, arg):
        if arg is None:
            return None
        assert not isinstance(arg, Tracer)

        node_hash = hash((OP.VALUE, self.nodes.constant_hash(arg)))
        if node_hash in self._node_hashmap:
            return Tracer(self, self._node_hashmap[node_hash])

        dim = get_dim_by_class(arg)
        idx = len(self.dim)
        self.nodes.push_op(OP.VALUE, arg)
        self.dim.append(dim)
        self._node_hashmap[node_hash] = idx
        return Tracer(self, idx)

    def insert_function_value(self, reference: CallableReference) -> Tracer:
        node_hash = hash((OP.FUNCTION_VALUE, reference._archive_index))
        if node_hash in self._node_hashmap:
            return Tracer(self, self._node_hashmap[node_hash])

        index = len(self.dim)
        self.nodes.push_op(OP.FUNCTION_VALUE, reference)
        self.dim.append(
            FunctionValueDimension(
                reference.function_space,
                reference.result_dimension,
            )
        )
        self._node_hashmap[node_hash] = index
        return Tracer(self, index)

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
    except (AttributeError, TypeError, ValueError):
        pass

    return False


class Tracer(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, tape: Tape, index: int):
        self._tape = weakref.ref(tape)
        self.index = index

    @property
    def tape(self):
        return self._tape()

    def _active_tape(self) -> Tape:
        """Return the tape that operations on this Tracer should use.

        During tracing the current TraceContext tape is returned so that
        all ops land on the active tape. That tape may differ from
        ``self.tape`` when a closure captures a tracer from an enclosing
        trace. Falls back to ``self.tape`` when no TraceContext is
        active.
        """
        ctx = TraceContext.get_local_tape()
        return ctx if ctx is not None else self.tape

    def _emit(self, op: OP, *args) -> Tracer:
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
        op, *_ = self.tape.nodes[self.index]
        if op != OP.EVALUATE:
            return False
        return True

    def as_halfplane_bound(self) -> Tuple[Tracer, float, float]:
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
        if isinstance(key, tuple):
            op, *args = self.tape.nodes[self.index]
            if isinstance(op, ReshapeOP) and op.order == "C":
                (base,) = args
                if base.dim.is_vector():
                    flat = np.arange(np.prod(self.shape)).reshape(self.shape)[
                        key
                    ]
                    if np.isscalar(flat):
                        return base[int(flat)]
                    flat = np.asarray(flat)
                    if flat.ndim == 1:
                        p = np.zeros((flat.size, base.shape[0]), dtype=float)
                        p[np.arange(flat.size), flat] = 1
                        return base._emit(OP.MATMUL, p, base)

        def leading_item(tracer, item):
            dimension = tracer.tape.dim[tracer.index]
            if isinstance(item, slice):
                if isinstance(dimension, FunctionSpace):
                    assert len(dimension.output_dimensions()) == 1
                    dimension = dimension.output_dimensions()[0]
                if dimension.is_matrix():
                    dimension = Dimension((dimension.dim[0],))
                assert dimension.is_vector(), "Tried to index a non-vector"
                return tracer._emit(
                    OP.MATMUL, get_projection(dimension, item), tracer
                )
            if isinstance(item, int):
                if dimension.is_matrix():
                    rows = dimension.dim[0]
                    p = get_basis(Dimension((rows,)), item).reshape((1, rows))
                    return tracer._emit(OP.MATMUL, p, tracer).T
                assert dimension.is_vector(), "Tried to index a non-vector"
                return tracer._emit(OP.DOT, get_basis(dimension, item), tracer)
            raise NotImplementedError(
                f"Cannot get key {item}, not yet implemented"
            )

        if not isinstance(key, tuple):
            return leading_item(self, key)

        result = self
        for i, item in enumerate(key):
            if i and result.dim.is_matrix():
                result = leading_item(result.T, item)
                if result.dim.is_matrix():
                    result = result.T
                continue
            result = leading_item(result, item)
        return result

    def __setitem__(self, key, value):
        if len(key) != len(self.shape):
            raise ValueError(
                f"Cannot set item {key} = {value} on {self} with shape "
                f"{self.shape}"
            )

        # when we set an item, we need to do 2 things.
        # 1. Store the operation in the tape
        # 2. Mutate this object so that it points to the new tracer
        assert self[key].shape == value.shape, (
            f"Expected shape {self[key].shape} but got {value.shape} "
            f"for {key} = {value} on {self} with shape {self.shape}"
        )

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
            f"Cannot set item. SET operation not implemented yet for "
            f"{key} = {value} on {self} with shape {self.shape}"
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
        if func is np.clip:
            value, lower, upper = args

            def clip_scalar(item, item_lower, item_upper):
                result = item
                if item_upper is not None:
                    result = if_then_else(
                        item <= item_upper, result, item_upper
                    )
                if item_lower is not None:
                    result = if_then_else(
                        item <= item_lower, item_lower, result
                    )
                return result

            if isinstance(value, Tracer) and value.dim.is_vector():
                count = value.shape[0]
                lower_values = (
                    lower
                    if lower is not None
                    and isinstance(lower, (list, tuple, np.ndarray))
                    else [lower] * count
                )
                upper_values = (
                    upper
                    if upper is not None
                    and isinstance(upper, (list, tuple, np.ndarray))
                    else [upper] * count
                )
                return np.concatenate(
                    [
                        clip_scalar(
                            value[index],
                            lower_values[index],
                            upper_values[index],
                        )
                        for index in range(count)
                    ]
                )

            return clip_scalar(value, lower, upper)

        if func is np.linalg.norm and args[0].dim.is_vector():
            value = args[0]
            order = kwargs.get("ord", args[1] if len(args) > 1 else None)
            if order in (None, 2):
                return np.sqrt(np.dot(value, value))
            if order == 1:
                return sum(
                    abs(value[index]) for index in range(value.shape[0])
                )
            raise NotImplementedError(
                f"np.linalg.norm order {order!r} is not supported"
            )
        if (
            func is np.linalg.norm
            and args[0].dim.is_matrix()
            and args[0].tape.backend == "coker"
            and kwargs.get("ord", args[1] if len(args) > 1 else None) == 1
        ):
            value = args[0]
            column_norms = [
                sum(abs(value[row, column]) for row in range(value.shape[0]))
                for column in range(value.shape[1])
            ]
            result = column_norms[0]
            for column_norm in column_norms[1:]:
                result = if_then_else(
                    result <= column_norm, column_norm, result
                )
            return result

        try:
            if func == np.reshape:
                shape = (
                    args[1]
                    if len(args) > 1
                    else kwargs.get("newshape", kwargs.get("shape"))
                )
                return self._emit(ReshapeOP(shape), args[0])
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


def strip_symbols_from_array(array: np.ndarray, float_type=float):
    if not isinstance(array, np.ndarray):
        return array
    if array.size == 0:
        return array.astype(float)

    symbols = defaultdict(list)

    with np.nditer(
        array, flags=["refs_ok", "multi_index"], op_flags=[["readwrite"]]
    ) as it:
        for x in it:
            try:
                x[...] = float_type(x)
            except TypeError:

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


def normalise(v: np.ndarray | Tracer):
    if isinstance(v, np.ndarray):
        if all(v_i == 0 for v_i in v):
            return np.zeros_like(v), 0

        norm_v = np.linalg.norm(v)
        return v / norm_v, norm_v

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
            to a :class:`~coker.algebra.graph.Tracer`.
        true_branch: Value returned when ``expression`` is ``True``.
        false_branch: Value returned when ``expression`` is ``False``.  Must
            have the same shape as ``true_branch``.

    Returns:
        ``true_branch`` or ``false_branch``, or a symbolic
        :class:`~coker.algebra.graph.Tracer` representing the choice.

    Raises:
        TypeError: If ``expression`` is a :class:`~coker.algebra.graph.Tracer`
            that was not produced by a comparison operator, or if it is a
            multi-element array that cannot be unambiguously cast to ``bool``.
        :class:`~coker.algebra.exceptions.InvalidShape`: If ``true_branch`` and
            ``false_branch`` have different shapes.

    Example:
        >>> from coker import function, Scalar, if_then_else
        >>> import numpy as np
        >>> f = function(
        ...     [Scalar("x")],
        ...     lambda x: if_then_else(
            ...         x == 0,
            ...         np.array([1.0, 0.0]),
            ...         np.array([0.0, 1.0]),
            ...     ),
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
            # Raw input variable â€” not a comparison result.
            raise TypeError(
                "expression must result from a comparison operator "
                "(==, <, <=), got a raw input variable"
            )
        cond_op, *_ = node
        if cond_op not in _comparison_ops:
            raise TypeError(
                "expression must result from a comparison operator "
                f"(==, <, <=), got {cond_op}"
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
    def __init__(self, tape: Tape | None = None, backend: str | None = None):
        self._tape = tape
        self._backend = backend

    def __enter__(self):
        if self._tape is None:
            self._tape = Tape(self._backend)
        _local.trace.append(self._tape)
        return self._tape

    def __exit__(self, exc_type, exc_val, exc_tb):
        _local.trace.pop()

    @staticmethod
    def get_local_tape() -> Tape | None:
        if not _local.trace:
            return None
        return _local.trace[-1]
