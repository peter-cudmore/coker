import weakref

import numpy as np
from typing import Callable, Union, Tuple, List, Optional, Set
from collections import defaultdict

from coker.algebra.dimensions import (
    Dimension,
    VectorSpace,
    Scalar,
    FunctionSpace,
    Element,
)
from coker.algebra.tensor import SymbolicVector
from coker.algebra.ops import OP, Noop

from coker.algebra.ops import numpy_atomics, numpy_composites

import threading
import dataclasses


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


Inferred = None


def get_dim_by_class(arg):
    if isinstance(arg, scalar_types):
        return Dimension(None)
    try:

        d = Dimension(arg.shape)
        return d
    except (AttributeError, ValueError):
        pass

    raise NotImplementedError(f"Don't know the shape of {type(arg)}")


class Tape:
    NONE = -1
    MAP_TO_NONE = -2

    def __init__(self):
        self.nodes = []
        self.constants = []
        self.dim = []
        self.input_indicies = []
        self.input_names = []

    def op(self, i):
        return self.nodes[i][0]

    def find_dependents(self, tracer: "Tracer") -> Set[int]:
        index = tracer.index
        result = set()
        for i, (op, *args) in enumerate(self.nodes[tracer.index :]):
            if op == OP.VALUE:
                continue
            for arg in args:
                if isinstance(arg, Tracer):
                    if arg.index == index or arg.index in result:
                        result.add(index + i)
        return result

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
            assert arg.tape is self, "Tracer belongs to another tape"
            dims.append(arg.dim)

        return op.compute_shape(*dims)

    def append(self, op: OP, *args) -> int:
        args = [strip_symbols_from_array(a) for a in args]

        args = [
            self.insert_value(a) if not isinstance(a, Tracer) else a.copy()
            for a in args
        ]
        out_dim = self._compute_shape(op, *args)
        index = len(self.dim)
        self.nodes.append((op, *args))
        self.dim.append(out_dim)
        return index

    def insert_value(self, arg):
        if arg is None:
            return None
        assert not isinstance(arg, Tracer)
        dim = get_dim_by_class(arg)
        idx = len(self.dim)
        self.nodes.append((OP.VALUE, arg))
        self.dim.append(dim)
        return Tracer(self, idx)

    def input(self, v: VectorSpace | Scalar):
        if v is None:
            self.input_indicies.append(Tape.NONE)
            return None

        if isinstance(v, Noop):
            self.input_indicies.append(Tape.MAP_TO_NONE)
            return v

        index = len(self.dim)
        if isinstance(v, VectorSpace):
            self.dim.append(Dimension(v.dimension))
        elif isinstance(v, Scalar):
            self.dim.append(Dimension(None))

        elif isinstance(v, FunctionSpace):
            self.dim.append(v)
        else:
            assert (
                False
            ), f"Invalid input type {v}: of {type(v)} at index {index}"
        tracer = Tracer(self, index)
        self.nodes.append(tracer)
        self.input_indicies.append(index)
        self.input_names.append(v.name)
        return tracer

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
        index = self.tape.append(OP.MUL, self, other)
        return Tracer(self.tape, index)

    def __rmul__(self, other):
        index = self.tape.append(OP.MUL, other, self)

        return Tracer(self.tape, index)

    def __add__(self, other):
        if is_additive_identity(other, self):
            return self

        if not isinstance(other, Tracer):
            other = self.tape.insert_value(other)

        if self.dim.is_scalar() and not other.dim.is_scalar():
            return self * np.ones(other.shape) + other
        elif not self.dim.is_scalar() and other.dim.is_scalar():
            return self + other * np.ones(self.shape)

        index = self.tape.append(OP.ADD, self, other)
        return Tracer(self.tape, index)

    def __radd__(self, other):
        if is_additive_identity(self.dim, other):
            return self

        if not isinstance(other, Tracer):
            other = self.tape.insert_value(other)

        return other + self

    def __sub__(self, other):
        index = self.tape.append(OP.SUB, self, other)
        return Tracer(self.tape, index)

    def __rmatmul__(self, other):
        index = self.tape.append(OP.MATMUL, other, self)
        return Tracer(self.tape, index)

    def __matmul__(self, other):
        index = self.tape.append(OP.MATMUL, self, other)
        return Tracer(self.tape, index)

    def __pow__(self, power, modulo=None):
        if isinstance(power, float) and power == 0.5:
            index = self.tape.append(OP.SQRT, self)
        elif isinstance(power, int):
            return self._do_integer_power(power)

        else:
            index = self.tape.append(OP.PWR, self, power)

        return Tracer(self.tape, index)

    @property
    def T(self):
        index = self.tape.append(OP.TRANSPOSE, self)
        return Tracer(self.tape, index)

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
            assert dimension.is_vector(), f"Tried to index a vector"
            p = get_projection(dimension, key)
            index = self.tape.append(OP.MATMUL, p, self)
            return Tracer(self.tape, index)

        elif isinstance(key, int):
            dimension = self.tape.dim[self.index]
            assert dimension.is_vector(), f"Tried to index a vector"
            p = get_basis(dimension, key)
            index = self.tape.append(OP.DOT, p, self)
            return Tracer(self.tape, index)

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
        idx = self.tape.append(OP.LESS_EQUAL, self, other)
        return Tracer(self.tape, idx)

    def __ge__(self, other):
        idx = self.tape.append(OP.LESS_EQUAL, other, self)
        return Tracer(self.tape, idx)

    def __lt__(self, other):
        idx = self.tape.append(OP.LESS_THAN, self, other)
        return Tracer(self.tape, idx)

    def __gt__(self, other):
        idx = self.tape.append(OP.LESS_THAN, other, self)
        return Tracer(self.tape, idx)

    def __eq__(self, other):
        idx = self.tape.append(OP.EQUAL, self, other)
        return Tracer(self.tape, idx)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        try:
            op = numpy_atomics[ufunc]

            index = self.tape.append(op, *inputs)
            return Tracer(self.tape, index)

        except KeyError:
            pass

        if ufunc == np.less:
            # lhs -> numpy item
            # rhs -> Tracer
            # op lhs < rhs
            lhs, rhs = inputs
            return rhs > lhs

        raise NotImplementedError(f"{ufunc} is not implemented")

    def __array_function__(self, func, types, args, kwargs):
        try:
            op = numpy_atomics[func]
            index = self.tape.append(op, *args)
            return Tracer(self.tape, index)
        except KeyError:
            pass

        try:
            op = numpy_composites[func](**kwargs)
            args = op.pre_process(*args)
            index = self.tape.append(op, *args)
            return Tracer(self.tape, index)
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

        condition = norm == 0

        output = self.tape.append(OP.CASE, condition, self, self / norm)

        return Tracer(self.tape, output)

    def __call__(self, *args):
        index = self.tape.append(OP.EVALUATE, self, *args)
        return Tracer(self.tape, index)


class Function:
    def __init__(
        self, tape: Tape, outputs: List[Tracer], backend="coker", name=None
    ):
        self.name = name
        self.tape = tape
        self.backend = backend
        if isinstance(outputs, Tracer):
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

    def input_shape(self) -> Tuple[Dimension, ...]:
        special_inputs = {
            Tape.NONE: None,
            Tape.MAP_TO_NONE: Noop().cast_to_function_space(None),
        }

        return tuple(
            self.tape.dim[i] if i >= 0 else special_inputs[i]
            for i in self.tape.input_indicies
        )

    def output_shape(self) -> Tuple[Dimension, ...]:
        return tuple(o.dim for o in self.output)

    def _prepare_argument(self, arg, index):
        if index == Tape.MAP_TO_NONE:
            return Noop()
        elif index == Tape.NONE:
            return None
        elif isinstance(self.tape.dim[index], FunctionSpace):
            return function(self.tape.dim[index].arguments, arg, self.backend)
        return arg

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
            backend = get_backend_by_name("numpy", set_current=False)
        else:
            backend = get_backend_by_name(self.backend)

        output = backend.evaluate(self, args)

        if self.is_single:
            return output[0]
        return output

    def lower(self):
        from coker.backends import get_backend_by_name

        backend = get_backend_by_name(self.backend)
        return backend.lower(self)

    def compile(self, backend: str):
        raise NotImplementedError("Not yet implemented")


def function(
    arguments: List[Scalar | VectorSpace | FunctionSpace],
    implementation: Callable[[Element, ...], Element],
    backend: str = "coker",
    name: Optional[str] = None,
) -> Function:
    # create symbols
    # call function to construct expression graph

    tape = Tape()
    with TraceContext() as tape:
        args = [tape.input(v) for v in arguments]
        result = implementation(*args)

        if isinstance(result, np.ndarray):
            result = strip_symbols_from_array(result)

        if isinstance(result, SymbolicVector):
            result = result.collapse()

        def wrap(v):
            if isinstance(v, np.ndarray):
                v = strip_symbols_from_array(v)
            if isinstance(v, SymbolicVector):
                v = v.collapse()

            if isinstance(v, Tracer):
                return v

            return tape.insert_value(v)

        if isinstance(result, (list, tuple)):
            result = [wrap(r) for r in result]

        else:
            result = wrap(result)

        if result is None or result is [None]:
            return Noop()

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


_local = threading.local()


class TraceContext:
    def __enter__(self):
        tape = Tape()
        _local.trace = tape
        return tape

    def __exit__(self, exc_type, exc_val, exc_tb):
        _local.trace = None

    @staticmethod
    def get_local_tape() -> Tape | None:
        return getattr(_local, "trace", None)
