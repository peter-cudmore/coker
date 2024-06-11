import dataclasses
import enum

import numpy as np
from typing import List, Callable, Tuple, Any
from collections import defaultdict

from coker.algebra.tensor import Tensor
from coker.algebra.dimensions import Dimension
from coker.algebra.ops import OP, compute_shape, numpy_atomics, numpy_composites


def get_basis(dimension: Dimension , i: int):
    return np.array([
        1 if j == i else 0 for j in range(dimension.dim[0])
    ])


def get_projection(dimension:Dimension, slc: slice):
    if isinstance(dimension.dim, tuple):
        cols = dimension.dim[0]
    else:
        return 1

    indicies = list(range(cols))[slc]
    rows = len(indicies)
    proj = np.zeros((rows, cols), dtype=float)
    for row, col in enumerate(indicies):
        proj[row, col] = 1
    return proj


@dataclasses.dataclass
class VectorSpace:
    name: str
    dimension: int


@dataclasses.dataclass
class Scalar:
    name: str


class Element:
    parent: VectorSpace


Inferred = None


def get_dim_by_class(arg):
    if isinstance(arg, (float, complex, int)):
        return Dimension(None)
    if isinstance(arg, np.ndarray):
        d = Dimension(arg.shape)
        return d
    else:
        raise NotImplemented("Don't know the shape of {}".format(arg))


class ExprOp(enum.Enum):
    LESS_THAN = "<"
    GREATER_THAN = ">"


class Expression:
    def __init__(self, op: ExprOp, lhs, rhs):

        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def as_halfplane_bound(self):
        if self.op == ExprOp.LESS_THAN:
            return self.lhs - self.rhs

        return self.rhs - self.lhs

    @property
    def tape(self) -> 'Tape':
        l_tape = self.lhs.tape if isinstance(self.lhs, Tracer) else None
        r_tape = self.rhs.tape if isinstance(self.rhs, Tracer) else None
        if l_tape and not r_tape:
            return l_tape
        elif r_tape and not l_tape:
            return r_tape
        elif l_tape == r_tape and l_tape is not None:
            return l_tape
        else:
            raise NotImplementedError("No tape found")


class Tape:
    def __init__(self):
        self.nodes = []
        self.constants = []
        self.dim = []
        self.input_indicies = []

    def __hash__(self):
        return id(self)

    def _compute_shape(self, op: OP, *args) -> Dimension:
        dims = []
        for arg in args:
            assert isinstance(arg, Tracer)
            assert arg.tape is self, "Tracer belongs to another tape"
            dims.append(arg.dim)

        return op.compute_shape(*dims)

    def append(self, op: OP, *args) -> int:
        args = [
            strip_symbols_from_array(a) for a in args
        ]
        args = [
            self.insert_value(a) if not isinstance(a, Tracer) else a
            for a in args
        ]
        out_dim = self._compute_shape(op, *args)
        index = len(self.dim)
        self.nodes.append((op, *args))
        self.dim.append(out_dim)
        return index

    def insert_value(self, arg):
        dim = get_dim_by_class(arg)
        idx = len(self.dim)
        self.nodes.append((OP.VALUE, arg))
        self.dim.append(dim)
        return Tracer(self, idx)

    def input(self, v: VectorSpace | Scalar):
        if isinstance(v, VectorSpace):
            self.dim.append(Dimension(v.dimension))
        elif isinstance(v, Scalar):
            self.dim.append(Dimension(None))
        else:
            assert False, "Unreachable"
        index = len(self.nodes)
        tracer = Tracer(self, index)
        self.nodes.append(tracer)
        self.input_indicies.append(index)
        return tracer

    def substitute(self, index, value):
        assert index in self.input_indicies
        op, args = value

        self.input_indicies.remove(index)
        self.nodes[index] = value


def is_additive_identity(space: Dimension, arg) -> bool:

    if isinstance(arg, (float, int, complex)) and arg == 0:
        return True

    try:
        return (space.dim == arg.shape) and (arg == 0).all()
    except:
        pass

    return False


class Tracer(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, tape: Tape, index: int):
        self.tape = tape
        self.index = index

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
        index = self.tape.append(OP.ADD, self, other)

        return Tracer(self.tape, index)

    def __radd__(self, other):
        if is_additive_identity(self.dim, other):
            return self
        index = self.tape.append(OP.ADD, other, self)
        return Tracer(self.tape, index)

    def __sub__(self, other):
        index = self.tape.append(OP.SUB, self, other)
        return Tracer(self.tape, index)

    def __rmatmul__(self, other):
        index = self.tape.append(OP.MATMUL,other, self)
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
            raise NotImplementedError()
        result = self
        for _ in range(1, power):
            result = self * result

        return result

    def __getitem__(self, item):
        if isinstance(item, slice):
            dimension = self.tape.dim[self.index]
            assert not dimension.is_scalar()
            p = get_projection(dimension, item)
            index = self.tape.append(OP.MATMUL, p, self)
            return Tracer(self.tape, index)

        elif isinstance(item, int):
            dimension = self.tape.dim[self.index]
            assert not dimension.is_scalar(), f"Tried to index a scalar"
            p = get_basis(dimension, item)
            index = self.tape.append(OP.DOT, p, self)
            return Tracer(self.tape, index)

        raise NotImplementedError()

    def __lt__(self, other):
        difference = other - self

        return Expression(ExprOp.GREATER_THAN, 0, difference)

    def __gt__(self, other):
        difference = self - other
        return Expression(ExprOp.GREATER_THAN, 0, difference)

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


def py_evaluate_tape(tape, args, outputs, backend='numpy'):
    from coker.backends.evaluator import evaluate
    return evaluate(tape, args, outputs, backend)


class Kernel:
    def __init__(self, tape: Tape, outputs: List[Tracer]):
        self.tape = tape
        if isinstance(outputs, Tracer):
            self.output = [outputs]
            self.is_single = True
        else:
            self.outputs = outputs
            self.is_single = False
    def __repr__(self):
        return f"Kernel:{self.input_shape()} -> {self.output_shape()}"

    def input_shape(self) -> Tuple[Dimension, ...]:
        return tuple(Dimension(self.tape.dim[i]) for i in self.tape.input_indicies)

    def output_shape(self) -> Tuple[Dimension, ...]:
        return tuple(o.dim for o in self.output)

    def __call__(self, *args):
        assert len(args) == len(self.tape.input_indicies)
        # todo: check dimensions
        #

        output = py_evaluate_tape(self.tape, args, self.output)
        if self.is_single:
            return output[0]
        return output

    def compile(self, backend: str):
        pass


def kernel(arguments: List[Scalar | VectorSpace],
           implementation: Callable[[Element, ...], Element]):
    # create symbols
    # call function to construct expression graph

    tape = Tape()
    args = [tape.input(v) for v in arguments]

    result = implementation(*args)

    if isinstance(result, np.ndarray):
        result = strip_symbols_from_array(result)

    if isinstance(result, Tensor):
        result = result.collapse()

    return Kernel(tape, result)


def strip_symbols_from_array(array: np.ndarray, float_type=float):

    if not isinstance(array, np.ndarray):
        return array

    symbols = defaultdict(list)

    with np.nditer(array, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
        for x in it:
            try:
                x[...] = float_type(x)
            except TypeError as e:

                value = x.tolist()
                assert isinstance(value, Tracer), "Unexpected object in array: {}".format(value)
                symbols[value].append(it.multi_index)
                x[...] = 0.0

    symbol_array = array.astype(float)

    for symbol, coords in symbols.items():
        basis = np.zeros_like(array)
        for c in coords:
            basis[c] = 1
        symbol_array = symbol_array + basis * symbol

    return symbol_array
