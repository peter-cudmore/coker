import enum
import numpy as np
from typing import Dict, Callable
from coker.algebra.exceptions import InvalidShape, InvalidArgument
from coker.algebra.dimensions import Dimension, FunctionSpace
from typing_extensions import final


class OP(enum.Enum):
    VALUE = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    MATMUL = 5
    INT_PWR = 6
    PWR = 7
    EXP = 8
    SIN = 9
    COS = 10
    TAN = 11
    ARCSIN = 12
    ARCCOS = 13
    ARCTAN = 14
    SQRT = 15
    DOT = 16
    CROSS = 17
    TRANSPOSE = 18
    NEG = 19
    ABS = 20
    CASE = 21
    EQUAL = 22
    ARCTAN2 = 23
    LESS_THAN = 24
    LESS_EQUAL = 25
    EVALUATE = 26

    def compute_shape(self, *dims: Dimension) -> Dimension:
        return compute_shape[self](*dims)

    def is_linear(self):
        return self in {OP.ADD, OP.SUB, OP.TRANSPOSE, OP.NEG}

    def is_bilinear(self):
        return self in {OP.MUL, OP.MATMUL, OP.CROSS, OP.DOT}

    def is_nonlinear(self):
        return not self.is_linear() and not self.is_bilinear()


@final
class Noop:
    def __call__(self, *args, **kwargs):
        return None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    @staticmethod
    def cast_to_function_space(arguments=None):
        return FunctionSpace("noop", arguments, None)


class Operator:
    def pre_process(self, *args):
        return args

    def compute_shape(self, *dims: Dimension) -> Dimension:
        raise NotImplementedError("Compute shape not implemented for {}", self)

    def is_linear(self):
        raise NotImplementedError("is_linear not implemented for {}", self)

    def is_bilinear(self):
        return False

    def is_nonlinear(self):
        return not self.is_linear() and not self.is_bilinear()


class ConcatenateOP(Operator):
    __slots__ = ("axis",)

    def __init__(self, axis=0):
        self.axis = axis

    def is_linear(self):
        return True

    def pre_process(self, *args):

        assert len(args) == 1
        return args[0]

    def compute_shape(self, *dims: Dimension) -> Dimension:

        if all(d.is_scalar() or d.is_vector() for d in dims):
            dim = Dimension((sum(d.flat() for d in dims),))
            return dim

        out_dims = list(dims[0].dim)
        for d in dims[1:]:
            assert all(
                d.dim[i] == out_dims[i]
                for i in range(len(out_dims))
                if i != self.axis
            )
            out_dims[self.axis] += d.dim[self.axis]

        return Dimension(tuple(out_dims))


class ReshapeOP(Operator):
    def __init__(self, shape, order="C"):
        self.newshape = shape
        self.order = order

    def compute_shape(self, dim: Dimension) -> Dimension:
        return Dimension(self.newshape)

    def is_linear(self):
        return True

    def pre_process(self, value, *args):
        if self.order == "C":
            return value, *args
        else:
            return value.T, *args


class NormOP(Operator):
    def __init__(self, ord=2):
        self.ord = 2

    def compute_shape(self, *dims: Dimension) -> Dimension:
        return Dimension(None)

    def is_linear(self):
        return False


class ClipOP(Operator):
    def __init__(self, a=None, a_min=None, a_max=None):
        self.lower = a_min
        self.higher = a_max
        _ = a

    def pre_process(self, *args):
        if len(args) == 1:
            return args[0]

        a, a_min, a_max = args
        self.lower = a_min
        self.higher = a_max
        return a

    def compute_shape(self, *dims: Dimension) -> Dimension:
        assert all(d.is_scalar() for d in dims)
        return Dimension(None)

    def is_linear(self):
        return False


compute_shape: Dict[OP, Callable[[Dimension, Dimension], Dimension]] = {}


def register_shape(*ops: OP):
    def inner(func):
        for op in ops:
            compute_shape[op] = func

        return func

    return inner


@register_shape(OP.EVALUATE)
def evaluate_shape(function_sig: FunctionSpace, *args: Dimension):

    if len(args) != len(function_sig.arguments):
        raise InvalidShape(
            f"Expected {len(function_sig.arguments)} arguments, got {len(args)}"
        )

    for i, (arg_dim, input_dim) in enumerate(
        zip(args, function_sig.input_dimensions())
    ):
        if arg_dim != input_dim:
            raise InvalidShape(
                f"Argument {i} has dimension {arg_dim.dim}, expected {input_dim.dim}"
            )
    try:
        (out_dim,) = function_sig.output_dimensions()
        return out_dim
    except ValueError:
        return function_sig.output_dimensions()


@register_shape(
    OP.VALUE,
    OP.NEG,
    OP.ABS,
    OP.SIN,
    OP.COS,
    OP.TAN,
    OP.ARCSIN,
    OP.ARCCOS,
    OP.ARCTAN,
)
def dimension_identity(dim: Dimension):
    return dim


@register_shape(OP.EQUAL, OP.LESS_EQUAL, OP.LESS_THAN)
def comparison_dimension(dim1: Dimension, dim2: Dimension):
    if dim1.dim != dim2.dim:
        raise InvalidShape(dim1, dim2)
    return dim1


@register_shape(OP.CASE)
def case_dimension(
    condition: Dimension, false_branch: Dimension, true_branch: Dimension
):
    if not condition.is_scalar():
        raise InvalidShape("Condition must be a scalar")

    if false_branch != true_branch:
        raise InvalidShape(
            "Arguments are of different dimensions", false_branch, true_branch
        )

    return true_branch


@register_shape(OP.ADD, OP.SUB, OP.ARCTAN2)
def same_dimension(d_1: Dimension, d_2: Dimension) -> Dimension:
    if d_1.is_scalar() and d_2.is_scalar():
        return d_1

    if d_1 != d_2:
        raise InvalidShape("Arguments are of different dimensions", d_1, d_2)

    return d_1


@register_shape(OP.MUL, OP.DIV)
def shape_mul(d_1: Dimension, d_2: Dimension):
    if d_1.is_scalar():
        return d_2

    if d_2.is_scalar():
        return d_1
    if d_1 == d_2:
        return d_1

    raise InvalidArgument(
        "Multiplication is not define between two non-scalars. "
        "Consider using other operations",
        d_1,
        d_2,
    )


@register_shape(OP.MATMUL)
def shape_matmul(d_1: Dimension, d_2: Dimension):

    if d_1.is_scalar() or d_2.is_scalar():
        raise InvalidArgument(
            "Matrix multiplication is not defined for scalars"
        )

    if d_1.is_vector():
        raise InvalidArgument("Cannot multiply vectors")

    c = d_1.dim[-1]
    out_dims = []
    # if d_1.is_matrix() or d_1.is_multilinear_map():
    try:
        out_dims += [d for d in d_1.dim[:-1]]
    except IndexError:
        pass

    r = d_2.dim[0]

    #    if d_2.is_matrix() or d_2.is_multilinear_map():
    try:
        out_dims += [d for d in d_2.dim[1:]]
    except IndexError:
        pass

    if c != r:
        raise InvalidArgument(
            "Cannot multiply: product axis has different shape",
            d_1.shape,
            d_2.shape,
        )

    if out_dims:
        try:
            return Dimension(tuple(out_dims))
        except TypeError as ex:
            raise ex
    else:
        return Dimension(None)


@register_shape(OP.EXP, OP.SQRT)
def scalar_shape(d: Dimension):
    if d.is_scalar():
        return d
    raise InvalidArgument("Can only operate on scalar dimensions.")


@register_shape(OP.CROSS)
def cross_shape(d_1: Dimension, d_2: Dimension):
    if d_1.dim != d_2.dim != (3,):
        raise InvalidArgument("Cross product is only defined for (3,) vectors")
    return Dimension((3,))


@register_shape(OP.DOT)
def dot_shape(d_1: Dimension, d_2: Dimension):
    if d_1.dim == d_2.dim and (d_1.is_vector() or d_1.is_covector()):
        return Dimension(None)

    raise InvalidArgument(
        "Dot product only defined for vectors from the same space."
    )


@register_shape(OP.TRANSPOSE)
def transpose_shape(d: Dimension):
    if d.is_scalar():
        return d

    if d.is_vector():
        return Dimension((1, d.dim[0]))
    if d.is_covector():
        return Dimension((d.dim[1],))

    return Dimension(tuple(reversed(d.dim)))


numpy_atomics = {
    np.cross: OP.CROSS,
    np.dot: OP.DOT,
    np.matmul: OP.MATMUL,
    np.add: OP.ADD,
    np.multiply: OP.MUL,
    np.subtract: OP.SUB,
    np.sin: OP.SIN,
    np.cos: OP.COS,
    np.tan: OP.TAN,
    np.exp: OP.EXP,
    np.power: OP.PWR,
    np.transpose: OP.TRANSPOSE,
    np.divide: OP.DIV,
    np.arccos: OP.ARCCOS,
    np.arcsin: OP.ARCSIN,
    np.sqrt: OP.SQRT,
    np.negative: OP.NEG,
    np.abs: OP.ABS,
    np.arctan2: OP.ARCTAN2,
}


numpy_composites = {
    np.concatenate: ConcatenateOP,
    np.reshape: ReshapeOP,
    np.linalg.norm: NormOP,
    np.clip: ClipOP,
}
