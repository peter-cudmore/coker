import dataclasses
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union


@dataclasses.dataclass
class VectorSpace:
    """An argument space representing a finite-dimensional vector.

    Args:
        name: Identifier used in error messages and tape node labels.
        dimension: Size of the vector.  An ``int`` for a 1-D vector; a tuple
            of ints for a multi-dimensional array (e.g. ``(3, 3)`` for a
            3×3 matrix).

    Attributes:
        size: Total number of scalar elements (product of all dimensions).
    """

    name: str
    dimension: Union[int, Tuple[int, ...]]

    @property
    def size(self) -> int:
        if isinstance(self.dimension, int):
            return self.dimension
        return reduce(mul, self.dimension)


@dataclasses.dataclass
class Scalar:
    """An argument space representing a single scalar value.

    Args:
        name: Identifier used in error messages and tape node labels.
    """

    name: str

    @property
    def size(self) -> int:
        return 1


class Dimension:
    """Shape descriptor for a single value in the computation graph.

    Wraps a tuple of ints (for array-valued nodes) or ``None`` (for scalars).
    Used internally to track shapes through the tape and by
    :func:`~coker.algebra.kernel.get_projection` when constructing slice
    projections.

    Args:
        tuple_or_none: Shape as a tuple of ints, a single int (converted to a
            1-tuple), or ``None`` for a scalar.
    """

    def __init__(self, tuple_or_none):

        if isinstance(tuple_or_none, int):
            tuple_or_none = (tuple_or_none,)

        assert tuple_or_none is None or isinstance(tuple_or_none, tuple)
        self.dim = tuple_or_none

    def flat(self):
        d = 1
        if self.dim is not None:
            for d_i in self.dim:
                d *= d_i
        return d

    def to_space(self, name) -> Union[Scalar, VectorSpace]:
        if self.is_scalar():
            return Scalar(name)
        return VectorSpace(name, self.dim)

    def index_iterator(self, row_major=False):

        if not self.dim:
            return (0,)

        count = 1
        for d in self.dim:
            count *= d

        # Compute per-dimension strides for mixed-radix decomposition.
        # column-major (row_major=False): first index varies fastest → strides [1, d0, d0*d1, ...]
        # row-major (row_major=True):    last  index varies fastest → strides [..., d1, 1]
        strides = []
        stride = 1
        if row_major:
            for d in reversed(self.dim):
                strides.insert(0, stride)
                stride *= d
        else:
            for d in self.dim:
                strides.append(stride)
                stride *= d

        for i in range(count):
            yield tuple((i // s) % d for s, d in zip(strides, self.dim))

    def __eq__(self, other):
        return self.dim == other.dim

    def is_scalar(self):
        return self.dim is None

    def is_vector(self):
        return self.dim is not None and len(self.dim) == 1

    def is_covector(self):
        return not self.is_scalar() and len(self.dim) == 2 and self.dim[0] == 1

    def is_matrix(self):
        return not self.is_scalar() and len(self.dim) == 2 and self.dim[0] > 1

    def is_multilinear_map(self):
        return isinstance(self.dim, tuple) and len(self.dim) > 2

    def __iter__(self):
        return iter(self.dim)

    def __repr__(self):
        if self.dim is None:
            return "R"

        return repr(self.dim)

    @property
    def shape(self):
        if self.dim is None:
            return (1,)
        return self.dim


@dataclasses.dataclass
class FunctionSpace:
    """Represents a function space, including its domain and codomain.

    Attributes:
        name (str): The name of the function space.
        arguments (List[Scalar | VectorSpace]): A list specifying the input arguments of
            the function, where each argument can be either a scalar or a vector space.
        output (List[Scalar | VectorSpace]): A list specifying the output of the function,
            where each element can be either a scalar or a vector space.
        signature (Optional[Tuple[int]]): Optional list of integers denoting the degree
            of differentiability for each argument. If not provided, the default is
            smooth (infinitely differentiable) functions.

    """

    name: str
    arguments: List[Scalar | VectorSpace]
    output: List[Scalar | VectorSpace]
    signature: Optional[Tuple[int]] = None
    """Optional list of integers, specifying the degree of differentiability for each argument. Defaults to infinite (i.e. smooth function)."""

    def input_dimensions(self):
        return [
            (
                Dimension(None)
                if isinstance(arg, Scalar)
                else Dimension(arg.dimension)
            )
            for arg in self.arguments
        ]

    def output_dimensions(self):
        if self.output is None:
            return (None,)
        return [
            (
                Dimension(None)
                if isinstance(out, Scalar)
                else Dimension(out.dimension)
            )
            for out in self.output
        ]

    def is_scalar(self):
        return (
            len(self.output_dimensions()) == 1
            and self.output_dimensions()[0].is_scalar()
        )


@dataclasses.dataclass
class Element:
    parent: VectorSpace
