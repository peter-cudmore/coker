import dataclasses
from typing import List, Optional, Tuple, Union


@dataclasses.dataclass
class VectorSpace:
    name: str
    dimension: Union[int, Tuple[int, ...]]


@dataclasses.dataclass
class Scalar:
    name: str


class Dimension:
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

        mods = []

        count = 1
        iterator = reversed(self.dim) if row_major else iter(self.dim)

        for d in iterator:
            count = d * count
            mods.insert(0, count)

        for i in range(count):
            yield tuple(i % m for m in mods)

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
