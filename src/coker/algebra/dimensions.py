from __future__ import annotations

import dataclasses
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
from coker.interfaces import FunctionSignatureValue
from coker.algebra.exceptions import InvalidArgument, InvalidShape


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
    :func:`~coker.algebra.graph.get_projection` when constructing slice
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
        # column-major (row_major=False): first index varies fastest →
        # strides [1, d0, d0*d1, ...]
        # row-major (row_major=True): last index varies fastest →
        # strides [..., d1, 1]
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

    @staticmethod
    def scalar():
        return Dimension(None)


@dataclasses.dataclass(frozen=True)
class ResultBundleDimension:
    """Ordered dimensions returned by one multi-output callable."""

    outputs: tuple[Dimension | FunctionSpace, ...]

    def select(self, index: int) -> Dimension | FunctionSpace:
        return self.outputs[index]


@dataclasses.dataclass
class FunctionSpace:
    """Represents a function space, including its domain and codomain.

    Attributes:
        name (str): The name of the function space.
        arguments (List[Scalar | VectorSpace | FunctionSpace]): A list
            specifying the input arguments.
        output (List[Scalar | VectorSpace | FunctionSpace]): A list
            specifying ordered outputs.
        signature (Optional[Tuple[int]]): Optional list of integers
            denoting the degree of differentiability for each argument.
            If not provided, the default is smooth (infinitely
            differentiable) functions.

    """

    name: str
    arguments: List[Scalar | VectorSpace | FunctionSpace]
    output: List[Scalar | VectorSpace | FunctionSpace]
    signature: Optional[Tuple[int]] = None
    """Optional list of integers specifying the degree of
    differentiability for each argument.

    Defaults to infinite (that is, smooth functions).
    """

    def __post_init__(self) -> None:
        if self.output is None or any(output is None for output in self.output):
            raise TypeError("FunctionSpace output spaces must not be None")

    def input_dimensions(self):
        return [
            (
                arg
                if isinstance(arg, FunctionSpace)
                else (
                    Dimension.scalar()
                    if isinstance(arg, Scalar)
                    else Dimension(arg.dimension)
                )
            )
            for arg in self.arguments
        ]

    def output_dimensions(self) -> list[Dimension | FunctionSpace]:
        return [
            (
                out
                if isinstance(out, FunctionSpace)
                else (
                    Dimension.scalar()
                    if isinstance(out, Scalar)
                    else Dimension(out.dimension)
                )
            )
            for out in self.output
        ]

    def matches_signature(self, other: FunctionSpace) -> bool:
        """Return whether another function space has the same I/O shapes."""
        if not isinstance(other, FunctionSpace):
            return False
        input_dimensions_match = tuple(self.input_dimensions()) == tuple(
            other.input_dimensions()
        )
        output_dimensions_match = tuple(self.output_dimensions()) == tuple(
            other.output_dimensions()
        )
        return input_dimensions_match and output_dimensions_match

    def validate_argument(
        self,
        argument_dimension: Dimension | FunctionSpace,
        position: int,
    ) -> None:
        """Validate one argument against this callable's declared input."""
        if type(position) is not int:
            raise InvalidArgument("argument position must be an integer")
        input_dimensions = self.input_dimensions()
        if position < 0 or position >= len(input_dimensions):
            raise InvalidArgument(
                f"argument position {position} is outside the callable "
                f"signature with {len(input_dimensions)} arguments"
            )
        actual_dimension = argument_dimension
        expected_dimension = input_dimensions[position]
        if isinstance(actual_dimension, FunctionSpace) and isinstance(
            expected_dimension, FunctionSpace
        ):
            dimensions_match = actual_dimension.matches_signature(expected_dimension)
        elif isinstance(actual_dimension, Dimension) and isinstance(
            expected_dimension, Dimension
        ):
            dimensions_match = actual_dimension == expected_dimension
        else:
            dimensions_match = False
        if not dimensions_match:
            raise InvalidShape(
                f"argument {position} has dimension {actual_dimension!r}, "
                f"expected {expected_dimension!r}"
            )

    def bind_argument(
        self,
        argument_dimension: Dimension | FunctionSpace,
        position: int,
    ) -> FunctionSpace:
        """Return this callable signature with one argument bound."""
        try:
            self.validate_argument(argument_dimension, position)
        except InvalidShape as exc:
            raise InvalidShape(f"BIND {exc}") from exc
        except InvalidArgument as exc:
            raise InvalidArgument(f"BIND {exc}") from exc
        signature = self.signature
        if signature is not None:
            signature = signature[:position] + signature[position + 1 :]
        return FunctionSpace(
            self.name,
            arguments=[
                argument
                for index, argument in enumerate(self.arguments)
                if index != position
            ],
            output=list(self.output),
            signature=signature,
        )

    def __contains__(self, value) -> bool:
        """Return whether a Coker function has this input/output signature."""
        if not isinstance(value, FunctionSignatureValue):
            return False
        input_dimensions_match = tuple(value.input_shape()) == tuple(
            self.input_dimensions()
        )
        output_dimensions_match = tuple(
            dimension for dimension in value.output_shape() if dimension is not None
        ) == tuple(self.output_dimensions())
        return input_dimensions_match and output_dimensions_match

    def evaluation_dimension(
        self,
    ) -> Dimension | FunctionSpace | ResultBundleDimension:
        """Return the graph result shape declared by this callable."""
        output_dimensions = self.output_dimensions()
        return (
            output_dimensions[0]
            if len(output_dimensions) == 1
            else ResultBundleDimension(tuple(output_dimensions))
        )

    def is_scalar(self):
        output_dimensions = self.output_dimensions()
        if len(output_dimensions) != 1:
            return False
        (output_dimension,) = output_dimensions
        is_dimension = isinstance(output_dimension, Dimension)
        return is_dimension and output_dimension.is_scalar()

    @staticmethod
    def create_scalar_function_space(
        name: str, continuity_index: Optional[Tuple[int]] = None
    ):
        return FunctionSpace(
            name=name,
            arguments=[Scalar(f"{name}_input")],
            output=[Scalar(f"{name}_output")],
            signature=continuity_index,
        )


@dataclasses.dataclass
class Element:
    parent: VectorSpace
