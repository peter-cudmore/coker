"""Compatibility facade for Coker's algebra graph and function APIs."""

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.function import (
    BoundCallable,
    Function,
    InequalityExpression,
    SymbolicCallable,
    create_function_from_native,
    function,
)
from coker.algebra.graph import (
    CallableReference,
    DanglingTracerError,
    Tape,
    TapeInner,
    TraceContext,
    Tracer,
    get_basis,
    get_dim_by_class,
    get_projection,
    if_then_else,
    normalise,
    strip_symbols_from_array,
)
from coker.algebra.ops import OP, Noop

__all__ = [
    "BoundCallable",
    "CallableReference",
    "DanglingTracerError",
    "Dimension",
    "Function",
    "FunctionSpace",
    "InequalityExpression",
    "Noop",
    "OP",
    "Scalar",
    "SymbolicCallable",
    "Tape",
    "TapeInner",
    "TraceContext",
    "Tracer",
    "VectorSpace",
    "create_function_from_native",
    "function",
    "get_basis",
    "get_dim_by_class",
    "get_projection",
    "if_then_else",
    "normalise",
    "strip_symbols_from_array",
]
