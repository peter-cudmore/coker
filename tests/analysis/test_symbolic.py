import pytest
import sympy as sp

from coker.algebra.dimensions import Dimension
from coker.dynamics.analysis.symbolic import (
    UnsupportedSystemError,
    _expressions_for_dimension,
)


def test_symbolic_analysis_flattens_declared_sympy_arrays():
    state_0, state_1 = sp.symbols("state_0 state_1")

    values = _expressions_for_dimension(
        sp.Array([state_0, state_1]), Dimension(2), "state"
    )

    assert values == (state_0, state_1)


def test_symbolic_analysis_rejects_duck_shaped_values():
    class DuckShapedValue:
        shape = (2,)

        def __getitem__(self, index):
            return sp.Symbol(f"value_{index}")

        def _sympy_(self):
            return sp.Symbol("duck")

    with pytest.raises(UnsupportedSystemError, match="unsupported symbolic"):
        _expressions_for_dimension(DuckShapedValue(), Dimension(2), "state")
