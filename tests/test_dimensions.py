import pytest

from coker import Dimension, FunctionSpace, function


def test_scalar_function_space_factory_creates_usable_declaration():
    space = FunctionSpace.create_scalar_function_space(
        "identity", continuity_index=(1,)
    )

    assert space.signature == (1,)
    assert space.input_dimensions() == [Dimension.scalar()]
    assert space.output_dimensions() == [Dimension.scalar()]

    identity = function(space.arguments, lambda value: value, backend="numpy")

    assert identity in space
    assert identity(2.5) == 2.5


@pytest.mark.parametrize("output", (None, [None]))
def test_function_space_rejects_absent_outputs(output):
    with pytest.raises(TypeError, match="output spaces"):
        FunctionSpace("invalid", arguments=[], output=output)
