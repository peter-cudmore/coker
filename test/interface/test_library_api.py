from coker import Block, BlockSpec, Signal
from coker.api import list_components



def test_list_libraries():
    components = list_components()
    assert len(components) > 4

    for component, library_path, _hint in components:
        assert library_path.startswith('coker/std_lib'), \
            f"Only components in the standard library should be imported: {library_path}"