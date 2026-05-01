import coker.toolkits.system_modelling.std_lib as std_lib
from coker.toolkits.system_modelling.api import list_components


def test_list_libraries():
    assert std_lib is not None
    components = list_components()
    assert len(components) > 4

    for component, library_path, _hint in components:
        assert library_path.startswith(
            "coker/toolkits/system_modelling/std_lib"
        ), (
            "Only components in the standard library should be "
            f"imported: {library_path}"
        )
