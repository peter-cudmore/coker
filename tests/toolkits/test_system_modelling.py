from coker.toolkits.system_modelling.std_lib import Gain, Integrator
from coker.toolkits.system_modelling import list_components


def test_list_components_reports_loaded_standard_library_components():
    components = list_components()
    for component in (Gain, Integrator):
        assert (
            component.__name__,
            "coker/toolkits/system_modelling/std_lib/common/signal_processing",
            None,
        ) in components
