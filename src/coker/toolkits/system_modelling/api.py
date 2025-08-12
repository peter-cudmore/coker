from typing import List, Tuple
from coker.toolkits.system_modelling.modelling.coker_abc import (
    CokerListableSubclasses,
)


class ComponentHint:
    pass


def list_components() -> List[Tuple[str, str, ComponentHint]]:
    """List all components

    Returns:
        list of  component name, library path, usage hint

    """
    entries = []
    for baseclass in CokerListableSubclasses.list_subclasses():
        for component in baseclass.list_subclasses():
            path = component.__module__.replace(".", "/")

            entries.append((component.__name__, path, None))

    return entries
