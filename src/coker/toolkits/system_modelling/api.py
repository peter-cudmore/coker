from typing import List, Tuple
from coker.toolkits.system_modelling.modelling.coker_abc import (
    CokerListableSubclasses,
)


class ComponentHint:
    pass


def list_components() -> List[Tuple[str, str, ComponentHint]]:
    """Discover currently imported component subclasses.

    Import the desired standard-library modules before calling this function;
    it does not scan the filesystem or import libraries.

    Returns:
        list[tuple[str, str, None]]: Class name, slash-separated module path,
            and an unimplemented hint (always None).
    """
    entries = []
    for baseclass in CokerListableSubclasses.list_subclasses():
        for component in baseclass.list_subclasses():
            path = component.__module__.replace(".", "/")

            entries.append((component.__name__, path, None))

    return entries
