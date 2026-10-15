from coker.toolkits.system_modelling.modelling.coker_abc import (
    CokerListableSubclasses,
)


def list_components() -> list[tuple[str, str, None]]:
    """Return registered components as ``(name, library_path, None)`` tuples.

    Import component libraries before discovery to register their classes.
    The third entry is always ``None``; usage hints are not implemented.
    """
    entries = []
    for baseclass in CokerListableSubclasses.list_subclasses():
        for component in baseclass.list_subclasses():
            path = component.__module__.replace(".", "/")

            entries.append((component.__name__, path, None))

    return entries
