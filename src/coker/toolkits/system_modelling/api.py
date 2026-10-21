from coker.toolkits.system_modelling.modelling.coker_abc import (
    CokerListableSubclasses,
)


def list_components() -> list[tuple[str, str, None]]:
    """Discover currently imported component subclasses.

    Import component libraries before calling this function; it does not scan
    the filesystem or import libraries.

    Returns:
        Class name, slash-separated module path and hint tuples. The hint is
        always None; usage hints are not implemented.

    Examples:
        >>> from coker.toolkits.system_modelling import list_components
        >>> from coker.toolkits.system_modelling.std_lib.common import (
        ...     signal_processing,
        ... )
        >>> any(name == "Gain" for name, path, hint in list_components())
        True
    """
    entries = []
    for baseclass in CokerListableSubclasses.list_subclasses():
        for component in baseclass.list_subclasses():
            path = component.__module__.replace(".", "/")

            entries.append((component.__name__, path, None))

    return entries
