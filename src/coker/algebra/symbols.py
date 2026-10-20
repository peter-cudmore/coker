"""Immutable callable metadata and weak tape-bound symbol handles."""

from __future__ import annotations

import dataclasses
import weakref
from typing import TYPE_CHECKING, Any

from coker.algebra.dimensions import FunctionSpace

if TYPE_CHECKING:
    from coker.algebra.graph import Tape


@dataclasses.dataclass(frozen=True)
class SymbolEntry:
    """Store callable metadata with an explicit interning identity.

    Args:
        target: Callable or backend target interpreted during evaluation.
        function_space: Declared argument and result spaces.
        name: Optional display name for symbolic external calls.
        _identity: Hashable identity used for equality and interning.

    Examples:
        Create metadata before interning it on a tape::

            entry = SymbolEntry(target, space, "external", (id(target),))
            symbol = tape.intern_symbol(entry)
    """

    target: Any
    function_space: FunctionSpace
    name: str | None
    _identity: tuple[Any, ...] = dataclasses.field(repr=False, compare=False)

    def __hash__(self) -> int:
        return hash(self._identity)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SymbolEntry) and (self._identity == other._identity)


class FunctionSymbol:
    """Refer weakly to an interned callable without owning its tape.

    Args:
        tape: Owning tape, which must outlive this handle.
        archive_index: Index of the callable's metadata in the tape.

    Examples:
        Obtain a handle through its owning tape::

            symbol = tape.intern_symbol(entry)
            target = symbol.target
            space = symbol.function_space
            name = symbol.symbol_name
    """

    def __init__(self, tape: Tape, archive_index: int):
        self._tape = weakref.ref(tape)
        self._archive_index = archive_index

    @property
    def _entry(self) -> SymbolEntry:
        return self._tape().nodes._symbol_table[self._archive_index]

    @property
    def target(self) -> Any:
        """Return the target interpreted by a backend.

        Returns:
            The callable or backend target stored in the symbol's metadata.
        """
        return self._entry.target

    @property
    def function_space(self) -> FunctionSpace:
        """Return the callable's declared argument and result spaces.

        Returns:
            The function space stored in the symbol's metadata.
        """
        return self._entry.function_space

    @property
    def symbol_name(self) -> str:
        """Return the stable name used for symbolic external calls.

        Returns:
            The declared display name, or a name derived from the archive index.
        """
        return self._entry.name or f"function_{self._archive_index}"
