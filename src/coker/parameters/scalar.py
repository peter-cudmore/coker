"""Scalar parameter and decision declarations."""

import abc
from dataclasses import dataclass
from typing import Union

import numpy as np

from coker.algebra.dimensions import Scalar, VectorSpace


class ParameterMixin(abc.ABC):
    """Define the scalar decision-count interface for parameter declarations."""

    @abc.abstractmethod
    def degrees_of_freedom(self, *interval) -> int:
        """Return the number of independent scalar decisions.

        Args:
            *interval: Optional interval information interpreted by a subclass.
                Scalar and dense declarations ignore it.

        Returns:
            int: Number of scalar decisions represented by the declaration.
        """
        pass


@dataclass
class BoundedVariable(ParameterMixin):
    """Declare one scalar decision with lower and upper bounds.

    Args:
        name: Key used to identify the fitted value in named parameter results.
        lower_bound: Inclusive lower bound in the same units as the decision.
        upper_bound: Inclusive upper bound in the same units as the decision.
        guess: Initial scalar value. Defaults to zero.

    Construction does not validate finiteness, bound ordering or whether the
    guess lies within the bounds. Supply consistent numeric values and units;
    validation is the responsibility of the consuming builder.

    Examples:
        >>> from coker.parameters import BoundedVariable
        >>> gain = BoundedVariable("gain", 0.0, 2.0, guess=1.0)
        >>> (gain.name, gain.degrees_of_freedom())
        ('gain', 1)
    """

    name: str
    lower_bound: float
    upper_bound: float
    guess: float = 0

    def degrees_of_freedom(self, *interval):
        """Return one scalar decision, ignoring optional interval arguments.

        Args:
            *interval: Accepted for the decision-count interface; ignored.

        Returns:
            int: One.
        """
        return 1


@dataclass
class UnboundedVariable(ParameterMixin):
    """Declare one unbounded scalar decision.

    Args:
        name: Key used to identify the fitted value in named parameter results.
        guess: Initial scalar value. Defaults to zero; supply a finite value
            in the units expected by the consuming model.

    Construction does not validate the guess or attach physical units.

    Examples:
        >>> from coker.parameters import UnboundedVariable
        >>> offset = UnboundedVariable("offset", guess=3.0)
        >>> (offset.degrees_of_freedom(), offset.guess)
        (1, 3.0)
    """

    name: str
    guess: float = 0

    def degrees_of_freedom(self, *interval):
        """Return one scalar decision, ignoring optional interval arguments.

        Args:
            *interval: Accepted for the decision-count interface; ignored.

        Returns:
            int: One.
        """
        return 1

    @property
    def upper_bound(self):
        """float: Positive infinity, indicating no upper bound."""
        return np.inf

    @property
    def lower_bound(self):
        """float: Negative infinity, indicating no lower bound."""
        return -np.inf


Constant = Union[float, int]
ValueType = Scalar | VectorSpace
