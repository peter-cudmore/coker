from dataclasses import dataclass
from typing import Callable

import numpy as np

from coker.algebra.dimensions import Scalar
from coker.parameters import (
    ParameterMixin as _ParameterMixin,
    ValueType as _ValueType,
)


@dataclass
class PiecewiseConstantVariable(_ParameterMixin):
    """Declare one scalar control channel with uniformly spaced coefficients.

    Variational transcription uses normalised time ``tau = t / t_final`` on
    ``[0, 1]``. This declaration does not attach units or rescale values:
    coefficients and bounds have the physical units of the model input.

    Args:
        name: Name identifying this control channel.
        sample_rate: Positive number of coefficients per unit normalised time,
            not samples per second. The coefficient spacing is
            ``t_final / sample_rate`` in physical time.
        upper_bound: Inclusive upper bound for every coefficient.
        lower_bound: Inclusive lower bound for every coefficient.

    Examples:
        >>> import numpy as np
        >>> from coker.dynamics import PiecewiseConstantVariable
        >>> control = PiecewiseConstantVariable("force", 2.0, lower_bound=0.0)
        >>> control.degrees_of_freedom(0.0, 1.0)
        2
        >>> force = control.to_solution(np.array([1.0, 3.0]))
        >>> float(force(0.75))
        3.0
    """

    name: str
    sample_rate: float
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        """Return ``ceil((end - start) * sample_rate)`` scalar decisions.

        Args:
            *interval: Exactly ``(start, end)`` in normalised time;
                transcription uses ``(0, 1)``. The start affects only the count,
                not the solution's time origin.

        Examples:
            >>> from coker.dynamics import PiecewiseConstantVariable
            >>> PiecewiseConstantVariable("u", 2.5).degrees_of_freedom(0, 1)
            3
        """
        start, end = interval
        return int(np.ceil((end - start) * self.sample_rate))

    def to_solution(self, value):
        """Wrap coefficients in a :class:`PiecewiseControlSolution`.

        Args:
            value: Indexable coefficients in input units, ordered from
                normalised time zero with one entry per degree of freedom.
                Retained without copying, reshaping or bound checks; the
                callable preserves each indexed entry's scalar/row shape.

        Examples:
            >>> import numpy as np
            >>> from coker.dynamics import PiecewiseConstantVariable
            >>> control = PiecewiseConstantVariable("u", 2)
            >>> float(control.to_solution(np.array([2.0, 4.0]))(0.5))
            4.0
        """
        return PiecewiseControlSolution(self.name, self.sample_rate, value)


@dataclass
class SpikeVariable(_ParameterMixin):
    """Declare one scalar pointwise spike amplitude.

    This is not a Dirac impulse: it supplies the amplitude only at evaluation
    sites within ``1e-9`` of ``time`` and zero elsewhere. Its amplitude has
    the model input's units, not input-times-time units.

    Args:
        name: Name identifying this control channel.
        time: Fixed spike location in normalised time ``t / t_final``.
        upper_bound: Inclusive upper bound on the amplitude.
        lower_bound: Inclusive lower bound on the amplitude.

    Examples:
        >>> from coker.dynamics import SpikeVariable
        >>> control = SpikeVariable("pulse", time=0.5)
        >>> control.degrees_of_freedom(0, 1)
        1
        >>> pulse = control.to_solution(2.0)
        >>> (pulse(0.5), pulse(0.0))
        (2.0, 0.0)
    """

    name: str
    time: float
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        """Return one amplitude decision, even if the spike is outside the interval.

        Args:
            *interval: Optional normalised endpoints; ignored.

        Examples:
            >>> from coker.dynamics import SpikeVariable
            >>> SpikeVariable("u", 0.5).degrees_of_freedom(0, 1)
            1
        """
        return 1

    def to_solution(self, value):
        """Return a :class:`SpikeControlSolution` with tolerance ``1e-9``.

        Args:
            value: Scalar amplitude in input units, stored without conversion
                or bound checks. The spike retains this declaration's name
                and normalised-time location.

        Examples:
            >>> from coker.dynamics import SpikeVariable
            >>> SpikeVariable("u", 0.5).to_solution(3.0)(0.5)
            3.0
        """
        return SpikeControlSolution(self.name, self.time, value)


@dataclass
class ConstantControlVariable(_ParameterMixin):
    """Declare one scalar control amplitude that is constant over the horizon.

    Args:
        name: Name identifying this control channel.
        upper_bound: Inclusive upper bound in the model input's units.
        lower_bound: Inclusive lower bound in the model input's units.

    Examples:
        >>> from coker.dynamics import ConstantControlVariable
        >>> control = ConstantControlVariable("force", lower_bound=0.0)
        >>> control.degrees_of_freedom(0, 1)
        1
        >>> control.to_solution(2.0)(0.75)
        2.0
    """

    name: str
    upper_bound: float = np.inf
    lower_bound: float = -np.inf

    def degrees_of_freedom(self, *interval):
        """Return one amplitude decision.

        Args:
            *interval: Optional normalised endpoints; ignored.

        Examples:
            >>> from coker.dynamics import ConstantControlVariable
            >>> ConstantControlVariable("u").degrees_of_freedom(0, 1)
            1
        """
        return 1

    def to_solution(self, value):
        """Return a :class:`ConstantControlSolution` retaining ``value``.

        Args:
            value: Amplitude in input units, stored without conversion or bound
                checks. Its scalar or backend-native one-element shape is
                preserved.

        Examples:
            >>> from coker.dynamics import ConstantControlVariable
            >>> ConstantControlVariable("u").to_solution(3.0)(0.25)
            3.0
        """
        return ConstantControlSolution(self.name, value)


@dataclass
class ConstantControlSolution:
    """Represent a constant scalar control channel.

    Args:
        name: Control channel name.
        value: Constant amplitude in the model input's units. Stored without
            numeric conversion; backend-produced values can be one-element
            containers rather than Python floats.

    Examples:
        >>> from coker.dynamics import ConstantControlSolution
        >>> control = ConstantControlSolution("force", 2.0)
        >>> control(0.5)
        2.0
    """

    name: str
    value: float

    def __call__(self, t):
        """Return the stored amplitude without changing its shape.

        Args:
            t: Scalar normalised time ``t_physical / t_final``. Ignored;
                this representation imposes no interval restriction.

        Returns:
            Stored scalar amplitude, or its original backend-native
            scalar/one-element container, in model input units.

        Examples:
            >>> from coker.dynamics import ConstantControlSolution
            >>> ConstantControlSolution("u", 2.0)(0.75)
            2.0
        """
        return self.value


@dataclass
class SpikeControlSolution:
    """Represent a scalar pointwise spike, not a distributional impulse.

    Args:
        name: Control channel name.
        time: Spike location in normalised time ``t_physical / t_final``.
        value: Amplitude in model input units, stored without conversion.
        tolerance: Strict proximity threshold in normalised time. Its physical
            width scales with ``t_final``.

    Examples:
        >>> from coker.dynamics import SpikeControlSolution
        >>> pulse = SpikeControlSolution("pulse", time=0.5, value=3.0)
        >>> (pulse(0.5), pulse(0.25))
        (3.0, 0.0)
    """

    name: str
    time: float
    value: float
    tolerance: float = 1e-9

    def __call__(self, t):
        """Return the amplitude only within the spike tolerance.

        Args:
            t: Scalar normalised time, not a physical time or time grid.

        Returns:
            Stored amplitude if ``abs(t - time) < tolerance``, otherwise
            ``0.0``. The amplitude retains its original scalar or backend-native
            one-element shape; zero is a Python float.

        Examples:
            >>> from coker.dynamics import SpikeControlSolution
            >>> SpikeControlSolution("u", 0.5, 2.0)(0.0)
            0.0
        """
        return self.value if abs(t - self.time) < self.tolerance else 0.0


@dataclass
class PiecewiseControlSolution:
    """Represent one scalar channel by uniform normalised-time coefficients.

    Args:
        name: Control channel name.
        sample_rate: Coefficients per unit normalised time, not hertz.
        value: Indexable coefficients ordered from normalised time zero.
            A one-dimensional NumPy array yields NumPy scalars. Other
            containers retain the shape returned by their indexing operation.
            Coefficients have the model input's units.

    No time-origin shift, interpolation, endpoint clipping or unit conversion
    is applied. For ``n`` coefficients and positive ``sample_rate``, use
    ``0 <= t < n / sample_rate``. In particular, an exactly filled interval's
    right endpoint is excluded.

    Examples:
        >>> import numpy as np
        >>> from coker.dynamics import PiecewiseControlSolution
        >>> control = PiecewiseControlSolution("u", 2, np.array([1.0, 4.0]))
        >>> [float(control(t)) for t in (0.0, 0.49, 0.5, 0.99)]
        [1.0, 1.0, 4.0, 4.0]
    """

    name: str
    sample_rate: float
    value: np.ndarray

    def __call__(self, t):
        """Select the coefficient at a scalar normalised time.

        Args:
            t: Normalised time ``t_physical / t_final``. Use a nonnegative
                scalar whose index ``int(t * sample_rate)`` is in range.

        Returns:
            The indexed coefficient in model input units, with the container's
            indexing shape; a one-dimensional NumPy array returns a NumPy
            scalar, not a length-one array.

        Raises:
            AssertionError: The index is at least the coefficient count.

        Negative times are not explicitly rejected and can use the
        container's negative-index semantics; they are not extrapolation.

        Examples:
            >>> import numpy as np
            >>> from coker.dynamics import PiecewiseControlSolution
            >>> float(PiecewiseControlSolution("u", 2, np.array([1., 4.]))(0.5))
            4.0
        """
        idx = int(t * self.sample_rate)
        assert idx < len(self.value), f"Time {t} is outside of the interval"
        return self.value[idx]


ControlSolution = (
    SpikeControlSolution | PiecewiseControlSolution | ConstantControlSolution
)
ControlLaw = Callable[[Scalar], _ValueType]
ControlVariable = ConstantControlVariable | PiecewiseConstantVariable | SpikeVariable

Solution = "DynamicalSystem" | Callable[[Scalar, ControlLaw, _ValueType], Scalar]
LossFunction = Callable[[Solution, ControlLaw, _ValueType], Scalar]
