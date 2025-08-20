from functools import reduce
from operator import mul
from typing import List, Tuple, Callable, Optional, Iterator

import numpy as np


from coker.dynamics.types import (
    TranscriptionOptions,
    ControlVariable,
    PiecewiseConstantVariable,
    SpikeVariable,
)


def legendre_coefficient(n, k):
    """Compute the coefficient of the kth Legendre polynomial of degree n."""
    if k > n:
        return 0
    if (n, k) == (0, 0):
        return 1
    if (n, k) == (1, 1):
        return 1
    if k <= 1:
        if (k == 1 and n % 2 == 0) or (k == 0 and n % 2 == 1):
            return 0
        coeff = -(n + k - 1) / (n - k)
        return coeff * legendre_coefficient(n - 2, k)

    coeff = -((n - k + 2) * (n + k - 1) / (k * (k - 1)))
    return coeff * legendre_coefficient(n, k - 2)


def expand_coefficients(roots: List[float]):
    """Expand polynomials from root form to coefficient form."""
    # p(x) = (x - x_0) (x - x_1) ...  (x  - x_n)
    #
    # final case:
    #   p(x) = [] , r(x) = [a_0, a_1, ... a_n, 1]
    #   where r(x) = sum_i(x^i a_i)
    #   -> return r(x)

    # (x - r_0)(x - r_1) (\sum c_i x^i)
    #  -> (x - r_0)( \sum( (r_1 c_i + c_{i-1}) x^i)

    # initial case: p(x) = coeffs, r(x) = [1]
    coeffs = [1]
    while roots:
        root = roots.pop()
        coeffs = [root * coeffs[0]] + [
            root * c + c_last for c, c_last in zip(coeffs[1:] + [0], coeffs)
        ]
    return coeffs


def lgr_points(n: int) -> List[float]:
    """Compute LGR collocation points for a given order n."""
    assert n > 0
    # Points are roots of P_n(x) + P_{n-1}(x)
    # Where P_n(x) is the nth Legendre Polynomial

    companion_matrix = np.diag(np.ones(n - 1), 1)
    leading_term = legendre_coefficient(n, n)
    for i in range(n):
        companion_matrix[-1, i] = (
            -(legendre_coefficient(n - 1, i) + legendre_coefficient(n, i))
            / leading_term
        )
    roots = np.linalg.eigvals(companion_matrix).tolist()
    roots.append(1)
    roots.sort()

    return [-1] + roots[1:]


def evaluate_legendre_polynomial(x, n):
    return np.polynomial.legendre.legval(x, [0] * n + [1])


def generate_discritisation_operators(
    interval: Tuple[float, float], n: int
) -> Tuple[
    List[float],
    Callable[[float], float],
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
]:
    """
    Generates discretisation operators over an interval using Legendre-Gauss-Radau (LGR)
    collocation points. This function computes knot points, Legendre polynomial bases,
    as well as derivative and integral operators for numerical methods.

    Args:
        interval: A tuple representing the interval [a, b] over which the discretisation
            is computed.
        n: The number of discretisation points.

    Returns:
        Tuple containing:
            List[float]: Knot points of length (n + 1).
            Callable[[float], float]: Transformed time variable.
            List[np.ndarray]: Legendre polynomial bases at each point.
            List[np.ndarray]: Derivative operators at each knot point.
            np.ndarray: Integral operator for the interval.
    """
    colocation_times = np.array(lgr_points(n))

    # using n LRG collocation points covering [-1, 1)
    # plus an additional non-collocated point a +1
    #

    bases = np.empty((n + 1, n + 1))
    time_scaling_factor = (interval[1] - interval[0]) / 2
    colocation_coeff = np.zeros((n + 1, n + 1))
    continuity_coeff = np.zeros(n + 1)
    quad_coeff = np.zeros_like(continuity_coeff)

    for i in range(n + 1):
        tau_i = colocation_times[i]
        factors = [
            np.poly1d([1, -tau_j]) / (tau_i - tau_j)
            for tau_j in colocation_times
            if tau_i != tau_j
        ]
        basis_i = reduce(mul, factors)
        bases[:, i] = basis_i.c[::-1]
        dbasis_i = np.polyder(basis_i)

        continuity_coeff[i] = basis_i(1)
        colocation_coeff[i, :] = [
            dbasis_i(tau_j) / time_scaling_factor for tau_j in colocation_times
        ]
        quad_coeff[i] = np.polyint(basis_i)(1.0) * time_scaling_factor
    # see https://mathworld.wolfram.com/RadauQuadrature.html
    weights = (
        np.array(
            [2 / n**2]
            + [
                (1 - x_i) / (n * evaluate_legendre_polynomial(x_i, n - 1)) ** 2
                for x_i in colocation_times[1:-1]
            ]
            + [0]
        )
        * time_scaling_factor
    )

    t = lambda tau: (interval[0] + interval[1]) / 2 + tau * time_scaling_factor

    derivative: List[np.ndarray] = [row for row in colocation_coeff.T]
    bases: List[np.ndarray] = [row for row in bases]
    integral_weights: np.ndarray = weights.reshape(1, n + 1)
    return colocation_times, t, bases, derivative, integral_weights


def split_at_non_differentiable_points(
    control_variables: List[ControlVariable],
    t_final: float,
    transcription_options: TranscriptionOptions,
    additional_points: Optional[List[float]] = None,
) -> List[Tuple[float, float]]:
    """
    Splits the time domain into intervals taking into account control variables, additional
    integration points, and transcription requirements.

    The function determines interval boundaries based on the characteristics and constraints
    of the provided control variables. It also considers any additional points passed, ensuring
    subdivision complies with the transcription options. The intervals are adjusted to guarantee
    a minimum required number and their lengths are subdivided iteratively when necessary.

    Args:
        control_variables (List[ControlVariable]): A list of control variables, which define
            parameters affecting the differentiation process. This may include variables
            such as piecewise constants or spike events.
        t_final (float): The total duration of the time domain or integration window.
        transcription_options (TranscriptionOptions): Configuration options specifying
            transcription constraints, such as the minimum number of intervals.
        additional_points (Optional[List[float]]): Additional time points that should be
            included as interval boundaries, if provided.

    Returns:
        List[Tuple[float, float]]: A list of tuples representing the sorted interval boundaries.
            Each tuple contains the start and end of an interval.
    """
    interval_boundaries = (
        set(additional_points) if additional_points else set()
    )
    for d in control_variables:
        if isinstance(d, PiecewiseConstantVariable):
            assert d.sample_rate > 0, "Sample rate must be positive"
            steps = t_final * d.sample_rate
            interval_boundaries |= {
                i * t_final / steps for i in range(int(steps))
            }

        if isinstance(d, SpikeVariable):
            assert (
                0 <= d.time < t_final
            ), "Spike time must be within integration window"

            interval_boundaries.add(d.time)

    if 0 not in interval_boundaries:
        interval_boundaries.add(0)

    if t_final not in interval_boundaries:
        interval_boundaries.add(t_final)

    sorted_boundaries = list(interval_boundaries)

    sorted_boundaries.sort()

    while len(sorted_boundaries) < transcription_options.minimum_n_intervals:
        intervals = [
            (stop - start, (stop + start) / 2)
            for start, stop in zip(
                sorted_boundaries[:-1], sorted_boundaries[1:]
            )
        ]
        max_length = max(l for l, _ in intervals)

        if all(length == max_length for length, _ in intervals):
            sorted_boundaries += [mid for _, mid in intervals]
        else:
            sorted_boundaries += [
                point for length, point in intervals if point == max_length
            ]
        sorted_boundaries.sort()

    return [
        (start, stop)
        for start, stop in zip(sorted_boundaries[:-1], sorted_boundaries[1:])
    ]


class InterpolatingPoly:
    """Represents a multidimensional polynomial interpolation over a specified
    interval.

    This class facilitates the computation and evaluation of an interpolating
    polynomial given a set of values, the polynomial degree, and its interval.
    It provides functionality to obtain knot points, start and end points, and
    evaluate the polynomial at a given point in the interval, while maintaining
    information about discrete operators and transformations.

    Legendre-Gauss-Radau collocation points are used as the discritisation
    scheme.

    Attributes:
        interval (Tuple[float, float]): The interval [a, b] over which the
            polynomial is defined.
        dimension (int): Dimensionality of the interpolated values.
        degree (int): Degree of the interpolating polynomial.
        values (np.ndarray): The values to interpolate, corresponding to knot
            points of the polynomial.
        s (np.ndarray): Array of discrete points in polynomial parameter space.
        s_to_interval (Callable[[float], float]): Function to map points from
            parameter space to the original interval.
        integrals (np.ndarray): Integration operator for the interpolating
            polynomial.
        width (float): Half-width of the interval, used for transformations
            between parameter space and interval.
        bases (np.ndarray): Basis functions of the interpolating polynomial.
        derivatives (List[np.ndarray]): Derivative basis functions of the
            polynomial.
    """

    def __init__(self, dimension, interval, degree, values):
        """
        Initializes an instance of the class with given parameters and specific configurations
        required for discretization and calculations.

        Args:
            dimension: The dimension of the problem that defines the size of matrices and reshaping.
            interval: The interval [lower_bound, upper_bound] over which discretization is computed.
            degree: The degree of the polynomial basis used for computation.
            values: A numpy array representing the knot points, stacked in order.

        Raises:
            AssertionError: If the shape of the `values` input does not match the expected shape
                calculated from the interval, dimension, and degree parameters.
        """
        self.interval = interval
        self.dimension = dimension
        self.degree = degree
        self.values = values

        op_values = generate_discritisation_operators(interval, degree)

        self.s, self.s_to_interval, bases, derivatives, self.weights = (
            op_values
        )
        self.width = (self.interval[1] - self.interval[0]) / 2
        size = len(self.s) * dimension
        if len(values.shape) == 1:
            values = values.reshape(size, 1)
        assert values.shape == (
            size,
            1,
        ), f"Expected shape ({len(self.s) * dimension}, 1), got {values.shape}"

        self.bases = np.vstack(
            [np.reshape(np.array(base), (1, len(self.s))) for base in bases]
        )
        self.derivatives = [
            np.reshape(np.array(d[:-1]), (1, len(self.s) - 1))
            for d in derivatives
        ]

    def size(self) -> int:
        return len(self.s) * self.dimension

    def _interval_to_s(self, t):
        mean = (self.interval[1] + self.interval[0]) / 2
        return (t - mean) / self.width

    def knot_times(self) -> List[float]:
        # we skip the end point
        return [self.s_to_interval(s_i) for s_i in self.s]

    def start_point(self) -> Tuple[float, np.ndarray]:
        return self.s_to_interval(self.s[0]), self.values[: self.dimension]

    def end_point(self) -> Tuple[float, np.ndarray]:
        return self.s_to_interval(self.s[-1]), self.values[-self.dimension :]

    def as_raw(self) -> np.ndarray:
        t = np.reshape(np.array(self.knot_times()), (-1, 1))
        x = np.reshape(self.values, (-1, self.dimension))
        return np.hstack([t, x]).T

    def knot_points(self) -> Iterator[Tuple[float, np.ndarray, np.ndarray]]:
        """Get the knot points of the interpolating polynomial, not including the end point."""
        # we skip the end point
        t_i = self.knot_times()[:-1]
        n = len(self.s)
        x_i = [
            self.values[i * self.dimension : (i + 1) * self.dimension]
            for i in range(n - 1)
        ]
        dx_i = []
        ds_dt = 1 / self.width
        for s in self.s:

            ds = ds_dt * np.array(
                [i * s ** (i - 1) if i > 0 else 0 for i in range(len(self.s))]
            ).reshape((1, n))
            projection = (ds @ self.bases).T

            # this treats each column as the interpolation point.
            v = self.values.reshape((self.dimension, -1))
            value = v @ projection
            dx_i.append(value)

        for t, x, dx in zip(t_i, x_i, dx_i):
            yield t, x, dx

    def __call__(self, t) -> np.ndarray:
        """Evaluate the interpolating polynomial at time t."""

        assert (
            self.interval[0] <= t <= self.interval[1]
        ), f"Value {t} is not in interval {self.interval}"

        s = self._interval_to_s(t)
        try:
            i = next(i for i, s_i in enumerate(self.s) if abs(s_i - s) < 1e-9)
            return np.reshape(
                self.values[i * self.dimension : (i + 1) * self.dimension],
                (self.dimension,),
            )
        except StopIteration:
            pass
        n = len(self.s)
        s_vector = np.vstack([s**i for i in range(n)])

        projection = s_vector.T @ self.bases

        value = projection @ np.reshape(self.values, (-1, self.dimension))

        return np.reshape(value, (self.dimension,))

    def map(self, func):
        values = np.concatenate(
            [
                func(
                    self.s_to_interval(s),
                    self.values[i * self.dimension : (i + 1) * self.dimension],
                )
                for i, s in enumerate(self.s)
            ]
        )
        (size,) = values.shape
        dimension = size // len(self.s)

        return InterpolatingPoly(dimension, self.interval, self.degree, values)
