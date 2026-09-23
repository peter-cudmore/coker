import math
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np


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
    # tol=1000 treats sub-1000-eps imaginary parts as round-off noise.
    roots = np.real_if_close(np.linalg.eigvals(companion_matrix), tol=1000)
    roots = np.sort(np.asarray(roots, dtype=float)).tolist()
    roots.append(1.0)

    return [-1] + roots[1:]


def evaluate_legendre_polynomial(x, n):
    return np.polynomial.legendre.legval(x, [0] * n + [1])


@dataclass(frozen=True)
class _ReferenceCollocationOperators:
    """Degree-specific operators on the reference interval [-1, 1]."""

    nodes: tuple[float, ...]
    basis_coefficients: tuple[tuple[float, ...], ...]
    derivative_matrix: tuple[tuple[float, ...], ...]
    quadrature_weights: tuple[float, ...]

    def scale_to_interval(self, interval: Tuple[float, float]) -> Tuple[
        List[float],
        Callable[[float], float],
        List[np.ndarray],
        List[np.ndarray],
        np.ndarray,
    ]:
        """Return fresh interval-scaled operators."""
        time_scaling_factor = (interval[1] - interval[0]) / 2

        def t(tau):
            return (interval[0] + interval[1]) / 2 + tau * time_scaling_factor

        derivative = [
            np.asarray(row, dtype=float) / time_scaling_factor
            for row in self.derivative_matrix
        ]
        weights = (
            np.asarray(self.quadrature_weights, dtype=float)
            * time_scaling_factor
        )
        return (
            np.asarray(self.nodes, dtype=float),
            t,
            [
                np.asarray(coefficients, dtype=float)
                for coefficients in self.basis_coefficients
            ],
            derivative,
            weights.reshape(1, len(self.nodes)),
        )


def _build_reference_operators(n: int) -> _ReferenceCollocationOperators:
    """Build immutable LGR operators that do not depend on an interval."""
    collocation_times = np.asarray(lgr_points(n), dtype=float)
    bases = np.empty((n + 1, n + 1))
    derivative_matrix = np.empty((n + 1, n + 1))

    for i, tau_i in enumerate(collocation_times):
        factors = [
            np.poly1d([1, -tau_j]) / (tau_i - tau_j)
            for tau_j in collocation_times
            if tau_i != tau_j
        ]
        basis_i = reduce(mul, factors)
        bases[:, i] = basis_i.c[::-1]
        dbasis_i = np.polyder(basis_i)

        derivative_matrix[:, i] = [
            dbasis_i(tau_j) for tau_j in collocation_times
        ]

    # See https://mathworld.wolfram.com/RadauQuadrature.html.
    quadrature_weights = np.array(
        [2 / n**2]
        + [
            (1 - x_i) / (n * evaluate_legendre_polynomial(x_i, n - 1)) ** 2
            for x_i in collocation_times[1:-1]
        ]
        + [0],
        dtype=float,
    )

    return _ReferenceCollocationOperators(
        nodes=tuple(collocation_times),
        basis_coefficients=tuple(tuple(row) for row in bases),
        derivative_matrix=tuple(tuple(row) for row in derivative_matrix),
        quadrature_weights=tuple(quadrature_weights),
    )


def generate_discritisation_operators(
    interval: Tuple[float, float], n: int
) -> Tuple[
    List[float],
    Callable[[float], float],
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
]:
    """Generate uncached discretisation operators for one interval."""
    return _build_reference_operators(n).scale_to_interval(interval)


def _predict_refined_degree(
    *, error: float, tolerance: float, degree: int
) -> int:
    """Predict the p-refined degree using the Patterson--Hager--Rao rule."""
    if not math.isfinite(error) or error <= 0:
        raise ValueError("error must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if degree <= 1:
        raise ValueError("degree must be greater than one")
    if error <= tolerance:
        return degree
    increment = math.ceil(
        (math.log(error) - math.log(tolerance)) / math.log(degree)
    )
    return degree + increment


def _split_refined_interval(
    *,
    interval: Tuple[float, float],
    predicted_degree: int,
    maximum_degree: int,
    minimum_degree: int,
    minimum_interval_duration: float,
) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[int, ...]]:
    """Select a p-refined interval or the required GPOPS h-refinement."""
    start, stop = interval
    if not math.isfinite(start) or not math.isfinite(stop) or start >= stop:
        raise ValueError("interval must have finite increasing bounds")
    if predicted_degree < 1:
        raise ValueError("predicted_degree must be positive")
    if maximum_degree < 1:
        raise ValueError("maximum_degree must be positive")
    if minimum_degree < 1:
        raise ValueError("minimum_degree must be positive")
    if (
        not math.isfinite(minimum_interval_duration)
        or minimum_interval_duration <= 0
    ):
        raise ValueError(
            "minimum_interval_duration must be finite and positive"
        )
    if predicted_degree <= maximum_degree:
        return (interval,), (predicted_degree,)

    count = max(2, math.ceil(predicted_degree / minimum_degree))
    duration = (stop - start) / count
    if duration < minimum_interval_duration:
        raise RuntimeError(
            "CasADi adaptive refinement cannot split interval "
            f"[{start}, {stop}] below minimum_interval_duration"
        )
    boundaries = [start + duration * index for index in range(count)]
    boundaries.append(stop)
    return (
        tuple(zip(boundaries[:-1], boundaries[1:])),
        (minimum_degree,) * count,
    )


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

    def __init__(
        self,
        dimension,
        interval,
        degree,
        values,
        reference_operators: Optional[_ReferenceCollocationOperators] = None,
    ):
        """
        Initializes an instance of the class with given parameters
        and specific configurations required for discretization and
        calculations.

        Args:
            dimension: The dimension of the problem that defines
                the size of matrices and reshaping.
            interval: The interval [lower_bound, upper_bound] over
                which discretization is computed.
            degree: The degree of the polynomial basis used for
                computation.
            values: A numpy array representing the knot points,
                stacked in order.

        Raises:
            AssertionError: If the shape of the `values` input does
                not match the expected shape calculated from the
                interval, dimension, and degree parameters.
        """
        self.interval = interval
        self.dimension = dimension
        self.degree = degree
        self.values = values
        self._reference_operators = (
            reference_operators
            if reference_operators is not None
            else _build_reference_operators(degree)
        )
        op_values = self._reference_operators.scale_to_interval(interval)
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

    def _map_to_reference_coordinate(self, t):
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
        """Get knot points of the interpolating polynomial, excluding
        the end point."""
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

        s = self._map_to_reference_coordinate(t)
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
        try:
            (size,) = values.shape
        except ValueError as ex:
            if len(values.shape) == 2 and values.shape[1] == 1:
                values = np.squeeze(values)
                (size,) = values.shape
            else:
                ex.add_note(
                    "Cannot reshape values of shape "
                    f"{values.shape}, expected a vector"
                )
                raise ex

        dimension = size // len(self.s)

        return InterpolatingPoly(
            dimension,
            self.interval,
            self.degree,
            values,
            reference_operators=self._reference_operators,
        )
