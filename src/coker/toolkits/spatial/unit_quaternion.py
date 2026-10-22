"""Scalar-first Hamilton quaternions for active spatial rotations.

Native NumPy vectors have shape ``(3,)``. Compatible Coker tracers can pass
through selected arithmetic; this is not a generic array-backend interface.
"""

import numpy as np

from coker.algebra.graph import Tracer, normalise
from coker.toolkits.spatial.types import Scalar, Vec3


def quaternion_mul(q, p):
    """Return the Hamilton product ``q * p`` without renormalisation.

    Args:
        q: Left quaternion, scalar first.
        p: Right quaternion, scalar first.

    Returns:
        Quaternion with scalar ``q.q_0 * p.q_0 - dot(q.v, p.v)`` and vector
        ``q.q_0 * p.v + p.q_0 * q.v + cross(q.v, p.v)``. For unit inputs its
        active action applies ``p`` first, then ``q``.

    Examples:
        >>> import numpy as np
        >>> from coker.toolkits.spatial import (
        ...     UnitQuaternion, quaternion_mul)
        >>> q = UnitQuaternion.from_elements(1, 0, 0, 0)
        >>> np.testing.assert_array_equal(
        ...     quaternion_mul(q, q).to_elements(), [1, 0, 0, 0])
    """

    qp_0 = q.q_0 * p.q_0 - np.dot(q.v, p.v)
    qp_v = q.q_0 * p.v + p.q_0 * q.v + np.cross(q.v, p.v)

    return UnitQuaternion(qp_0, qp_v)


class UnitQuaternion:
    """Scalar-first quaternion with caller-maintained unit norm.

    The constructor neither checks nor normalises the norm. Conjugation is an
    active rotation only for unit quaternions. ``q * p`` is the Hamilton product
    and applies ``p`` then ``q``; this differs from ``Rotation3.__mul__``.
    Components are retained by reference and are mutable; changing a quaternion
    wrapped by a Rotation3 can invalidate that rotation's cached axis-angle.

    Native vectors are NumPy arrays, not lists. Selected Coker tracer arithmetic
    is supported by axis-angle construction and conjugation; no universal
    symbolic shape, zero-axis or normalisation support is promised.

    Attributes:
        q_0: Retained dimensionless scalar component.
        v: Retained shape ``(3,)`` dimensionless vector component.

    Examples:
        >>> import numpy as np
        >>> from coker.toolkits.spatial import UnitQuaternion
        >>> q = UnitQuaternion.from_axis_angle(np.array([0., 0., 1.]), np.pi / 2)
        >>> np.allclose(q.conjugate(np.array([1., 0., 0.])), [0., 1., 0.])
        True
    """

    __slots__ = ["q_0", "v"]

    def __str__(self):
        """Return scalar and vector components in readable form."""
        return f"UnitQuaternion({self.q_0}, {self.v})"

    def __init__(self, q_0: Scalar, v: Vec3):
        """Store components without copying or normalisation.

        Args:
            q_0: Dimensionless scalar component, numeric or compatible tracer.
            v: Dimensionless shape ``(3,)`` native vector or compatible tracer.
        """
        self.q_0 = q_0
        self.v = v

    def __eq__(self, other):
        """Compare components exactly; opposite signs are not considered equal."""
        return (
            isinstance(other, UnitQuaternion)
            and self.q_0 == other.q_0
            and (self.v == other.v).all()
        )

    @staticmethod
    def from_axis_angle(axis: Vec3, angle: Scalar):
        """Construct from an axis and angle, normalising the axis.

        Args:
            axis: Nonzero native shape ``(3,)`` vector or compatible tracer.
                Zero-axis handling is not generally supported; use identity
                elements ``(1, 0, 0, 0)`` instead.
            angle: Rotation angle in radians, numeric or compatible tracer.

        Returns:
            Unit quaternion ``(cos(angle/2), sin(angle/2) * normalise(axis))``.
        """
        try:
            if not axis.nonzero():
                return UnitQuaternion(1, np.array([0, 0, 0]))
        except AttributeError:
            pass
        q_0 = np.cos(angle / 2)
        s = np.sin(angle / 2)

        u, _ = normalise(axis)

        v = s * u
        return UnitQuaternion(q_0, v)

    def __mul__(self, other) -> "UnitQuaternion":
        """Hamilton-multiply a quaternion or a pure-vector quaternion.

        Args:
            other: Quaternion, or shape ``(3,)`` vector interpreted as ``(0, v)``.

        Returns:
            Quaternion product. A vector operand is not rotated by this method
            alone and the resulting quaternion need not have unit norm.

        Raises:
            ValueError: A vector operand does not have shape ``(3,)``.
        """
        if isinstance(other, UnitQuaternion):
            return quaternion_mul(self, other)

        if other.shape != (3,):
            raise ValueError(f"Cannot mul a vector of shape {other.shape}")
        p = UnitQuaternion(0, other)

        return quaternion_mul(self, p)

    def __rmul__(self, other):
        """Hamilton-multiply from the left by a quaternion or native 3-vector.

        Unsupported left operands raise ``NotImplementedError``. Vector
        operands represent pure quaternions, not a conjugation action. Vector
        shapes other than ``(3,)`` raise ``ValueError``.
        """
        if isinstance(other, Vec3):
            if other.shape != (3,):
                raise ValueError(f"Cannot mul a vector of shape {other.shape}")
            p = UnitQuaternion(0, other)
        elif isinstance(other, UnitQuaternion):
            p = other
        else:
            raise NotImplementedError("Can't right multiply by {}".format(other))

        return quaternion_mul(p, self)

    def inverse(self):
        """Return ``(q_0, -v)``, the inverse only when the norm is one."""
        return UnitQuaternion(self.q_0, -self.v)

    def conjugate(self, other):
        """Return ``self * other * self.inverse()``.

        Args:
            other: Quaternion, native shape ``(3,)`` vector, or compatible
                shape ``(3,)`` Coker tracer.

        Returns:
            Quaternion for quaternion input, otherwise the resulting vector.
            Traced results are reshaped to the input shape. Units are unchanged
            for a unit quaternion.

        Raises:
            NotImplementedError: The input type is unsupported.
        """
        if isinstance(other, UnitQuaternion):
            return self * other * self.inverse()
        elif isinstance(other, Vec3):
            qpq_inv = self * other * self.inverse()
            return qpq_inv.v
        elif isinstance(other, Tracer):
            assert other.shape[0] == 3
            qpq_inv = self * other * self.inverse()
            result = qpq_inv.v
            return np.reshape(result, shape=other.shape)

        raise NotImplementedError(
            f"Quaternion conjugation not implemented for {type(other)}"
        )

    @staticmethod
    def from_elements(q_0, q_1, q_2, q_3):
        """Construct from scalar-first components without normalisation.

        Args:
            q_0: Scalar quaternion component.
            q_1: x vector component.
            q_2: y vector component.
            q_3: z vector component.

        Returns:
            Quaternion with a newly assembled native vector array.
        """
        return UnitQuaternion(q_0, np.array([q_1, q_2, q_3]))

    def to_elements(self):
        """Return ``[q_0, q_1, q_2, q_3]`` as a new scalar-first list."""
        return [self.q_0, self.v[0], self.v[1], self.v[2]]
