import numpy as np

from coker import normalise
from coker.toolkits.spatial.types import Vec3, Scalar
from coker.algebra.kernel import Tracer


def quaternion_mul(q, p):

    qp_0 = q.q_0 * p.q_0 - np.dot(q.v, p.v)
    qp_v = q.q_0 * p.v + p.q_0 * q.v + np.cross(q.v, p.v)

    return UnitQuaternion(qp_0, qp_v)


class UnitQuaternion:
    __slots__ = ["q_0", "v"]

    def __str__(self):
        return f"UnitQuaternion({self.q_0}, {self.v})"

    def __init__(self, q_0: Scalar, v: Vec3):
        self.q_0 = q_0
        self.v = v

    def __eq__(self, other):
        return (
            isinstance(other, UnitQuaternion)
            and self.q_0 == other.q_0
            and (self.v == other.v).all()
        )

    @staticmethod
    def from_axis_angle(axis: Vec3, angle: Scalar):
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
        if isinstance(other, UnitQuaternion):
            return quaternion_mul(self, other)

        assert other.shape == (
            3,
        ), f"Cannot mul a vector of shape {other.shape}"
        p = UnitQuaternion(0, other)

        return quaternion_mul(self, p)

    def __rmul__(self, other):
        if isinstance(other, Vec3):
            assert other.shape == (3,)
            p = UnitQuaternion(0, other)
        elif isinstance(other, UnitQuaternion):
            p = other
        else:
            raise NotImplementedError(
                "Can't right multiply by {}".format(other)
            )

        return quaternion_mul(p, self)

    def inverse(self):
        return UnitQuaternion(self.q_0, -self.v)

    def conjugate(self, other):
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
        return UnitQuaternion(q_0, np.array([q_1, q_2, q_3]))

    def to_elements(self):
        return [self.q_0, self.v[0], self.v[1], self.v[2]]
