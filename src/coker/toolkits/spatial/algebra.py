from typing import Optional
import numpy as np
from coker.toolkits.spatial.types import Vec3, Scalar
from coker.toolkits.spatial.unit_quaternion import UnitQuaternion

SE3_BASIS = np.array(
    [
        [[0, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        [[0, 0, 1],
         [0, 0, 0],
         [-1, 0, 0]],
        [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 0]]
    ],
    dtype=float
)


def hat(u: Vec3):
    return np.dot(SE3_BASIS, u)


class Rotation3:
    def __init__(self, axis, angle):
        self.axis = axis
        self.angle = angle

    @staticmethod
    def zero():
        return Rotation3(np.array([0, 0, 1], dtype=float), angle=0)

    @staticmethod
    def from_vector(vector):
        array = np.array(vector, dtype=float)
        angle = np.linalg.norm(array)
        axis = array / angle
        return Rotation3(axis, angle)

    def inverse(self):
        return Rotation3(self.axis, -self.angle)

    def as_quaternion(self):
        return UnitQuaternion.from_axis_angle(self.axis, self.angle)

    @staticmethod
    def from_quaterion(q: UnitQuaternion):
        if q.q_0 == 1:
            return Rotation3.zero()

        theta = 2 * np.arccos(q.q_0)
        r = np.sqrt(1 - q.q_0 * q.q_0)  # sin(theta/2)
        u = q.v / r
        return Rotation3(axis=u, angle=theta)

    def __mul__(self, other: 'Rotation3'):
        q_1 = self.as_quaternion()
        q_2 = other.as_quaternion()

        q_21 = q_2 * q_1

        return Rotation3.from_quaterion(q_21)

    def __eq__(self, other):
        return isinstance(other, Rotation3) and (self.axis == other.axis).all() and self.angle == other.angle


class Isometry3:
    def __init__(self,
                 rotation: Optional[Rotation3] = None,
                 translation: Optional[Vec3] = None):

        self.rotation = rotation or Rotation3.zero()

        self.translation = np.array([0, 0, 0], dtype=float) if translation is None else translation

    @staticmethod
    def identity():
        return Isometry3(
            Rotation3.zero(),
            translation=np.array([0, 0, 0], dtype=float)
        )

    def __matmul__(self, other):
        if isinstance(other, Isometry3):
            rotation = self.rotation * other.rotation
            translation = self.rotation.as_quaternion().conjugate(
                other.translation
            ) + self.translation
            return Isometry3(rotation, translation)
        elif isinstance(other, np.ndarray) and other.shape == (3,):
            return self.rotation.as_quaternion().conjugate(other) + self.translation
        elif isinstance(other, np.ndarray) and other.shape == (4, 1):
            result = np.reshape(self.rotation.as_quaternion().conjugate(other[0:3, 0]) + self.translation, newshape=(3, 1))
            return np.concatenate([result, other[3:4, 0:1]], axis=0)

        raise NotImplementedError

    def transpose(self):
        q = self.rotation.inverse()
        return Isometry3(q, -q.as_quaternion().conjugate(self.translation))


class Screw:
    __slots__ = ('rotation', 'translation', 'magnitude')

    def __init__(self, rotation: Vec3, translation: Vec3, magnitude: float = 1):
        self.rotation = rotation
        self.translation = translation
        self.magnitude = magnitude

    @staticmethod
    def from_tuple(*values):
        rotation = np.array(values[0:3])
        translation = np.array(values[3:6])
        if all(v == 0 for v in values[0:3]):
            return Screw(rotation, translation, 1)

        mag = np.linalg.norm(rotation).astype(float).flatten()[0]

        return Screw(rotation/mag, translation / mag, mag)


    @staticmethod
    def zero():
        return Screw(
            rotation=np.array([1, 0, 0], dtype=float),
            translation=np.array([0, 0, 0], dtype=float),
            magnitude=0
        )

    def __mul__(self, other: Scalar):
        return Screw(self.rotation, self.translation, self.magnitude * other)

    def exp(self, angle=1) -> Isometry3:
        alpha = self.magnitude * angle
        w = hat(self.rotation)
        s = np.sin(alpha)
        c = np.cos(alpha)
        ww = w @ w

        r_add = w * s + (1 - c) * ww
        rotation = Rotation3(axis=self.rotation, angle=alpha)
        translation = alpha * self.translation * np.dot(self.rotation, self.translation) - r_add @ np.cross(self.rotation, self.translation)

        return Isometry3(rotation=rotation, translation=translation)


class SE3Adjoint:
    def __init__(self, transform: Isometry3):
        self.transform = transform

    def apply(self, zeta: Screw) -> Screw:

        q = self.transform.rotation.as_quaternion()
        p = self.transform.translation

        rotation = q.conjugate(zeta.rotation)
        translation = hat(p) @ rotation + q.conjugate(zeta.translation)

        return Screw(
            rotation=rotation,
            translation=translation,
            magnitude=zeta.magnitude
        )

    def transpose(self):
        return SE3CoAdjoint(self.transform)

    def __matmul__(self, other):
        if isinstance(other, Screw):
            return self.apply(other)
        if isinstance(other, SE3Adjoint):
            t = self.transform @ other.transform
            return SE3Adjoint(t)

        raise NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, SE3Adjoint):
            t = other.transform @ self.transform
            return SE3Adjoint(t)
        raise NotImplemented


class SE3CoAdjoint:
    def __init__(self, transform: Isometry3):
        self.transform = transform

    def apply(self, zeta: Screw) -> Screw:
        q_inv = self.transform.rotation.as_quaternion().inverse()
        p = self.transform.translation

        rotation = q_inv.conjugate(zeta.rotation)
        translation = -q_inv.conjugate(hat(p) @ zeta.rotation) + q_inv.conjugate(zeta.translation)

        return Screw(
            rotation=rotation,
            translation=translation,
            magnitude=zeta.magnitude
        )

    def __matmul__(self, other):
        if isinstance(other, Screw):
            return self.apply(other)
        raise NotImplemented

    def transpose(self):
        return SE3Adjoint(self.transform)


class se3Adjoint:
    def __init__(self, vector: Screw):
        self.vector = vector

    def apply(self, other: Screw) -> Screw:
        w = np.cross(self.vector.rotation, other.rotation)
        v = np.cross(self.vector.translation, other.rotation) \
            - np.cross(self.vector.rotation, other.translation)

        m = self.vector.magnitude * other.magnitude

        return Screw(
            translation=v,
            rotation=w,
            magnitude=m
        )

    def transpose(self) -> 'se3CoAdjoint':
        return se3CoAdjoint(self.vector)

    def __matmul__(self, other: Screw):
        assert isinstance(other, Screw)

        return self.apply(other)


class se3CoAdjoint:
    def __init__(self, vector: Screw):
        self.vector = vector

    def __matmul__(self, other: Screw):
        assert isinstance(other, Screw)

        return self.apply(other)

    def apply(self, other: Screw) -> Screw:
        w = np.cross(self.vector.rotation, other.rotation) \
            + np.cross(self.vector.translation, other.translation)

        v = np.cross(self.vector.rotation, other.translation)

        m = self.vector.magnitude * other.magnitude

        return Screw(
            translation=-v,
            rotation=-w,
            angle=m
        )

    def transpose(self) -> se3Adjoint:
        return se3Adjoint(self.vector)


def se3_bracket(left: Screw, right: Screw) -> Screw:
    w = np.cross(left.rotation, right.rotation)
    v = np.cross(left.rotation, right.translation) - np.cross(right.rotation, left.translation)

    return Screw(translation=v, rotation=w, angle=right.magnitude * left.magnitude)
