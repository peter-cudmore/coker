from typing import Optional
import numpy as np
from coker.toolkits.spatial.types import Vec3, Scalar
from coker.toolkits.spatial.unit_quaternion import UnitQuaternion
from coker.algebra.kernel import normalise

SE3_BASIS = np.array(
    [
        [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
    ],
    dtype=float,
)

e_x = np.array([1, 0, 0])
e_y = np.array([0, 1, 0])
e_z = np.array([0, 0, 1])


def hat(u: Vec3):

    result = SE3_BASIS @ u
    return result


#   return np.dot(SE3_BASIS, u)


class Rotation3:
    def __init__(self, axis, angle):
        self.axis = axis
        self.angle = angle

    def __repr__(self):
        return f"Rotation3({repr(self.axis)}, {repr(self.angle)})"

    @staticmethod
    def zero():
        return Rotation3(np.array(e_z.copy(), dtype=float), angle=0)

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
        try:
            if q.q_0 == 1:
                return Rotation3.zero()
        except NotImplementedError:
            pass
        u, r = normalise(q.v)
        theta = 2 * np.arctan2(r, q.q_0)
        #        theta = 2 * np.arccos(q.q_0)
        #        r = np.sqrt(1 - q.q_0 * q.q_0)  # sin(theta/2)
        #        u = q.v / r
        return Rotation3(axis=u, angle=theta)

    def __mul__(self, other: "Rotation3"):
        q_1 = self.as_quaternion()
        q_2 = other.as_quaternion()

        q_21 = q_2 * q_1

        return Rotation3.from_quaterion(q_21)

    def __eq__(self, other):
        return (
            isinstance(other, Rotation3)
            and (self.axis == other.axis).all()
            and self.angle == other.angle
        )

    def as_matrix(self):
        k = hat(self.axis)
        s = np.sin(self.angle)
        c = np.cos(self.angle)

        return np.eye(3) + s * k + (1 - c) * (k @ k)


class Isometry3:
    def __init__(
        self, rotation: Optional[Rotation3] = None, translation: Optional[Vec3] = None
    ):

        self.rotation = rotation or Rotation3.zero()

        self.translation = (
            np.array([0, 0, 0], dtype=float) if translation is None else translation
        )

    def __repr__(self):
        return f"Isometry3(r={repr(self.rotation)}, t={repr(self.translation)})"

    def __eq__(self, other):
        return (
            isinstance(other, Isometry3)
            and self.rotation == other.rotation
            and (self.translation == other.translation).all()
        )

    @staticmethod
    def identity():
        return Isometry3(Rotation3.zero(), translation=np.array([0, 0, 0], dtype=float))

    def __matmul__(self, other):
        if isinstance(other, Isometry3):
            rotation = self.rotation * other.rotation
            translation = (
                self.rotation.as_quaternion().conjugate(other.translation)
                + self.translation
            )
            return Isometry3(rotation, translation)
        elif isinstance(other, np.ndarray) and other.shape == (3,):
            return self.rotation.as_quaternion().conjugate(other) + self.translation
        elif isinstance(other, np.ndarray) and other.shape == (4, 1):
            result = np.reshape(
                self.rotation.as_quaternion().conjugate(other[0:3, 0])
                + self.translation,
                newshape=(3, 1),
            )
            return np.concatenate([result, other[3:4, 0:1]], axis=0)

        raise NotImplementedError()

    def apply(self, other):
        assert other.shape == (3,)
        return self.rotation.as_quaternion().conjugate(other) + self.translation

    def transpose(self):
        q = self.rotation.inverse()
        return Isometry3(q, -q.as_quaternion().conjugate(self.translation))

    def as_matrix(self):
        r = self.rotation.as_matrix()
        p = np.reshape(self.translation, newshape=(3, 1))
        o = np.array([[0, 0, 0, 1.0]], dtype=float)

        result = np.concatenate((r, p), axis=1)

        return np.concatenate((result, o))

    def inverse(self):
        r = self.rotation.inverse()
        p = r.as_matrix() @ self.translation
        return Isometry3(rotation=r, translation=-p)


class Screw:
    __slots__ = ("rotation", "translation", "magnitude")

    def __init__(
        self,
        rotation: Optional[Vec3] = None,
        translation: Optional[Vec3] = None,
        magnitude: float = 1,
    ):
        self.rotation = rotation if rotation is not None else np.array([0, 0, 0])
        self.translation = (
            translation if translation is not None else np.array([0, 0, 0])
        )
        self.magnitude = magnitude

    def __repr__(self):
        l = self.to_array().tolist()
        return repr(l)

    @staticmethod
    def w_z():
        return Screw(rotation=e_z.copy())

    @staticmethod
    def w_x():
        return Screw(rotation=e_x.copy())

    @staticmethod
    def w_y():
        return Screw(rotation=e_y.copy())

    @staticmethod
    def e_z():
        return Screw(translation=e_z.copy())

    @staticmethod
    def e_x():
        return Screw(translation=e_x.copy())

    @staticmethod
    def e_y():
        return Screw(translation=e_y.copy())

    @staticmethod
    def from_tuple(*values):
        rotation = np.array(values[0:3])
        translation = np.array(values[3:6])
        if all(v == 0 for v in values[0:3]):
            return Screw(rotation, translation, 1)

        mag = np.linalg.norm(rotation).astype(float).flatten()[0]

        return Screw(rotation / mag, translation / mag, mag)

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.rotation, self.translation]) * self.magnitude

    @staticmethod
    def from_array(array):
        assert array.shape == (6,)
        rotation = array[0:3]
        translation = array[3:6]
        if (rotation == 0.0).all():
            return Screw(rotation, translation, 1)

        mag = np.linalg.norm(rotation).astype(float).flatten()[0]
        return Screw(rotation / mag, translation / mag, mag)

    @staticmethod
    def zero():
        return Screw(
            rotation=np.array([1, 0, 0], dtype=float),
            translation=np.array([0, 0, 0], dtype=float),
            magnitude=0,
        )

    def __mul__(self, other: Scalar):
        return Screw(self.rotation, self.translation, self.magnitude * other)

    def exp(self, angle=1) -> Isometry3:
        if self.magnitude != 0:
            alpha = self.magnitude * angle
        else:
            alpha = angle
        if (self.rotation == 0).all():
            return Isometry3(
                rotation=Rotation3.zero(), translation=self.translation * alpha
            )

        w = hat(self.rotation)
        s = np.sin(alpha)
        c = np.cos(alpha)
        ww = w @ w

        r_add = w * s + (1 - c) * ww

        rot_dot_t = np.dot(self.rotation, self.translation)
        rot_cross_t = np.cross(self.rotation, self.translation)
        proj_t = rot_dot_t * self.translation
        translation = alpha * proj_t - r_add @ rot_cross_t
        rotation = Rotation3(axis=self.rotation, angle=alpha)

        return Isometry3(rotation=rotation, translation=translation)

    def __eq__(self, other):
        return (
            isinstance(other, Screw)
            and (self.rotation == other.rotation).all()
            and (self.translation == other.translation).all()
            and self.magnitude == other.magnitude
        )


class SE3Adjoint:
    def __init__(self, transform: Isometry3):
        self.transform = transform

    def apply(self, zeta: Screw) -> Screw:

        q = self.transform.rotation.as_quaternion()
        p = self.transform.translation

        rotation = q.conjugate(zeta.rotation)
        translation = hat(p) @ rotation + q.conjugate(zeta.translation)

        return Screw(
            rotation=rotation, translation=translation, magnitude=zeta.magnitude
        )

    def transpose(self):
        return SE3CoAdjoint(self.transform)

    def __matmul__(self, other):
        if isinstance(other, Screw):
            return self.apply(other)
        if isinstance(other, SE3Adjoint):
            t = self.transform @ other.transform
            return SE3Adjoint(t)
        if isinstance(other, np.ndarray):
            return self.as_matrix() @ other

        raise NotImplemented()

    def __repr__(self):
        return f"Adj[{repr(self.transform)}]"

    def __rmatmul__(self, other):
        if isinstance(other, SE3Adjoint):
            t = other.transform @ self.transform
            return SE3Adjoint(t)
        raise NotImplemented()

    def __call__(self, arg: Screw) -> Screw:
        assert isinstance(arg, Screw)
        return self.apply(arg)

    def inverse(self):
        return SE3Adjoint(self.transform.inverse())

    def as_matrix(self):
        r = self.transform.rotation.as_matrix()
        p_hat = hat(self.transform.translation)
        row_1 = np.concatenate([r, np.zeros((3, 3), dtype=float)], axis=1)
        row_2 = np.concatenate([p_hat @ r, r], axis=1)
        return np.concatenate([row_1, row_2], axis=0)


class SE3CoAdjoint:
    def __init__(self, transform: Isometry3):
        self.transform = transform

    def as_matrix(self):
        r = self.transform.rotation.as_matrix().T
        p_hat = hat(self.transform.translation)
        row_1 = np.concatenate([r, -r @ p_hat], axis=1)
        row_2 = np.concatenate([np.zeros((3, 3), dtype=float), r], axis=1)
        return np.concatenate([row_1, row_2], axis=0)

    def __repr__(self):
        return f"Adj^*[{repr(self.transform)}]"

    def apply(self, zeta: Screw) -> Screw:
        q_inv = self.transform.rotation.as_quaternion().inverse()
        p = self.transform.translation

        translation = q_inv.conjugate(zeta.translation)
        rotation = -q_inv.conjugate(hat(p) @ zeta.translation) + q_inv.conjugate(
            zeta.rotation
        )

        return Screw(
            rotation=rotation, translation=translation, magnitude=zeta.magnitude
        )

    def __matmul__(self, other):
        if isinstance(other, Screw):
            return self.apply(other)
        raise NotImplemented()

    def __call__(self, arg: Screw) -> Screw:
        assert isinstance(arg, Screw)
        return self.apply(arg)

    def transpose(self):
        return SE3Adjoint(self.transform)


class se3Adjoint:
    def __init__(self, vector: Screw):
        self.vector = vector

    def apply(self, other: Screw) -> Screw:
        w = np.cross(self.vector.rotation, other.rotation)
        v = np.cross(self.vector.translation, other.rotation) - np.cross(
            self.vector.rotation, other.translation
        )

        m = self.vector.magnitude * other.magnitude

        return Screw(translation=v, rotation=w, magnitude=m)

    def transpose(self) -> "se3CoAdjoint":
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
        w = np.cross(self.vector.rotation, other.rotation) + np.cross(
            self.vector.translation, other.translation
        )

        v = np.cross(self.vector.rotation, other.translation)

        m = self.vector.magnitude * other.magnitude

        return Screw(translation=-v, rotation=-w, angle=m)

    def transpose(self) -> se3Adjoint:
        return se3Adjoint(self.vector)


def se3_bracket(left: Screw, right: Screw) -> Screw:
    w = np.cross(left.rotation, right.rotation)
    v = np.cross(left.rotation, right.translation) - np.cross(
        right.rotation, left.translation
    )

    return Screw(translation=v, rotation=w, angle=right.magnitude * left.magnitude)
