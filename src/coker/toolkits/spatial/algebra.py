from typing import Optional
import numpy as np
from coker.toolkits.spatial.types import Vec3, Scalar
from coker.toolkits.spatial.unit_quaternion import UnitQuaternion
from coker.algebra.graph import normalise

SE3_BASIS = np.array(
    [
        [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
    ],
)

e_x = np.array([1, 0, 0])
e_y = np.array([0, 1, 0])
e_z = np.array([0, 0, 1])


def hat(u: Vec3):
    """Return the skew matrix satisfying ``hat(u) @ v == cross(u, v)``.

    Args:
        u: Numeric or traceable vector of shape ``(3,)``.

    Returns:
        Array of shape ``(3, 3)``.
    """
    result = SE3_BASIS @ u
    return result


class Rotation3:
    """Represent an active three-dimensional rotation.

    Args:
        axis: Nonzero axis vector of shape ``(3,)``; quaternion conversion
            normalises it. A numeric zero axis denotes the identity.
        angle: Rotation angle in radians, numeric or traceable scalar.

    ``apply`` acts on column vectors. Historical multiplication order is
    ``(a * b).apply(v) == b.apply(a.apply(v))``, unlike matrix multiplication.
    Prefer ``Isometry3`` for conventional rightmost-first composition.

    Examples:
        >>> import numpy as np
        >>> from coker.toolkits.spatial import Rotation3
        >>> r = Rotation3(np.array([0., 0., 1.]), np.pi / 2)
        >>> np.testing.assert_allclose(r.apply(np.array([1., 0., 0.])),
        ...                            [0., 1., 0.], atol=1e-12)
    """

    def __init__(self, axis, angle):
        try:
            if np.allclose(axis, np.zeros(3)):
                axis = e_z
                angle = 0
        except NotImplementedError:
            pass
        self._axis = axis
        self._angle = angle
        self._quaternion = None

    @property
    def axis(self):
        """Return the materialised axis vector of shape ``(3,)``."""
        self._materialise_axis_angle()
        return self._axis

    @property
    def angle(self):
        """Return the materialised rotation angle in radians."""
        self._materialise_axis_angle()
        return self._angle

    def _materialise_axis_angle(self):
        if self._axis is not None:
            return

        q = self._quaternion
        if isinstance(q.q_0, (float, int, np.floating)) and q.q_0 == 1:
            self._axis = e_z.copy()
            self._angle = 0
            return

        axis, magnitude = normalise(q.v)
        self._axis = axis
        self._angle = 2 * np.arctan2(magnitude, q.q_0)

    @staticmethod
    def cast(other) -> "Rotation3":
        """Return a rotation from Rotation3, UnitQuaternion or a ``(3,)`` array.

        Args:
            other: Existing rotation value or numeric rotation vector.

        Returns:
            Rotation3, preserving an existing instance.

        Raises:
            NotImplementedError: If the representation is unsupported.
        """

        if isinstance(other, Rotation3):
            return other

        if isinstance(other, UnitQuaternion):
            return Rotation3.from_quaterion(other)
        if isinstance(other, np.ndarray) and other.shape == (3,):
            return Rotation3.from_vector(other)

        raise NotImplementedError(f"Cannot cast {other} to a rotation matrix")

    def as_vector(self) -> np.ndarray:
        """Return axis times angle as a rotation vector, shape ``(3,)``."""
        return self.axis * self.angle

    def __repr__(self):
        return f"Rotation3({repr(self.axis)}, {repr(self.angle)})"

    @staticmethod
    def zero():
        """Return the identity rotation, for example ``Rotation3.zero()``."""
        return Rotation3(
            np.array(
                e_z.copy(),
            ),
            angle=0,
        )

    @staticmethod
    def from_vector(vector):
        """Construct from a numeric rotation vector of shape ``(3,)``."""
        array = np.array(
            vector,
        )
        angle = np.linalg.norm(array)
        if angle == 0:
            return Rotation3.zero()
        axis = array / angle
        return Rotation3(axis, angle)

    def inverse(self):
        """Return the inverse rotation."""
        return Rotation3.from_quaterion(self.as_quaternion().inverse())

    def as_quaternion(self):
        """Return the cached unit quaternion representation."""
        if self._quaternion is None:
            self._quaternion = UnitQuaternion.from_axis_angle(self._axis, self._angle)
        return self._quaternion

    @staticmethod
    def from_quaterion(q: UnitQuaternion):
        """Construct from a unit quaternion without axis-angle conversion."""
        result = Rotation3.__new__(Rotation3)
        result._axis = None
        result._angle = None
        result._quaternion = q
        return result

    def __mul__(self, other: "Rotation3"):
        q_1 = self.as_quaternion()
        q_2 = other.as_quaternion()

        return Rotation3.from_quaterion(q_2 * q_1)

    def apply(self, other: np.ndarray):
        """Rotate a numeric or traceable vector of shape ``(3,)``."""
        return self.as_quaternion().conjugate(other)

    def __matmul__(self, other: np.ndarray):
        assert other.shape == (3,)
        return self.apply(other)

    def __eq__(self, other):
        return (
            isinstance(other, Rotation3)
            and (self.axis == other.axis).all()
            and self.angle == other.angle
        )

    def as_matrix(self):
        """Return the active rotation matrix of shape ``(3, 3)``."""
        q = self.as_quaternion()
        k = hat(q.v)
        return np.eye(3) + 2 * q.q_0 * k + 2 * (k @ k)


class Isometry3:
    """Map coordinates from a child frame into a parent frame.

    Args:
        rotation: Rotation3, UnitQuaternion or numeric rotation vector;
            defaults to identity.
        translation: Origin of the child in parent coordinates, shape
            ``(3,)``. Defaults to zero; use consistent length units.

    ``a @ b`` applies ``b`` first, then ``a``; ``a @ point`` rotates and
    translates a numeric point. For traceable points use ``apply``.

    Examples:
        >>> import numpy as np
        >>> from coker.toolkits.spatial import Isometry3
        >>> pose = Isometry3(translation=np.array([1., 0., 0.]))
        >>> np.testing.assert_allclose(pose.apply(np.zeros(3)), [1., 0., 0.])
    """

    def __init__(
        self,
        rotation: Optional[Rotation3] = None,
        translation: Optional[Vec3] = None,
    ):

        self.rotation = (
            Rotation3.zero() if rotation is None else (Rotation3.cast(rotation))
        )

        self.translation = (
            np.array(
                [0, 0, 0],
            )
            if translation is None
            else translation
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
        """Return an identity pose, for example ``Isometry3.identity()``."""
        return Isometry3(
            Rotation3.zero(),
            translation=np.array(
                [0, 0, 0],
            ),
        )

    def __matmul__(self, other):
        if isinstance(other, Isometry3):
            rotation = other.rotation * self.rotation
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
                shape=(3, 1),
            )
            return np.concatenate([result, other[3:4, 0:1]], axis=0)

        raise NotImplementedError(f"Isometry can't tranfrom {other.__class__}")

    def apply(self, other):
        """Map a numeric or traceable point of shape ``(3,)`` to the parent."""
        assert other.shape == (3,)
        return self.rotation.as_quaternion().conjugate(other) + self.translation

    def transpose(self):
        """Return the inverse rigid transform (not a raw matrix transpose)."""
        q = self.rotation.inverse()
        return Isometry3(q, -q.as_quaternion().conjugate(self.translation))

    def as_matrix(self):
        """Return the homogeneous matrix of shape ``(4, 4)``."""
        r = self.rotation.as_matrix()
        p = np.reshape(self.translation, shape=(3, 1))
        o = np.array(
            [[0, 0, 0, 1.0]],
        )

        result = np.concatenate((r, p), axis=1)

        return np.concatenate((result, o))

    def inverse(self):
        """Return the transform from the parent frame to the child frame."""
        r = self.rotation.inverse()
        p = r.as_matrix() @ self.translation
        return Isometry3(rotation=r, translation=-p)

    def as_vector(self) -> np.ndarray:
        """Return rotation-vector then translation as a ``(6, 1)`` array."""

        p = np.reshape(self.translation, shape=(3, 1))
        q = np.reshape(self.rotation.as_vector(), shape=(3, 1))
        return np.concatenate((q, p), axis=0)

    @staticmethod
    def from_vector(vector: np.ndarray):
        """Construct from a flat ``(6,)`` rotation-vector/translation array."""

        q = vector[0:3]
        p = vector[3:6]
        r = Rotation3.from_vector(q)
        return Isometry3(rotation=r, translation=p)


class Screw:
    """Store angular and linear parts of a spatial vector.

    Args:
        rotation: Angular part of shape ``(3,)``, default zero.
        translation: Linear part of shape ``(3,)``, default zero.
        magnitude: Scalar multiplier applied by ``to_array``. ``exp`` scales
            its angle by this value only when nonzero; zero uses the angle
            directly and does not guarantee an identity transform.

    Components are ordered angular then linear. Twists use radians and length
    per unit coordinate; wrenches use moment then force. No units are enforced.

    Examples:
        >>> import numpy as np
        >>> from coker.toolkits.spatial import Screw
        >>> axis = Screw.w_z()
        >>> np.testing.assert_allclose(axis.to_array(), [0, 0, 1, 0, 0, 0])
        >>> pose = axis.exp(np.pi / 2)
    """

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
        values = self.to_array().tolist()
        return repr(values)

    @staticmethod
    def w_z():
        """Return a unit angular z-axis screw."""
        return Screw(rotation=e_z.copy())

    @staticmethod
    def w_x():
        """Return a unit angular x-axis screw."""
        return Screw(rotation=e_x.copy())

    @staticmethod
    def w_y():
        """Return a unit angular y-axis screw."""
        return Screw(rotation=e_y.copy())

    @staticmethod
    def e_z():
        """Return a unit translational z-axis screw."""
        return Screw(translation=e_z.copy())

    @staticmethod
    def e_x():
        """Return a unit translational x-axis screw."""
        return Screw(translation=e_x.copy())

    @staticmethod
    def e_y():
        """Return a unit translational y-axis screw."""
        return Screw(translation=e_y.copy())

    def __neg__(self):
        return Screw(rotation=-self.rotation, translation=-self.translation)

    @staticmethod
    def from_tuple(*values):
        """Construct from six numeric angular-then-linear scalar values."""
        rotation = np.array(values[0:3])
        translation = np.array(values[3:6])
        if all(v == 0 for v in values[0:3]):
            return Screw(rotation, translation, 1)

        mag = np.linalg.norm(rotation).astype(float).flatten()[0]

        return Screw(rotation / mag, translation / mag, mag)

    def to_array(self) -> np.ndarray:
        """Return scaled angular-then-linear components, shape ``(6,)``."""
        return np.concatenate([self.rotation, self.translation]) * self.magnitude

    @staticmethod
    def from_array(array):
        """Construct from a numeric NumPy array with shape ``(6,)``.

        Coker vector ``Tracer`` inputs are unsupported by this constructor.
        """
        assert array.shape == (6,)
        rotation = array[0:3]
        translation = array[3:6]
        try:
            if (rotation == 0.0).all():
                return Screw(rotation, translation, 1)
            else:
                mag = np.linalg.norm(rotation)
        except TypeError:
            mag = 1

        return Screw(rotation / mag, translation / mag, mag)

    @staticmethod
    def zero():
        """Return a zero-magnitude screw, for example ``Screw.zero()``."""
        return Screw(
            rotation=np.array(
                [1, 0, 0],
            ),
            translation=np.array(
                [0, 0, 0],
            ),
            magnitude=0,
        )

    def __mul__(self, other: Scalar):
        return Screw(self.rotation, self.translation, self.magnitude * other)

    @property
    def axis(self) -> (Vec3, Vec3):
        """
        Returns:
            Tuple[Vec3, Vec3]: Point and direction of axis.

        """
        q = np.cross(self.rotation, self.translation)

        if (self.rotation == 0).all():
            return q, self.translation

        return q, self.rotation

    @property
    def pitch(self):
        """Return the angular-linear dot product times the stored magnitude."""
        return self.rotation.dot(self.translation) * self.magnitude

    def exp(self, angle=1) -> Isometry3:
        """Exponentiate a screw coordinate into a rigid transform.

        Args:
            angle: Numeric or traceable scalar coordinate; radians for a unit
                revolute screw, length for a unit translational screw.

        Returns:
            Isometry3: Transform generated with ``magnitude * angle`` when
            magnitude is nonzero, otherwise with ``angle`` directly. A zero
            magnitude does not guarantee an identity transform.
        """
        if self.magnitude != 0:
            alpha = self.magnitude * angle
        else:
            alpha = angle

        if isinstance(self.rotation, np.ndarray) and (self.rotation == 0).all():
            return Isometry3(
                rotation=Rotation3.zero(), translation=self.translation * alpha
            )

        rotation = Rotation3(axis=self.rotation, angle=alpha)

        rot_dot_t = np.dot(self.rotation, self.translation)
        rot_cross_t = np.cross(self.rotation, self.translation)

        t1 = -rotation.apply(rot_cross_t)
        t2 = alpha * self.rotation
        t3 = rot_dot_t * t2

        translation = rot_cross_t + t1 + t3

        return Isometry3(rotation=rotation, translation=translation)

    def as_matrix(self):
        """Attempt the legacy matrix conversion; currently raises ValueError.

        Use ``to_array()`` for six components or ``exp()`` for an Isometry3.
        """
        w_hat = hat(self.rotation)
        return np.concatenate(
            np.concatenate([w_hat, self.translation]),
            np.array([[0, 0, 0, 1]]),
            axis=1,
        )

    def __eq__(self, other):
        return (
            isinstance(other, Screw)
            and (self.rotation == other.rotation).all()
            and (self.translation == other.translation).all()
            and self.magnitude == other.magnitude
        )


class SE3Adjoint:
    """Transport twists with a finite rigid transform.

    Args:
        transform: Child-to-parent Isometry3.

    The ``(6, 6)`` matrix is ``[[R, 0], [hat(p) @ R, R]]``.
    This is a group adjoint, not the lowercase Lie-algebra operator.
    """

    def __init__(self, transform: Isometry3):
        self.transform = transform

    def apply(self, zeta: Screw) -> Screw:
        """Return a Screw transported into the transform's parent frame."""

        q = self.transform.rotation.as_quaternion()
        p = self.transform.translation

        rotation = q.conjugate(zeta.rotation)
        translation = hat(p) @ rotation + q.conjugate(zeta.translation)

        return Screw(
            rotation=rotation,
            translation=translation,
            magnitude=zeta.magnitude,
        )

    def transpose(self):
        """Return the transpose-dual SE3CoAdjoint for this transform."""
        return SE3CoAdjoint(self.transform)

    def __matmul__(self, other):
        if isinstance(other, Screw):
            return self.apply(other)
        if isinstance(other, SE3Adjoint):
            t = self.transform @ other.transform
            return SE3Adjoint(t)
        if isinstance(other, np.ndarray):
            return self.as_matrix() @ other

        raise NotImplementedError(f"Can't matmul by {other.__class__}")

    def __repr__(self):
        return f"Adj[{repr(self.transform)}]"

    def __rmatmul__(self, other):
        if isinstance(other, SE3Adjoint):
            t = other.transform @ self.transform
            return SE3Adjoint(t)
        raise NotImplementedError(f"Can't rmatmul by {other.__class__}")

    def __call__(self, arg: Screw) -> Screw:
        assert isinstance(arg, Screw)
        return self.apply(arg)

    def inverse(self):
        """Return the adjoint of the inverse transform."""
        return SE3Adjoint(self.transform.inverse())

    def as_matrix(self):
        """Return the angular-then-linear transport matrix, shape ``(6,6)``."""
        r = self.transform.rotation.as_matrix()
        p_hat = hat(self.transform.translation)
        row_1 = np.concatenate(
            [
                r,
                np.zeros(
                    (3, 3),
                ),
            ],
            axis=1,
        )
        row_2 = np.concatenate([p_hat @ r, r], axis=1)
        return np.concatenate([row_1, row_2], axis=0)


class SE3CoAdjoint:
    """Apply the transpose of a finite-transform adjoint to dual vectors.

    Args:
        transform: Isometry3 defining the paired twist transport.

    The matrix is ``[[R.T, -R.T @ hat(p)], [0, R.T]]``.
    It is the transpose, not the inverse-transpose, of SE3Adjoint's matrix.
    """

    def __init__(self, transform: Isometry3):
        self.transform = transform

    def as_matrix(self):
        """Return the dual transport matrix of shape ``(6, 6)``."""
        r = self.transform.rotation.as_matrix().T
        p_hat = hat(self.transform.translation)
        row_1 = np.concatenate([r, -r @ p_hat], axis=1)
        row_2 = np.concatenate(
            [
                np.zeros(
                    (3, 3),
                ),
                r,
            ],
            axis=1,
        )
        return np.concatenate([row_1, row_2], axis=0)

    def __repr__(self):
        return f"Adj^*[{repr(self.transform)}]"

    def apply(self, zeta: Screw) -> Screw:
        """Return a Screw transformed by the transpose-dual operator."""
        q_inv = self.transform.rotation.as_quaternion().inverse()
        p = self.transform.translation

        translation = q_inv.conjugate(zeta.translation)
        rotation = -q_inv.conjugate(hat(p) @ zeta.translation) + q_inv.conjugate(
            zeta.rotation
        )

        return Screw(
            rotation=rotation,
            translation=translation,
            magnitude=zeta.magnitude,
        )

    def __matmul__(self, other):
        if isinstance(other, Screw):
            return self.apply(other)
        raise NotImplementedError(f"Can't matmul by type {other.__class__}")

    def __call__(self, arg: Screw) -> Screw:
        assert isinstance(arg, Screw)
        return self.apply(arg)

    def transpose(self):
        """Return the paired finite-transform SE3Adjoint."""
        return SE3Adjoint(self.transform)


class se3Adjoint:
    """Apply the current infinitesimal screw operator.

    Args:
        vector: Screw generating the operator, not an Isometry3.

    The implementation's linear part is ``v x other.w - w x other.v``.
    Do not substitute this operator for a finite SE3Adjoint or assume it
    equals ``se3_bracket``.
    """

    def __init__(self, vector: Screw):
        self.vector = vector

    def apply(self, other: Screw) -> Screw:
        """Return the implemented angular/linear cross operation as a Screw."""
        w = np.cross(self.vector.rotation, other.rotation)
        v = np.cross(self.vector.translation, other.rotation) - np.cross(
            self.vector.rotation, other.translation
        )

        m = self.vector.magnitude * other.magnitude

        return Screw(translation=v, rotation=w, magnitude=m)

    def transpose(self) -> "se3CoAdjoint":
        """Return the paired lowercase se3CoAdjoint operator."""
        return se3CoAdjoint(self.vector)

    def __matmul__(self, other: Screw):
        assert isinstance(other, Screw)

        return self.apply(other)


class se3CoAdjoint:
    """Represent the paired infinitesimal dual operator.

    Args:
        vector: Screw generating the operator.

    The result uses angular-then-linear ordering and multiplies the two
    screws' stored magnitudes.
    """

    def __init__(self, vector: Screw):
        self.vector = vector

    def __matmul__(self, other: Screw):
        assert isinstance(other, Screw)

        return self.apply(other)

    def apply(self, other: Screw) -> Screw:
        """Return the dual cross operation as an angular-then-linear Screw."""
        w = np.cross(self.vector.rotation, other.rotation) + np.cross(
            self.vector.translation, other.translation
        )

        v = np.cross(self.vector.rotation, other.translation)

        m = self.vector.magnitude * other.magnitude

        return Screw(translation=-v, rotation=-w, magnitude=m)

    def transpose(self) -> se3Adjoint:
        """Return the paired lowercase se3Adjoint operator."""
        return se3Adjoint(self.vector)


def se3_bracket(left: Screw, right: Screw) -> Screw:
    """Compute the Lie bracket of two angular-then-linear screws.

    Args:
        left: Left angular-then-linear Screw.
        right: Right angular-then-linear Screw.

    Returns:
        Screw with the bracket components and the product of both magnitudes.
    """
    w = np.cross(left.rotation, right.rotation)
    v = np.cross(left.rotation, right.translation) - np.cross(
        right.rotation, left.translation
    )

    return Screw(translation=v, rotation=w, magnitude=right.magnitude * left.magnitude)
