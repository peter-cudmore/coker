"""Three-dimensional rotations, rigid transforms and angular-first screws.

Use native NumPy arrays of shape ``(3,)`` for spatial vectors. Angles are
in radians; translations and linear screw components use the caller's chosen
length unit. Selected arithmetic also accepts Coker tracers, but native-array
dispatch, comparisons, normalisation and materialisation are not general
symbolic operations. See the spatial API reference for executable examples.

Arrays are generally retained, not defensively copied. Treat inputs and
returned components as immutable: rotations cache their quaternion or
axis-angle representation and do not invalidate it after mutation.
"""

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
"""Shape ``(3, 3, 3)`` basis used by :func:`hat`; do not mutate."""

e_x = np.array([1, 0, 0])
"""Native unit vector along the x axis; shared mutable array, do not mutate."""
e_y = np.array([0, 1, 0])
"""Native unit vector along the y axis; shared mutable array, do not mutate."""
e_z = np.array([0, 0, 1])
"""Native unit vector along the z axis; shared mutable array, do not mutate."""


def hat(u: Vec3):
    """Return the skew matrix satisfying ``hat(u) @ v == cross(u, v)``.

    Args:
        u: Native or traced vector of shape ``(3,)``.

    Returns:
        Shape ``(3, 3)`` skew matrix, with the same units as ``u``.

    Examples:
        >>> import numpy as np
        >>> from coker.toolkits.spatial import hat, e_x, e_y, e_z
        >>> np.array_equal(hat(e_x) @ e_y, e_z)
        True
    """
    result = SE3_BASIS @ u
    return result


class Rotation3:
    """Axis-angle rotation acting actively on three-dimensional vectors.

    ``a * b`` applies ``a`` first, then ``b``; its matrix is ``R_b @ R_a``.
    This order differs from :class:`Isometry3` composition. A zero native axis
    selects identity. Nonzero axes are normalised when making the quaternion,
    while :meth:`as_vector` uses the stored axis without normalising it.
    Use a unit axis for a consistent axis-angle/vector representation.

    Native arrays and selected Coker traced arithmetic are supported; casting
    and rotation-vector construction are native-array operations. Inputs and
    cached representations are shared and must not be mutated.

    Examples:
        >>> import numpy as np
        >>> from coker.toolkits.spatial import Rotation3, e_x, e_y, e_z
        >>> np.allclose(Rotation3(e_z, np.pi / 2).apply(e_x), e_y)
        True
    """

    def __init__(self, axis, angle):
        """Initialise a rotation.

        Args:
            axis: Shape ``(3,)`` axis, preferably unit length; retained by reference.
            angle: Rotation angle in radians, numeric or a compatible tracer.
        """
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
        """Return the stored or lazily materialised shape ``(3,)`` axis."""
        self._materialise_axis_angle()
        return self._axis

    @property
    def angle(self):
        """Return the stored or lazily materialised angle in radians."""
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
        """Convert a supported native representation to a rotation.

        Args:
            other: Rotation (returned unchanged), unit quaternion (retained),
                or native NumPy rotation vector with shape ``(3,)``.

        Returns:
            Rotation represented by ``other``.

        Raises:
            NotImplementedError: The representation is unsupported.
        """

        if isinstance(other, Rotation3):
            return other

        if isinstance(other, UnitQuaternion):
            return Rotation3.from_quaterion(other)
        if isinstance(other, np.ndarray) and other.shape == (3,):
            return Rotation3.from_vector(other)

        raise NotImplementedError(f"Cannot cast {other} to a rotation matrix")

    def as_vector(self) -> np.ndarray:
        """Return ``axis * angle`` as a shape ``(3,)`` vector in radians."""
        return self.axis * self.angle

    def __repr__(self):
        """Return a representation using the materialised axis and angle."""
        return f"Rotation3({repr(self.axis)}, {repr(self.angle)})"

    @staticmethod
    def zero():
        """Return identity with a fresh z axis and zero angle."""
        return Rotation3(
            np.array(
                e_z.copy(),
            ),
            angle=0,
        )

    @staticmethod
    def from_vector(vector):
        """Construct from a native rotation vector in radians.

        Args:
            vector: Three components; its norm is the angle and its direction
                the axis. A zero vector gives identity. Not a tracing entry point.

        Returns:
            Rotation with normalised axis.
        """
        array = np.array(
            vector,
        )
        angle = np.linalg.norm(array)
        if angle == 0:
            return Rotation3.zero()
        axis = array / angle
        return Rotation3(axis, angle)

    def inverse(self):
        """Return the inverse rotation through quaternion conjugation."""
        return Rotation3.from_quaterion(self.as_quaternion().inverse())

    def as_quaternion(self):
        """Return the shared, lazily cached unit quaternion; do not mutate it."""
        if self._quaternion is None:
            self._quaternion = UnitQuaternion.from_axis_angle(self._axis, self._angle)
        return self._quaternion

    @staticmethod
    def from_quaterion(q: UnitQuaternion):
        """Wrap a unit quaternion without copying or normalising it.

        Args:
            q: Unit quaternion; its components may contain compatible tracers.

        Returns:
            Rotation whose axis and angle are materialised only on demand.

        Note:
            The spelling ``from_quaterion`` is the existing public API.
        """
        result = Rotation3.__new__(Rotation3)
        result._axis = None
        result._angle = None
        result._quaternion = q
        return result

    def __mul__(self, other: "Rotation3"):
        """Compose rotations, applying ``self`` first and ``other`` second."""
        q_1 = self.as_quaternion()
        q_2 = other.as_quaternion()

        return Rotation3.from_quaterion(q_2 * q_1)

    def apply(self, other: np.ndarray):
        """Act on a shape ``(3,)`` native or compatible traced vector.

        Args:
            other: Vector to rotate, in any consistent Cartesian units.

        Returns:
            Rotated vector, with the input's units.
        """
        return self.as_quaternion().conjugate(other)

    def __matmul__(self, other: np.ndarray):
        """Apply to a shape ``(3,)`` vector, equivalently ``self.apply(other)``.

        Raises:
            ValueError: The vector does not have shape ``(3,)``.
        """
        if other.shape != (3,):
            raise ValueError(f"Expected a vector of shape (3,), got {other.shape}")
        return self.apply(other)

    def __eq__(self, other):
        """Compare stored/materialised axis and angle exactly, not modulo turns."""
        return (
            isinstance(other, Rotation3)
            and (self.axis == other.axis).all()
            and self.angle == other.angle
        )

    def as_matrix(self):
        """Return the shape ``(3, 3)`` active rotation matrix."""
        q = self.as_quaternion()
        k = hat(q.v)
        return np.eye(3) + 2 * q.q_0 * k + 2 * (k @ k)


class Isometry3:
    """Rigid transform with action ``R @ point + translation``.

    ``a @ b`` applies ``b`` then ``a``; its matrix is ``T_a @ T_b``.
    Translations use a caller-chosen length unit. Native NumPy vectors are
    supported by ``@``; :meth:`apply` also supports compatible Coker tracers.
    Components are retained without copying; do not mutate them after use,
    particularly after the rotation has cached a representation.

    Attributes:
        rotation: Retained :class:`Rotation3` component.
        translation: Retained shape ``(3,)`` displacement in length units.

    Examples:
        >>> import numpy as np
        >>> from coker.toolkits.spatial import Isometry3, e_x, e_y
        >>> transform = Isometry3(translation=e_x)
        >>> np.array_equal(transform.apply(e_y), e_x + e_y)
        True
    """

    def __init__(
        self,
        rotation: Optional[Rotation3] = None,
        translation: Optional[Vec3] = None,
    ):
        """Initialise a transform, defaulting to identity.

        Args:
            rotation: Rotation, unit quaternion or native rotation vector
                accepted by :meth:`Rotation3.cast`; ``None`` gives identity.
            translation: Shape ``(3,)`` native or traced displacement;
                ``None`` gives a fresh zero array.
        """

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
        """Return a representation of the rotation and translation."""
        return f"Isometry3(r={repr(self.rotation)}, t={repr(self.translation)})"

    def __eq__(self, other):
        """Compare rotation and translation components exactly."""
        return (
            isinstance(other, Isometry3)
            and self.rotation == other.rotation
            and (self.translation == other.translation).all()
        )

    @staticmethod
    def identity():
        """Return an identity transform with fresh native components."""
        return Isometry3(
            Rotation3.zero(),
            translation=np.array(
                [0, 0, 0],
            ),
        )

    def __matmul__(self, other):
        """Compose transforms or transform a native point.

        Args:
            other: Another isometry, native point of shape ``(3,)``, or native
                column of shape ``(4, 1)``.

        Returns:
            Composed transform or transformed point. For a ``(4, 1)`` column,
            translation is always added and the last entry is preserved:
            only columns with last entry one have standard homogeneous-point
            semantics; this is not a homogeneous direction-vector action.

        Raises:
            NotImplementedError: The argument type or shape is unsupported.
        """
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
        """Return ``R @ other + translation`` for a shape ``(3,)`` point.

        Raises:
            ValueError: The point does not have shape ``(3,)``.
        """
        if other.shape != (3,):
            raise ValueError(f"Expected a point of shape (3,), got {other.shape}")
        return self.rotation.as_quaternion().conjugate(other) + self.translation

    def transpose(self):
        """Return the inverse rigid transform, not the literal matrix transpose."""
        q = self.rotation.inverse()
        return Isometry3(q, -q.as_quaternion().conjugate(self.translation))

    def as_matrix(self):
        """Return the shape ``(4, 4)`` matrix with blocks ``[[R, p], [0, 1]]``."""
        r = self.rotation.as_matrix()
        p = np.reshape(self.translation, shape=(3, 1))
        o = np.array(
            [[0, 0, 0, 1.0]],
        )

        result = np.concatenate((r, p), axis=1)

        return np.concatenate((result, o))

    def inverse(self):
        """Return the inverse transform with translation ``-R.T @ p``."""
        r = self.rotation.inverse()
        p = r.as_matrix() @ self.translation
        return Isometry3(rotation=r, translation=-p)

    def as_vector(self) -> np.ndarray:
        """Return a ``(6, 1)`` column of rotation vector then translation.

        This is a packed representation, not the SE(3) logarithm; the first
        three entries are radians and the last three are length units.
        """

        p = np.reshape(self.translation, shape=(3, 1))
        q = np.reshape(self.rotation.as_vector(), shape=(3, 1))
        return np.concatenate((q, p), axis=0)

    @staticmethod
    def from_vector(vector: np.ndarray):
        """Construct from a flat six-vector, angular entries before translation.

        Args:
            vector: Native shape ``(6,)`` array: rotation vector in radians,
                followed by translation in length units.

        Returns:
            Transform. Pass ``as_vector().reshape(6)`` for a round trip because
            :meth:`as_vector` returns a column rather than a flat vector.
        """

        q = vector[0:3]
        p = vector[3:6]
        r = Rotation3.from_vector(q)
        return Isometry3(rotation=r, translation=p)


class Screw:
    """Angular-first screw stored as two vectors and a scalar magnitude.

    ``to_array()`` is ``magnitude * [rotation, translation]``. For twists the
    first block is angular and the second linear; for wrenches the same storage
    means moment then force. Units are determined by that interpretation and
    the scalar parameter (for example radians/s and length/s for a twist).
    Constructors do not enforce dimensional consistency or unit angular axes.
    ``from_array`` and ``from_tuple`` normalise nonzero native angular blocks.

    Component arrays are retained; scalar multiplication shares them. Do not
    mutate inputs or returned components. Native arrays are the general path;
    selected arithmetic traces, but native comparisons and normalisation mean
    this is not an unrestricted symbolic screw type.

    Attributes:
        rotation: Retained shape ``(3,)`` angular or moment component.
        translation: Retained shape ``(3,)`` linear or force component.
        magnitude: Scalar multiplying both represented component blocks.

    Examples:
        >>> from coker.toolkits.spatial import Screw
        >>> Screw.e_x().exp(2).translation.tolist()
        [2, 0, 0]
    """

    __slots__ = ("rotation", "translation", "magnitude")

    def __init__(
        self,
        rotation: Optional[Vec3] = None,
        translation: Optional[Vec3] = None,
        magnitude: float = 1,
    ):
        """Initialise unnormalised screw components.

        Args:
            rotation: Shape ``(3,)`` angular/moment component; defaults to zero.
            translation: Shape ``(3,)`` linear/force component; defaults to zero.
            magnitude: Shared scalar multiplier, default one.
        """
        self.rotation = rotation if rotation is not None else np.array([0, 0, 0])
        self.translation = (
            translation if translation is not None else np.array([0, 0, 0])
        )
        self.magnitude = magnitude

    def __repr__(self):
        """Return the represented angular-first six-vector as a list string."""
        values = self.to_array().tolist()
        return repr(values)

    @staticmethod
    def w_z():
        """Return a fresh unit angular screw along z."""
        return Screw(rotation=e_z.copy())

    @staticmethod
    def w_x():
        """Return a fresh unit angular screw along x."""
        return Screw(rotation=e_x.copy())

    @staticmethod
    def w_y():
        """Return a fresh unit angular screw along y."""
        return Screw(rotation=e_y.copy())

    @staticmethod
    def e_z():
        """Return a fresh unit translational screw along z."""
        return Screw(translation=e_z.copy())

    @staticmethod
    def e_x():
        """Return a fresh unit translational screw along x."""
        return Screw(translation=e_x.copy())

    @staticmethod
    def e_y():
        """Return a fresh unit translational screw along y."""
        return Screw(translation=e_y.copy())

    def __neg__(self):
        """Negate component vectors, resetting magnitude to one.

        This legacy operation does not preserve a non-unit magnitude; use
        ``screw * -1`` to negate the represented six-vector in that case.
        """
        return Screw(rotation=-self.rotation, translation=-self.translation)

    @staticmethod
    def from_tuple(*values):
        """Construct from six native scalars, angular components first.

        Args:
            *values: Three angular components followed by three linear ones.

        Returns:
            Screw with unit angular direction and angular norm as magnitude,
            or magnitude one when the angular block is zero.
        """
        rotation = np.array(values[0:3])
        translation = np.array(values[3:6])
        if all(v == 0 for v in values[0:3]):
            return Screw(rotation, translation, 1)

        mag = np.linalg.norm(rotation).astype(float).flatten()[0]

        return Screw(rotation / mag, translation / mag, mag)

    def to_array(self) -> np.ndarray:
        """Return a fresh flat shape ``(6,)`` array, angular block first."""
        return np.concatenate([self.rotation, self.translation]) * self.magnitude

    @staticmethod
    def from_array(array):
        """Construct from a shape ``(6,)`` angular-first native array.

        Args:
            array: Angular then linear components, including their scale.

        Returns:
            Screw representing the input. Native nonzero angular blocks are
            normalised and their norm becomes the magnitude; pure translation
            retains magnitude one and may share slices with the input.

        Raises:
            ValueError: The array does not have shape ``(6,)``.
        """
        if array.shape != (6,):
            raise ValueError(f"Expected an array of shape (6,), got {array.shape}")
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
        """Return a zero represented vector with x angular axis and magnitude zero.

        The current :meth:`exp` zero-magnitude branch uses its argument directly;
        do not infer an identity exponential from this storage representation.
        """
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
        """Scale magnitude by a scalar, sharing both component arrays."""
        return Screw(self.rotation, self.translation, self.magnitude * other)

    @property
    def axis(self) -> (Vec3, Vec3):
        """Return an axis point and direction from the stored components.

        Returns:
            Tuple of ``cross(rotation, translation)`` and ``rotation``;
            for zero rotation the direction is ``translation`` instead.
            The point has geometric axis-point meaning for a unit angular
            direction; no normalisation or magnitude correction is performed.
        """
        q = np.cross(self.rotation, self.translation)

        if (self.rotation == 0).all():
            return q, self.translation

        return q, self.rotation

    @property
    def pitch(self):
        """Return ``dot(rotation, translation) * magnitude`` without normalisation."""
        return self.rotation.dot(self.translation) * self.magnitude

    def exp(self, angle=1) -> Isometry3:
        """Return the implemented screw motion as an isometry.

        Args:
            angle: Scalar motion parameter, default one. With unit angular
                direction, ``magnitude * angle`` is radians; for a pure linear
                screw it scales the stored displacement instead.

        Returns:
            Transform with ``alpha = magnitude * angle``, except that zero
            magnitude uses ``alpha = angle``. Pure native translation yields
            ``p = translation * alpha``. Otherwise, with stored ``w, v``,
            ``R = Rotation3(w, alpha)`` and
            ``p = cross(w, v) - R.apply(cross(w, v)) + alpha * dot(w, v) * w``.
            The usual screw exponential interpretation requires unit ``w``.
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
        """Attempt the legacy matrix conversion (currently unsupported).

        The native-array implementation passes incompatible dimensions and
        arguments to ``numpy.concatenate`` and raises instead of returning
        a matrix. Use :meth:`to_array` for supported screw serialisation or
        :meth:`exp` followed by ``as_matrix()`` for a transform matrix.
        """
        w_hat = hat(self.rotation)
        return np.concatenate(
            np.concatenate([w_hat, self.translation]),
            np.array([[0, 0, 0, 1]]),
            axis=1,
        )

    def __eq__(self, other):
        """Compare stored components and magnitude exactly."""
        return (
            isinstance(other, Screw)
            and (self.rotation == other.rotation).all()
            and (self.translation == other.translation).all()
            and self.magnitude == other.magnitude
        )


class SE3Adjoint:
    """SE(3) transform action on angular-first twists.

    For transform ``(R, p)``, the matrix is
    ``A = [[R, 0], [hat(p) @ R, R]]``. This is a group action, not the
    Lie-algebra operator :class:`se3Adjoint`. The transform is shared, not
    copied. Numeric arrays are supported by matrix multiplication; applying
    a screw uses compatible component arithmetic, including selected tracers.

    Examples:
        >>> from coker.toolkits.spatial import SE3Adjoint, Isometry3, Screw
        >>> SE3Adjoint(Isometry3.identity())(Screw.w_z()) == Screw.w_z()
        True
    """

    def __init__(self, transform: Isometry3):
        """Initialise with a retained :class:`Isometry3` transform."""
        self.transform = transform

    def apply(self, zeta: Screw) -> Screw:
        """Return a transformed screw, retaining its magnitude.

        Args:
            zeta: Angular-first twist screw.

        Returns:
            Screw with components ``R @ w`` and ``cross(p, R @ w) + R @ v``.
        """

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
        """Return :class:`SE3CoAdjoint`, whose matrix is this matrix's transpose."""
        return SE3CoAdjoint(self.transform)

    def __matmul__(self, other):
        """Apply to a screw/NumPy array or compose with another SE3 adjoint.

        Composition uses ``self.transform @ other.transform``. Array shapes
        follow NumPy matrix multiplication against the ``(6, 6)`` matrix.
        Unsupported operands raise ``NotImplementedError``.
        """
        if isinstance(other, Screw):
            return self.apply(other)
        if isinstance(other, SE3Adjoint):
            t = self.transform @ other.transform
            return SE3Adjoint(t)
        if isinstance(other, np.ndarray):
            return self.as_matrix() @ other

        raise NotImplementedError(f"Can't matmul by {other.__class__}")

    def __repr__(self):
        """Return a representation containing the transform."""
        return f"Adj[{repr(self.transform)}]"

    def __rmatmul__(self, other):
        """Compose a left SE3 adjoint; reject other types with NotImplementedError."""
        if isinstance(other, SE3Adjoint):
            t = other.transform @ self.transform
            return SE3Adjoint(t)
        raise NotImplementedError(f"Can't rmatmul by {other.__class__}")

    def __call__(self, arg: Screw) -> Screw:
        """Apply to a :class:`Screw`, equivalently :meth:`apply`.

        Raises:
            TypeError: The argument is not a :class:`Screw`.
        """
        if not isinstance(arg, Screw):
            raise TypeError("Expected a Screw argument")
        return self.apply(arg)

    def inverse(self):
        """Return the group adjoint of the inverse transform."""
        return SE3Adjoint(self.transform.inverse())

    def as_matrix(self):
        """Return the shape ``(6, 6)`` matrix ``[[R, 0], [hat(p) @ R, R]]``."""
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
    """Transpose SE(3) action on angular-first wrenches.

    For transform ``(R, p)``, this implements ``A.T``, where
    ``A = SE3Adjoint(transform).as_matrix()``: moment transforms as
    ``R.T @ (moment - cross(p, force))``, force as ``R.T @ force``.
    Thus ``dot(A @ twist, wrench) == dot(twist, A.T @ wrench)``.
    It is the transpose, not the inverse-transpose action for the same
    transform. The transform is retained by reference.

    Examples:
        >>> from coker.toolkits.spatial import SE3CoAdjoint, Isometry3, Screw
        >>> SE3CoAdjoint(Isometry3.identity())(Screw.e_x()) == Screw.e_x()
        True
    """

    def __init__(self, transform: Isometry3):
        """Initialise with a retained :class:`Isometry3` transform."""
        self.transform = transform

    def as_matrix(self):
        """Return the shape ``(6, 6)`` transpose of the transform adjoint matrix."""
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
        """Return a representation containing the transform."""
        return f"Adj^*[{repr(self.transform)}]"

    def apply(self, zeta: Screw) -> Screw:
        """Apply the transpose group action to a moment-first wrench screw.

        Args:
            zeta: Screw interpreted as moment then force, including magnitude.

        Returns:
            Transformed screw with unchanged magnitude.
        """
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
        """Apply to a Screw; unsupported types raise NotImplementedError."""
        if isinstance(other, Screw):
            return self.apply(other)
        raise NotImplementedError(f"Can't matmul by type {other.__class__}")

    def __call__(self, arg: Screw) -> Screw:
        """Apply to a :class:`Screw`, equivalently :meth:`apply`.

        Raises:
            TypeError: The argument is not a :class:`Screw`.
        """
        if not isinstance(arg, Screw):
            raise TypeError("Expected a Screw argument")
        return self.apply(arg)

    def transpose(self):
        """Return the corresponding :class:`SE3Adjoint` on the same transform."""
        return SE3Adjoint(self.transform)


class se3Adjoint:
    """Legacy Lie-algebra operator with explicit, nonstandard lower-block sign.

    For stored ``vector = (w, v)`` and operand ``(u, t)``, components are
    ``(cross(w, u), cross(v, u) - cross(w, t))``; magnitudes multiply.
    This is not :func:`se3_bracket`, whose last cross-product has the opposite
    sign. It is not an SE(3) transform action. The screw is retained by reference;
    numerical arrays and compatible traced cross-products use the same formula.

    Examples:
        >>> from coker.toolkits.spatial import se3Adjoint, Screw
        >>> (se3Adjoint(Screw.w_x()) @ Screw.e_y()).to_array().tolist()
        [0, 0, 0, 0, 0, -1]
    """

    def __init__(self, vector: Screw):
        """Initialise with a retained screw supplying the operator components."""
        self.vector = vector

    def apply(self, other: Screw) -> Screw:
        """Apply the class's documented formula to another screw."""
        w = np.cross(self.vector.rotation, other.rotation)
        v = np.cross(self.vector.translation, other.rotation) - np.cross(
            self.vector.rotation, other.translation
        )

        m = self.vector.magnitude * other.magnitude

        return Screw(translation=v, rotation=w, magnitude=m)

    def transpose(self) -> "se3CoAdjoint":
        """Return the paired legacy coadjoint, not a literal matrix transpose.

        In particular, its lower-right block is ``-hat(w)`` rather than the
        ``+hat(w)`` obtained by transposing this operator's lower-right block.
        """
        return se3CoAdjoint(self.vector)

    def __matmul__(self, other: Screw):
        """Apply to a :class:`Screw`, equivalently :meth:`apply`.

        Raises:
            TypeError: The operand is not a :class:`Screw`.
        """
        if not isinstance(other, Screw):
            raise TypeError("Expected a Screw operand")

        return self.apply(other)


class se3CoAdjoint:
    """Legacy Lie-algebra coadjoint with its actual implemented signs.

    For stored ``(w, v)`` and operand ``(u, t)``, the result components are
    ``(-cross(w, u) - cross(v, t), -cross(w, t))`` and magnitudes multiply.
    This is not the literal transpose of :class:`se3Adjoint` and must not be
    used to assume a twist/wrench duality with that class. It is distinct from
    the transform-based :class:`SE3CoAdjoint`. The screw is retained, not copied.

    Examples:
        >>> from coker.toolkits.spatial import se3CoAdjoint, Screw
        >>> (se3CoAdjoint(Screw.w_x()) @ Screw.e_y()).to_array().tolist()
        [0, 0, 0, 0, 0, -1]
    """

    def __init__(self, vector: Screw):
        """Initialise with a retained screw supplying the operator components."""
        self.vector = vector

    def __matmul__(self, other: Screw):
        """Apply to a :class:`Screw`, equivalently :meth:`apply`.

        Raises:
            TypeError: The operand is not a :class:`Screw`.
        """
        if not isinstance(other, Screw):
            raise TypeError("Expected a Screw operand")

        return self.apply(other)

    def apply(self, other: Screw) -> Screw:
        """Apply the class's documented formula to another screw."""
        w = np.cross(self.vector.rotation, other.rotation) + np.cross(
            self.vector.translation, other.translation
        )

        v = np.cross(self.vector.rotation, other.translation)

        m = self.vector.magnitude * other.magnitude

        return Screw(translation=-v, rotation=-w, magnitude=m)

    def transpose(self) -> se3Adjoint:
        """Return the paired legacy adjoint, not a literal matrix transpose."""
        return se3Adjoint(self.vector)


def se3_bracket(left: Screw, right: Screw) -> Screw:
    """Return the angular-first screw Lie bracket.

    Args:
        left: Screw with stored angular/linear components ``(w, v)``.
        right: Screw with stored angular/linear components ``(u, t)``.

    Returns:
        Screw with components ``(cross(w, u), cross(w, t) - cross(u, v))``;
        magnitudes multiply. Unlike :class:`se3Adjoint`, this uses a positive
        ``cross(w, t)`` term. Component arithmetic may use compatible tracers.

    Examples:
        >>> from coker.toolkits.spatial import se3_bracket, Screw
        >>> se3_bracket(Screw.w_x(), Screw.w_y()).to_array().tolist()
        [0, 0, 1, 0, 0, 0]
    """
    w = np.cross(left.rotation, right.rotation)
    v = np.cross(left.rotation, right.translation) - np.cross(
        right.rotation, left.translation
    )

    return Screw(translation=v, rotation=w, magnitude=right.magnitude * left.magnitude)
