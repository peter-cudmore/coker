Spatial algebra
===============

.. Author notes

   - Audience: users constructing rotations, rigid transforms, twists and wrenches.
   - State operator order, angular-first storage, units and native/tracing limits.
   - Keep numerical examples in this source; distinguish transform duality from
     the implemented Lie-algebra signs without changing mathematical behaviour.

Use rotations, isometries, screws, ``UnitQuaternion``, ``quaternion_mul``,
the basis vectors and ``hat`` from ``coker.toolkits.spatial``. These are the
spatial contracts, not a promise of support for the separate system-modelling
toolkit.

Vectors, units and ownership
-----------------------------------

Native spatial vectors are NumPy arrays with shape ``(3,)``, not lists or
``(3, 1)`` columns. Angles and rotation vectors use radians; translations use
a caller-chosen length unit. Screws store angular components first, then
linear components, with a shared scalar magnitude. A twist therefore uses
angular then linear velocity; a wrench uses moment then force. Consistent
units are the caller's responsibility.

Objects generally retain input arrays and return shared components. Scalar
screw multiplication also shares its component arrays. Treat these arrays as
immutable. ``Rotation3`` caches its quaternion or materialised axis-angle;
mutating the original axis, wrapped quaternion or returned components does
not invalidate those caches. Identity constructors and unit screw helpers
create fresh arrays, but the module-level basis vectors are shared.

``apply`` and selected quaternion/screw arithmetic support compatible Coker
tracers. Native type dispatch, equality, zero checks, normalisation and
axis-angle materialisation are not a general symbolic interface.
``Rotation3.cast`` accepts only a rotation, unit quaternion or native NumPy
``(3,)`` rotation vector; ``Rotation3.from_vector`` is a native numerical
constructor. ``Isometry3 @ point`` dispatches on native arrays; traced points
use ``Isometry3.apply``. These limits do not imply support for arbitrary
third-party array types.

Composition and application
---------------------------

A quaternion acts actively through ``q * vector * q.inverse()``. Its Hamilton
product ``q * p`` applies ``p`` first and then ``q``. In contrast,
``Rotation3`` multiplication ``a * b`` applies ``a`` first and then ``b``:
its matrix is ``R_b @ R_a``. ``Rotation3 @ vector`` and ``apply`` rotate a
vector without adding a translation.

An isometry acts as ``R @ point + p``. Its composition ``a @ b`` applies
``b`` first and then ``a``, so its matrix is ``T_a @ T_b``.
``Isometry3.transpose()`` returns the inverse transform, not the literal
matrix transpose. A ``(4, 1)`` native column is accepted by ``@``; translation
is always added and the fourth entry is copied unchanged. Use this route
only for homogeneous points with fourth entry one, not directions.

.. testcode:: spatial_composition

   import numpy as np
   from coker.toolkits.spatial import Isometry3, Rotation3

   x = np.array([1., 0., 0.])
   y = np.array([0., 1., 0.])
   z = np.array([0., 0., 1.])
   a = Rotation3(z, np.pi / 2)
   b = Rotation3(x, np.pi / 2)
   assert np.allclose((a * b).apply(x), b.apply(a.apply(x)))
   assert np.allclose((a * b).apply(x), z)
   assert not np.allclose((a * b).apply(x), (b * a).apply(x))
   left = Isometry3(a, x)
   right = Isometry3(b, y)
   point = np.array([2., 3., 4.])
   assert np.allclose((left @ right).apply(point), left.apply(right.apply(point)))
   assert np.allclose((left @ right).as_matrix(), left.as_matrix() @ right.as_matrix())
   assert np.allclose(left.inverse().apply(left.apply(point)), point)

Rotations and quaternions
-------------------------

``Rotation3(axis, angle)`` should receive a unit axis. Quaternion conversion
normalises a nonzero axis, but ``as_vector()`` multiplies the stored axis by
the angle without normalising it. A native zero axis selects identity in
``Rotation3``. Do not pass a zero axis to ``UnitQuaternion.from_axis_angle``;
construct identity with ``UnitQuaternion.from_elements(1, 0, 0, 0)`` instead.
``UnitQuaternion`` itself does not enforce unit norm; its ``inverse`` is the
conjugate and is only an inverse for a unit quaternion. Equality compares
components exactly, not equivalent rotations or opposite quaternion signs.
The existing constructor spelling is ``Rotation3.from_quaterion``.

.. autoclass:: coker.toolkits.spatial.Rotation3
   :members:
   :special-members: __mul__, __matmul__, __eq__

.. autoclass:: coker.toolkits.spatial.Isometry3
   :members:
   :special-members: __matmul__, __eq__

.. autoclass:: coker.toolkits.spatial.UnitQuaternion
   :members:
   :exclude-members: q_0, v
   :special-members: __mul__, __rmul__, __eq__

.. autofunction:: coker.toolkits.spatial.quaternion_mul

Screws and packed representations
----------------------------------------

``Screw.to_array()`` returns ``magnitude * [rotation, translation]`` with
shape ``(6,)``. ``from_array`` accepts that flat shape; ``from_tuple`` accepts
six scalar arguments. Both normalise a nonzero native angular block, keeping
the norm in ``magnitude``; pure translation retains magnitude one.
``Isometry3.as_vector()`` instead returns a ``(6, 1)`` column containing a
rotation vector and translation, not an SE(3) logarithm. Flatten that column
before passing it to ``Isometry3.from_vector``.

For a unit angular direction, ``Screw.exp(angle)`` uses
``alpha = magnitude * angle`` as the rotation angle and computes the
corresponding translation from the stored screw components. For pure native
translation it returns displacement ``translation * alpha``. The current
zero-magnitude branch uses ``alpha = angle`` instead, so
``Screw.zero().exp()`` must not be assumed to be identity. Unary screw
negation resets magnitude to one; use ``screw * -1`` to preserve scaling.
``pitch`` is exactly ``dot(rotation, translation) * magnitude``, not a
normalised geometric pitch for arbitrary stored components.

``Screw.as_matrix()`` is currently unsupported: its NumPy concatenation
arguments and dimensions are incompatible. Use ``to_array()`` for screw
serialisation, or ``exp(...).as_matrix()`` for a transform matrix.

.. autoclass:: coker.toolkits.spatial.Screw
   :members:
   :exclude-members: rotation, translation, magnitude
   :special-members: __mul__, __neg__, __eq__

Transform adjoints and duality
-------------------------------------

For transform ``(R, p)``, ``SE3Adjoint`` has matrix
``A = [[R, 0], [hat(p) @ R, R]]`` on angular-first twists.
``SE3CoAdjoint`` implements ``A.T``: it sends moment/force ``(m, f)`` to
``(R.T @ (m - cross(p, f)), R.T @ f)``. This is a transpose action, not the
inverse-transpose action for the same transform. The following pairing uses
that precise convention.

.. testcode:: spatial_transform_duality

   import numpy as np
   from coker.toolkits.spatial import (
       Isometry3, Rotation3, Screw, SE3Adjoint, SE3CoAdjoint,
   )

   transform = Isometry3(
       Rotation3(np.array([0., 0., 1.]), np.pi / 2),
       np.array([1., 2., 3.]),
   )
   twist = Screw.from_array(np.array([1., 2., 3., 4., 5., 6.]))
   wrench = Screw.from_array(np.array([2., -1., 4., -3., 5., 1.]))
   adjoint = SE3Adjoint(transform)
   coadjoint = SE3CoAdjoint(transform)
   assert np.allclose(coadjoint.as_matrix(), adjoint.as_matrix().T)
   assert np.allclose(
       (adjoint @ twist).to_array() @ wrench.to_array(),
       twist.to_array() @ (coadjoint @ wrench).to_array(),
   )
   assert np.allclose(
       (adjoint.inverse() @ (adjoint @ twist)).to_array(), twist.to_array()
   )

.. autoclass:: coker.toolkits.spatial.SE3Adjoint
   :members:
   :special-members: __matmul__, __rmatmul__, __call__

.. autoclass:: coker.toolkits.spatial.SE3CoAdjoint
   :members:
   :special-members: __matmul__, __call__

Lie-algebra operators
---------------------

The lower-case ``se3Adjoint`` and ``se3CoAdjoint`` are screw-based operators,
not transform adjoints. Their implemented signs are not interchangeable with
standard Lie-algebra transpose identities. For stored screw ``(w, v)`` and
operand ``(u, t)``, ignoring the common product of magnitudes:

* ``se3_bracket`` returns ``(cross(w, u), cross(w, t) - cross(u, v))``.
* ``se3Adjoint`` returns ``(cross(w, u), cross(v, u) - cross(w, t))``.
* ``se3CoAdjoint`` returns
  ``(-cross(w, u) - cross(v, t), -cross(w, t))``.

Their ``transpose()`` methods return the paired class on the same screw;
the resulting operators are not literal matrix transposes. In particular,
the adjoint's lower-right block is ``-hat(w)``, whose transpose would be
``+hat(w)``, whereas the coadjoint's lower-right block is ``-hat(w)``.
Do not infer twist/wrench duality for this pair from the class names.

.. testcode:: spatial_algebra_signs

   import numpy as np
   from coker.toolkits.spatial import Screw, se3_bracket, se3CoAdjoint, se3Adjoint

   angular = Screw.w_x()
   linear = Screw.e_y()
   assert np.array_equal(se3_bracket(angular, linear).to_array(), [0, 0, 0, 0, 0, 1])
   assert np.array_equal((se3Adjoint(angular) @ linear).to_array(), [0, 0, 0, 0, 0, -1])
   assert np.array_equal((se3CoAdjoint(angular) @ linear).to_array(), [0, 0, 0, 0, 0, -1])

.. autoclass:: coker.toolkits.spatial.se3Adjoint
   :members:
   :special-members: __matmul__

.. autoclass:: coker.toolkits.spatial.se3CoAdjoint
   :members:
   :special-members: __matmul__

.. autofunction:: coker.toolkits.spatial.se3_bracket

Basis helpers
-------------

``hat(u) @ v`` equals ``numpy.cross(u, v)``; this fixes the cross-product sign
used above. The shared native arrays ``e_x``, ``e_y`` and ``e_z`` are unit
Cartesian basis vectors. ``SE3_BASIS`` has shape ``(3, 3, 3)`` and supplies the
skew generators used by ``hat``. Do not mutate these module-level arrays.

.. autofunction:: coker.toolkits.spatial.hat

.. py:data:: coker.toolkits.spatial.SE3_BASIS

   Shared ``(3, 3, 3)`` NumPy array of skew-symmetric rotation generators.

.. py:data:: coker.toolkits.spatial.e_x

   Shared NumPy unit vector ``[1, 0, 0]``.

.. py:data:: coker.toolkits.spatial.e_y

   Shared NumPy unit vector ``[0, 1, 0]``.

.. py:data:: coker.toolkits.spatial.e_z

   Shared NumPy unit vector ``[0, 0, 1]``.
