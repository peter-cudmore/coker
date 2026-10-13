Spatial algebra and robotics
============================

Current public imports
----------------------

Import spatial values from ``coker.toolkits.spatial`` and articulated-body
builders from ``coker.toolkits.kinematics``. These are the current exports;
this reference does not change backend defaults or export policy.

Frames, units and composition
-----------------------------

An ``Isometry3`` maps a child-frame point into parent-frame coordinates.
``a @ b`` applies ``b`` first, then ``a``. Points, axes and translations
normally have shape ``(3,)``. Homogeneous matrices have shape ``(4, 4)``.
Angles are radians; choose consistent length, mass and time units (SI gives
metres, kilograms, seconds and newton-metres). Units are not checked.
``Rotation3`` multiplication is historically reversed: ``a * b`` applies
``a`` first. Its current factory spelling is ``from_quaterion``.

.. doctest::

   >>> import numpy as np
   >>> from coker.toolkits.spatial import Isometry3, Rotation3, Screw, SE3Adjoint
   >>> turn = Isometry3(Rotation3(np.array([0., 0., 1.]), np.pi / 2))
   >>> shift = Isometry3(translation=np.array([1., 0., 0.]))
   >>> point = np.array([1., 0., 0.])
   >>> np.testing.assert_allclose((turn @ shift).apply(point), [0., 2., 0.], atol=1e-12)
   >>> np.testing.assert_allclose((turn @ shift).as_matrix(), turn.as_matrix() @ shift.as_matrix(), atol=1e-12)
   >>> np.testing.assert_allclose(SE3Adjoint(shift).apply(Screw.w_z()).to_array(), [0., 0., 1., 0., -1., 0.])

Spatial user values
-------------------

.. autoclass:: coker.toolkits.spatial.Rotation3
   :members: axis, angle, zero, cast, from_vector, from_quaterion, as_vector, as_quaternion, as_matrix, inverse, apply

.. autoclass:: coker.toolkits.spatial.Isometry3
   :members: identity, from_vector, as_vector, as_matrix, apply, inverse, transpose

.. autoclass:: coker.toolkits.spatial.UnitQuaternion
   :members: from_axis_angle, from_elements, to_elements, conjugate, inverse

.. autoclass:: coker.toolkits.spatial.Screw
   :members: zero, w_x, w_y, w_z, e_x, e_y, e_z, from_tuple, from_array, to_array, axis, pitch, exp

.. autofunction:: coker.toolkits.spatial.hat

.. autofunction:: coker.toolkits.spatial.quaternion_mul

Finite transforms versus infinitesimal operators
------------------------------------------------

Capitalized ``SE3Adjoint`` transports twists through an ``Isometry3``.
``SE3CoAdjoint`` implements its transpose-dual operation, not its
inverse-transpose. Components are angular then linear (moment then force for
wrenches). Lowercase ``se3Adjoint`` and ``se3CoAdjoint`` instead take a
``Screw`` and represent infinitesimal cross operations. The current
``se3Adjoint`` linear formula differs from ``se3_bracket``; the latter and
``se3CoAdjoint.apply`` currently fail with an unsupported ``Screw(angle=...)``
keyword. They remain exported, but are not supported working examples.
``Screw.as_matrix`` also has an incomplete array construction and is not a
supported conversion; use ``to_array`` and ``exp``.

.. autoclass:: coker.toolkits.spatial.SE3Adjoint
   :members: apply, as_matrix, inverse, transpose

.. autoclass:: coker.toolkits.spatial.SE3CoAdjoint
   :members: apply, as_matrix, transpose

.. autoclass:: coker.toolkits.spatial.se3Adjoint
   :members: apply, transpose

.. autoclass:: coker.toolkits.spatial.se3CoAdjoint
   :members: apply, transpose

.. autofunction:: coker.toolkits.spatial.se3_bracket

The exported ``e_x``, ``e_y`` and ``e_z`` are numeric Cartesian basis arrays;
``SE3_BASIS`` is the skew-map coefficient tensor used by ``hat``. They are
low-level algebra constants, not model constructors. ``spatial.types`` aliases
and private representation caches are implementation details.

Articulated-body user API
-------------------------

Links are inserted parent-first. Coordinates are flattened in link insertion
order, then joint-axis order, with shape ``(n,)`` for both NumPy values and
traced inputs. Forward kinematics returns effector-to-world transforms.
Spatial Jacobians have angular/spatial-linear rows; a point Jacobian requires
``J[3:, :] - hat(position) @ J[:3, :]``. The existing ``to_function`` bridge
performs that conversion and returns positions ``(3*e,)`` and Jacobians
``(3*e, n)`` for ``e`` effectors. At least one effector is required.

.. doctest::

   >>> from coker import VectorSpace, function
   >>> from coker.toolkits.kinematics import Inertia, Revolute, RigidBody
   >>> body = RigidBody()
   >>> link = body.add_link(body.WORLD, Isometry3.identity(), Revolute(Screw.w_z()), Inertia.zero())
   >>> tip = body.add_effector(link, Isometry3(translation=np.array([1., 0., 0.])))
   >>> positions, jacobian = body.to_function()(np.zeros(1))
   >>> np.testing.assert_allclose(positions, [1., 0., 0.])
   >>> np.testing.assert_allclose(jacobian, [[0.], [1.], [0.]])
   >>> fk = function([VectorSpace("q", 1)], implementation=body.to_function(), backend="numpy")
   >>> positions, jacobian = fk(np.array([np.pi / 2]))
   >>> np.testing.assert_allclose(positions, [0., 1., 0.], atol=1e-12)
   >>> np.testing.assert_allclose(jacobian, [[-1.], [0.], [0.]], atol=1e-12)

.. autoclass:: coker.toolkits.kinematics.Inertia
   :members: zero, as_matrix, as_matrix_origin

.. autoclass:: coker.toolkits.kinematics.RigidBody
   :members: add_link, add_effector, add_body, total_joints, joint_bases, joint_global_basis, joint_transforms, forward_kinematics, get_dependant_joints, get_dependent_links, spatial_manipulator_jacobian, spatial_single_manipulator_jacobian, body_manipulator_jacobian, potential_energy, inverse_dynamics, mass_matrix, to_function

.. autoclass:: coker.toolkits.kinematics.Revolute
   :members: axes

.. autoclass:: coker.toolkits.kinematics.Planar
   :members: axes

.. autoclass:: coker.toolkits.kinematics.Free
   :members: axes

.. autoclass:: coker.toolkits.kinematics.Weld

Joint extensions
----------------

Subclass ``JointType`` to supply ordered local screw axes. Private tree
accumulation and recursion methods are internal, not extension hooks.

.. autoclass:: coker.toolkits.kinematics.JointType
   :members: axes

Optional plotting
-----------------

Install ``matplotlib`` and import ``KinematicsVisualiser`` from
``coker.toolkits.kinematics.visualiser`` (it is not re-exported by
``kinematics``). Plotting accepts numeric configurations, creates a figure,
and ``animate_sweep`` calls ``pyplot.show``. It is not a tracing backend.
A headless environment can select Matplotlib's ``Agg`` backend before import.
Widget classes and their artist bookkeeping are internal plotting machinery.
MuJoCo conversion in ``kinematics.mujoco`` requires the separate optional
``mujoco`` dependency and is not a canonical package-root user API.

The body constructed above can be rendered headlessly with the optional
Matplotlib dependency:

.. doctest::

   >>> from matplotlib import pyplot as plt
   >>> from coker.toolkits.kinematics.visualiser import KinematicsVisualiser
   >>> visualiser = KinematicsVisualiser(body)
   >>> visualiser.update_values(np.array([np.pi / 2]))
   >>> visualiser.set_view(x_lim=(-1., 1.), y_lim=(-1., 1.), z_lim=(-1., 1.))
   >>> visualiser.figure.canvas.draw()
   >>> plt.close(visualiser.figure)

.. autoclass:: coker.toolkits.kinematics.visualiser.KinematicsVisualiser
   :members: update_values, set_view, animate_sweep
