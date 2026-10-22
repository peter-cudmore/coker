Kinematics API
==============

.. Author notes

   - Audience: users constructing articulated numerical or traceable models.
   - State frame directions, coordinate ordering, spatial conventions and shapes.
   - Keep the numerical-to-traceable example on the existing to_function bridge.
   - Reference only exported robotics declarations; visualisation is optional.

Import these declarations from ``coker.toolkits.kinematics``. They model a
parent-before-child link tree with one or more world-rooted branches. Link
indices and end-effector indices are separate from flattened motion-coordinate
indices. ``RigidBody.WORLD`` is the root-parent sentinel, not an effector.

Conventions
-----------

A rest pose maps child-frame coordinates into parent-frame coordinates.
``A @ B`` applies ``B`` first. At nonzero coordinates, each link's world pose
is its parent's world pose, then its rest pose, then its ordered local screw
exponentials. An effector pose adds its effector-to-link offset on the right.

Coordinates concatenate links in insertion order and each joint's axes in
their declared order. ``Weld`` contributes no coordinates. ``Free`` contributes
translations x/y/z followed by rotations x/y/z; these are ordered screw
exponentials, not a single rotation-vector exponential. ``Planar`` stores the
supplied axes without enforcing planar geometry or a particular axis count.
``Revolute`` accepts one screw, including a pure translation screw for sliding
motion. There is no exported ``Prismatic`` class.

Use metres, kilograms and seconds, and radians for unit rotational screws.
Spatial twists use angular then linear components. Spatial Jacobians have
shape ``(6, n)`` and model-coordinate column order, with zero columns for
unrelated branches. Their lower three rows are spatial-twist components, not
an effector's Cartesian velocity Jacobian. For position ``p``, use
``J[3:, :] - hat(p) @ J[:3, :]``. Body Jacobians apply the inverse effector-pose
adjoint to spatial Jacobians. Within a multi-axis link the spatial Jacobian
uses ancestor-transported rest screws without successive within-link updates;
do not assume a general product-of-exponentials derivative. Inverse dynamics
likewise treats a joint's local motion subspace as fixed. A chain of
single-axis links avoids that multi-axis interpretation.

Numerical and traced use
------------------------

``to_function()`` is the existing bridge between direct numerical kinematics
and a Coker function. With ``m`` effectors and ``n`` coordinates it returns
positions of shape ``(3*m,)`` and Cartesian Jacobians of shape ``(3*m, n)``.
Both stack effectors in insertion order. Configure at least one effector and
a model with nonempty Jacobian columns before calling the bridge.

.. testcode:: kinematics_api_bridge

   import numpy as np
   from coker import VectorSpace, function
   from coker.toolkits.kinematics import Inertia, Revolute, RigidBody
   from coker.toolkits.spatial import Isometry3, Screw

   model = RigidBody()
   link = model.add_link(
       model.WORLD, Isometry3.identity(), Revolute(Screw.w_z()), Inertia.zero()
   )
   model.add_effector(link, Isometry3(translation=np.array([1.0, 0.0, 0.0])))
   numeric = model.to_function()
   q = np.array([np.pi / 2])
   positions, jacobian = numeric(q)
   np.testing.assert_allclose(positions, [0.0, 1.0, 0.0], atol=1e-12)
   np.testing.assert_allclose(jacobian[:, 0], [-1.0, 0.0, 0.0], atol=1e-12)
   traced = function([VectorSpace("q", 1)], numeric, backend="numpy")
   traced_positions, traced_jacobian = traced(q)
   np.testing.assert_allclose(traced_positions, positions, atol=1e-12)
   np.testing.assert_allclose(traced_jacobian, jacobian, atol=1e-12)

Tracing uses the operations supported by the selected backend; it is not a
separate robot representation. See :doc:`../how_to/robot_arm` for a modelling
workflow and :doc:`../toolkits` for optional visualisation.

Declarations
------------

.. autoclass:: coker.toolkits.kinematics.Inertia
   :members:

.. autoclass:: coker.toolkits.kinematics.JointType
   :members:

.. autoclass:: coker.toolkits.kinematics.Weld
   :members:

.. autoclass:: coker.toolkits.kinematics.Free
   :members:

.. autoclass:: coker.toolkits.kinematics.Revolute
   :members:

.. autoclass:: coker.toolkits.kinematics.Planar
   :members:

.. autoclass:: coker.toolkits.kinematics.RigidBody
   :members:
