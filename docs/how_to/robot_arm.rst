Model a two-link robot arm
==========================

Build a planar arm, compile its forward kinematics with the NumPy backend,
and plot three configurations. This recipe assumes coordinate-frame,
joint-coordinate and Coker function knowledge. Install
``pip install coker matplotlib``.

The shoulder is fixed at the world origin. Both joints rotate about their
local positive z axis; zero joint angles align the links with positive x.
Lengths are in metres and angles are in radians. No dynamics are evaluated,
so the links use zero inertia.

Build and evaluate the model
----------------------------

1. Add the shoulder and elbow links with their joint axes and fixed transforms.
2. Add effectors at the elbow and tip, in that order.
3. Compile their positions into a Coker function and evaluate the three joint
   configurations. Check the positions and link lengths before plotting.

.. plot::
   :include-source: true

   import matplotlib.pyplot as plt
   import numpy as np

   from coker import VectorSpace, function
   from coker.toolkits.kinematics import Inertia, Revolute, RigidBody
   from coker.toolkits.spatial import Isometry3, Screw

   length_1, length_2 = 1.0, 0.7
   arm = RigidBody()
   shoulder = arm.add_link(
       arm.WORLD,
       Isometry3.identity(),
       Revolute(Screw.w_z()),
       Inertia.zero(),
   )
   elbow = arm.add_link(
       shoulder,
       Isometry3(translation=np.array([length_1, 0.0, 0.0])),
       Revolute(Screw.w_z()),
       Inertia.zero(),
   )
   # Effector order defines the order of the two returned positions.
   arm.add_effector(
       shoulder, Isometry3(translation=np.array([length_1, 0.0, 0.0]))
   )
   arm.add_effector(
       elbow, Isometry3(translation=np.array([length_2, 0.0, 0.0]))
   )
   assert arm.total_joints() == 2
   positions = function(
       [VectorSpace("q", 2)],
       lambda q: np.concatenate(
           [frame.translation for frame in arm.forward_kinematics(q)]
       ),
       backend="numpy",
   )

   configurations = [
       (np.array([0.0, 0.0]), np.array([1.7, 0.0, 0.0]), "Straight"),
       (np.array([np.pi / 2, 0.0]), np.array([0.0, 1.7, 0.0]), "Upright"),
       (np.array([np.pi / 2, -np.pi / 2]), np.array([0.7, 1.0, 0.0]), "Bent"),
   ]
   fig, ax = plt.subplots()
   for q, known_tip, label in configurations:
       elbow_position, tip_position = np.asarray(positions(q)).reshape(2, 3)
       analytic_elbow = np.array(
           [length_1 * np.cos(q[0]), length_1 * np.sin(q[0]), 0.0]
       )
       analytic_tip = analytic_elbow + np.array(
           [length_2 * np.cos(q.sum()), length_2 * np.sin(q.sum()), 0.0]
       )
       np.testing.assert_allclose(elbow_position, analytic_elbow, atol=1e-12)
       np.testing.assert_allclose(tip_position, analytic_tip, atol=1e-12)
       np.testing.assert_allclose(tip_position, known_tip, atol=1e-12)
       direct_tip = arm.forward_kinematics(q)[1].translation
       np.testing.assert_allclose(tip_position, direct_tip, atol=1e-12)
       points = np.vstack((np.zeros(3), elbow_position, tip_position))
       np.testing.assert_allclose(
           np.linalg.norm(np.diff(points, axis=0), axis=1),
           [length_1, length_2],
           atol=1e-12,
       )
       ax.plot(points[:, 0], points[:, 1], "o-", label=label)

   ax.scatter(0.0, 0.0, marker="s", color="black", label="Fixed base", zorder=4)
   ax.set(xlabel="x [m]", ylabel="y [m]", title="Two-link arm: forward kinematics")
   ax.set_aspect("equal", adjustable="box")
   ax.grid(True)
   ax.legend()
   fig.tight_layout()
   plt.show()

Inspect and adapt the result
----------------------------

The assertions compare the compiled positions with the analytic planar
formula and the tip with direct model evaluation, using a tolerance of
``1e-12``. In the bent configuration, the elbow is at ``(0, 1, 0)`` and the tip
at ``(0.7, 1, 0)``.

Change the link lengths or joint configurations using the frame and unit
conventions above, then update the reference positions. The plot needs no
separate robot visualiser. For dynamics, replace the zero inertias with
physical mass properties.

.. Author notes

   - Assume coordinate-frame, joint-coordinate and Coker function knowledge.
   - Keep links, joint types and axes, dimensions, frame conventions and inertial assumptions explicit.
   - Show model construction, effector selection, forward-kinematics compilation and evaluation; include dynamics only if it serves the task.
   - Retain reference positions and a labelled plot with units and configurations.
   - Identify the evaluation backend and plotting dependencies; distinguish modelling from visualisation.
   - Retain hand-checkable configurations and numerical tolerances; include physical checks if dynamics are added.
