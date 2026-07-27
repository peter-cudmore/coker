Examples and workflows
======================

This repository already contains several concrete entry points beyond the short
snippets in :doc:`getting_started`. If you want to learn the package by reading
real models, start here.

Recommended reading order
-------------------------

1. :doc:`getting_started` for the core ``function(...)`` API.
2. :doc:`backends` to decide how you want a model to execute.
3. :doc:`dynamics` if your work involves ODEs, fitting, or optimal control.
4. :doc:`toolkits` if your work is closer to robotics or block-model assembly.

Repository entry points
-----------------------

``examples/pid_example.py``
   Builds a motor plant and a PID-style controller from the system-modelling
   standard library. This is the best starting point for block/component
   composition.

``scripts/double_pendulum.py``
   Builds a two-link rigid-body model with ``RigidBody``, ``Revolute``,
   ``Inertia``, :class:`coker.toolkits.spatial.Isometry3`, and
   :class:`coker.toolkits.spatial.Screw`, then drives a visualiser sweep.

``tests/dynamical_systems/test_variational_solver.py``
   Contains the repository's clearest collection of dynamics and
   parameter-fitting examples, including constraints and regularisation.

``tests/toolkits/kinematics/test_kinematics_examples.py``
   Collects small but realistic robotics models: pendulums, SCARA, elbow-arm,
   and hexapod-leg scenarios.

``tests/benchmarks/benchmark_backends.py``
   Compares backend behaviour for function evaluation, ODE integration, and a
   variational parameter-fitting workload.

How to use the examples productively
------------------------------------

A good workflow is:

- start from ``examples/pid_example.py`` or the one-dimensional snippets in the
  tests;
- swap the backend only after the model is numerically correct under
  ``numpy``;
- move to ``casadi`` when the workflow becomes solve-heavy;
- use the kinematics and dynamics tests as executable specifications for more
  complex models.
