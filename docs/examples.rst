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


``examples/pytorch_explicit_network_training.py``
   Defines a small network as a Coker function with explicit weight and bias
   inputs, then trains caller-owned ``torch.nn.Parameter`` tensors through the
   lowered PyTorch graph.

``examples/pytorch_imported_module_training.py``
   Imports a native ``torch.nn.Module`` using a direct
   ``FunctionSignature`` declaration and trains the module's own parameters
   through the returned Coker function.

``examples/pytorch_mathematical_program_training.py``
   Builds the same style of explicit network as a CUDA float32
   ``MathematicalProgram``. The example selects Adam with
   ``PytorchNLPSolverOptions`` and lets the program own its packed decision
   vector.
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

PyTorch examples
----------------

The PyTorch examples require the optional extra:

.. code-block:: console

   uv run --extra pytorch python examples/pytorch_explicit_network_training.py
   uv run --extra pytorch python examples/pytorch_imported_module_training.py
   uv run --extra pytorch python examples/pytorch_mathematical_program_training.py

The mathematical-program example additionally requires CUDA. The direct
explicit-parameter and imported-module examples use ordinary PyTorch execution
and can run without the CUDA NLP solver.
