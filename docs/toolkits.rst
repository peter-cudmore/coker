Toolkit guides
==============

Beyond the core symbolic API, Coker ships several domain-focused toolkits under
``src/coker/toolkits/``. They share the same tracing and backend model, so the
same function/back-end ideas apply across robotics, optimisation, and
system-modelling workflows.

Spatial algebra
---------------

:mod:`coker.toolkits.spatial` exports the low-level rigid-motion primitives used
throughout the robotics code:

- :class:`coker.toolkits.spatial.Rotation3`
- :class:`coker.toolkits.spatial.Isometry3`
- :class:`coker.toolkits.spatial.Screw`
- adjoint and Lie-algebra helpers such as ``hat`` and ``se3_bracket``

These types show up directly in the kinematics tests and in the double-pendulum
script.

Rigid-body kinematics and dynamics
----------------------------------

:mod:`coker.toolkits.kinematics` exposes :class:`~coker.toolkits.kinematics.RigidBody`
and joint/inertia types for articulated models. The test suite demonstrates:

- single-pendulum forward kinematics;
- double-pendulum mass-matrix and inverse-dynamics checks;
- SCARA-style manipulator Jacobians;
- a hexapod leg example with symbolic forward kinematics.

A representative pattern is:

.. code-block:: python

   import numpy as np
   from coker import VectorSpace, function
   from coker.toolkits.kinematics import Inertia, Revolute, RigidBody
   from coker.toolkits.spatial import Isometry3, Screw

   model = RigidBody()
   link = model.add_link(
       parent=model.WORLD,
       at=Isometry3.identity(),
       joint=Revolute(Screw.from_tuple(1, 0, 0, 0, 0, 0)),
       inertia=Inertia(
           centre_of_mass=Isometry3(translation=np.array([0, 0, -0.25])),
           mass=1,
           moments=np.array([1, 0, 0, 1, 0, 1]),
       ),
   )
   tip = model.add_effector(parent=link, at=Isometry3(translation=np.array([0, 0, -0.5])))

   origin = np.zeros((3,), dtype=float)
   fk = function(
       [VectorSpace("q", 1)],
       implementation=lambda q: model.forward_kinematics(q)[tip].apply(origin),
       backend="numpy",
   )

System modelling
----------------

The system-modelling toolkit provides component/block modelling helpers and a
standard-library discovery API. ``list_components()`` walks registered standard
library components and returns ``(component_name, library_path, hint)`` tuples.
The repository test checks that these components resolve from
``coker/toolkits/system_modelling/std_lib``.

The most concrete entry point today is ``examples/pid_example.py``, which builds
both a plant model and a PID-style controller from standard-library blocks such
as ``Gain``, ``Integrator``, ``Difference``, and ``Sum``.

Codesign-style mathematical programs
------------------------------------

:mod:`coker.toolkits.codesign` provides a small builder for optimisation-style
programs over symbolic variables. It is useful when you want a compact host-side
API instead of assembling a :class:`coker.dynamics.VariationalProblem`
directly.

.. code-block:: python

   from coker import FunctionSpace, Scalar, VectorSpace
   from coker.dynamics import RadialBasisFunction
   from coker.toolkits.codesign import Minimise, ProblemBuilder

   with ProblemBuilder(arguments=[VectorSpace("target", 2)]) as builder:
       (target,) = builder.arguments
       x = builder.new_variable(name="x", shape=(2,), initial_value=np.ones(2))
       delta = x - target
       builder.objective = Minimise(np.dot(delta, delta))
       builder.outputs = [x]
       problem = builder.build("casadi")

   objective, x_val = problem(np.array([3.0, -1.0]))

Use ``bounded(residual, lower, upper)`` when bounds are part of the model.
Bounds may depend on runtime parameters; CasADi evaluates them as residual
constraints at solve time.

Function-valued decisions use ``new_function_parameter`` with a declared
``FunctionSpace`` and finite ``FunctionParameter`` realization. The returned
callable participates directly in the objective and constraints:

.. code-block:: python

   rate = FunctionSpace("rate", [Scalar("t")], [Scalar("rate")])
   declaration = RadialBasisFunction([0.0], 1.0, name="response")
   with ProblemBuilder() as builder:
       response = builder.new_function_parameter(rate, declaration)
       value = response(0.0)
       builder.objective = Minimise((value - 0.5) ** 2)
       builder.outputs = [value]
       problem = builder.build("casadi")

   objective, value = problem()
   fitted_response = problem.parameters["response"]

After a numeric solve, ``problem.parameters`` maps each named decision to its
reconstructed value. Scalar decisions are floats and vector or tensor decisions
are shaped NumPy arrays. Function-valued decisions are ``FittedFunction``
instances: they retain the original declaration in ``specification``, the
declared ``FunctionSpace`` in ``space``, their native callable in ``function``,
and the fitted basis in ``parameters``. Their private solver captures never
change the public ``(objective, *outputs)`` tuple.

For bounded CasADi/IPOPT solves, configure CasADi's flat IPOPT options through
the builder's ``solver_options``:

.. code-block:: python

   with ProblemBuilder(
       solver_options=CasadiNLPSolverOptions(
           optimiser_options={
               "ipopt.max_iter": 50,
               "ipopt.max_cpu_time": 30.0,
           }
       )
   ) as builder:
       ...

Where to continue
-----------------

- :doc:`examples` for the main repository entry points.
- ``tests/toolkits/kinematics/test_kinematics_examples.py`` for richer robotics
  models.
- ``tests/toolkits/test_codesign.py`` for more codesign scenarios, including
  constraint handling and failure cases.
