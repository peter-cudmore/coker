Toolkits
========

Coker's toolkits provide spatial algebra, rigid-body modelling, and mathematical
programming through the symbolic tracing and backend model.

Spatial algebra
---------------

:mod:`coker.toolkits.spatial` provides rigid-motion primitives:

- :class:`coker.toolkits.spatial.Rotation3`
- :class:`coker.toolkits.spatial.Isometry3`
- :class:`coker.toolkits.spatial.Screw`
- adjoint and Lie-algebra helpers such as ``hat`` and ``se3_bracket``


Rigid-body kinematics and dynamics
----------------------------------

:mod:`coker.toolkits.kinematics` provides
:class:`~coker.toolkits.kinematics.RigidBody`, joint types, and inertia
declarations for articulated models. Supported joints include welded,
revolute, prismatic, and free joints. Use the model to calculate forward
kinematics, Jacobians, energy, mass matrices, and inverse dynamics.

Define links and effectors, then wrap the required calculation in a Coker
function:

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

Run ``uv run scripts/double_pendulum.py`` for a visualisation example. The
script's inline metadata declares NumPy, Matplotlib, and the local Coker source.
To visualise your own model, install Matplotlib and import
``KinematicsVisualiser`` from ``coker.toolkits.kinematics.visualiser``. The main
kinematics module does not import Matplotlib.

Mathematical programs
---------------------

:mod:`coker.toolkits.codesign` builds optimisation problems over symbolic
variables. Use :class:`~coker.toolkits.codesign.ProblemBuilder` for algebraic
programs and :class:`coker.dynamics.VariationalProblemBuilder` for trajectory
problems. ``ProblemBuilder.build`` returns a public
:class:`~coker.toolkits.codesign.MathematicalProgram` callable with the declared
runtime arguments.

These examples require ``pip install "coker[casadi]"``, which includes NumPy.

.. code-block:: python

   import numpy as np
   from coker import VectorSpace
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

Use ``new_function_parameter`` to declare a function-valued decision with a
``FunctionSpace`` and finite ``FunctionParameter`` realisation. Use the returned
callable in objectives and constraints:

.. code-block:: python

   from coker import FunctionSpace, Scalar
   from coker.parameters.function_parameters import RadialBasisFunction
   from coker.toolkits.codesign import Minimise, ProblemBuilder

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

After a numeric solve, ``problem.parameters`` maps named decisions to their
reconstructed values. Scalars become floats; vectors and tensors become shaped
NumPy arrays. Function-valued decisions become ``FittedFunction`` instances
with these attributes:

- ``specification``: the original parameter declaration;
- ``space``: the declared ``FunctionSpace``;
- ``function``: the native callable;
- ``parameters``: the fitted basis.

Private solver captures are not part of the public ``(objective, *outputs)``
return tuple.

``MathematicalProgram.input_shape()`` describes the runtime arguments.
``output_shape()`` excludes the objective; ``result_shape`` includes it.
``backend`` names the solver backend. ``lower()`` returns a solver-pinned
``Function``, not a ``LoweredFunction``. Calling it solves the program and
returns ``(objective, *outputs)``.

``OptimisationModule`` is a supported solver implementation interface, not a
model builder. ``WeightedNorm`` is backend recognition metadata produced by
``weighted_norm``; model code should use the helper, not construct the record.

Pass flat IPOPT options through ``solver_options`` for CasADi/IPOPT solves:

.. code-block:: python

   from coker.backends.casadi import CasadiNLPSolverOptions
   from coker.toolkits.codesign import Minimise, ProblemBuilder, bounded

   with ProblemBuilder(
       solver_options=CasadiNLPSolverOptions(
           optimiser_options={
               "ipopt.max_iter": 50,
               "ipopt.max_cpu_time": 30.0,
           }
       )
   ) as builder:
       x = builder.new_variable("x", initial_value=0.0)
       builder.constraints = [bounded(x, -1.0, 1.0)]
       builder.objective = Minimise((x - 0.5) ** 2)
       builder.outputs = [x]
       problem = builder.build("casadi")

   objective, value = problem()

Further reading
----------------

- :doc:`how_to/robot_arm` for a robot-arm workflow.
- :doc:`dynamics` for trajectory problems and solver contracts.
- :doc:`examples` for repository examples and backend selection.
