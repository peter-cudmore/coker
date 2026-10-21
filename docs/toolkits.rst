Toolkits
========

Coker's toolkits provide spatial algebra, rigid-body modelling, and mathematical
programming through the symbolic tracing and backend model.

Spatial algebra
---------------

Import rigid-motion values from ``coker.toolkits.spatial``. An ``Isometry3``
maps child-frame coordinates into the parent frame, and ``a @ b`` applies
``b`` first. Angles are radians; vector inputs normally have shape ``(3,)``.
The capitalised ``SE3Adjoint``/``SE3CoAdjoint`` act on finite transforms;
lowercase ``se3*`` operators take screws and are not interchangeable.

See :doc:`api/robotics` for value/member contracts, a spatial composition example,
and limitations of exported helpers.

Rigid-body kinematics and dynamics
----------------------------------

:mod:`coker.toolkits.kinematics` provides
:class:`~coker.toolkits.kinematics.RigidBody`, joint types, and inertia
declarations for articulated models. The exported joint types are ``Weld``,
``Revolute``, ``Planar`` and ``Free``. A translational screw supplied to
``Revolute`` defines a one-coordinate prismatic joint. Use the model to calculate
forward kinematics, Jacobians, energy, mass matrices, and inverse dynamics.

Define links and effectors, then wrap the required calculation in a Coker
function:

.. testcode:: toolkits_kinematics

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

   rest_position = fk(np.array([0.0]))
   assert rest_position.shape == (3,)
   np.testing.assert_allclose(rest_position, [0.0, 0.0, -0.5], atol=1e-12)
   np.testing.assert_allclose(
       fk(np.array([np.pi / 2])), [0.0, 0.5, 0.0], atol=1e-12,
   )

Run ``uv run scripts/double_pendulum.py`` for a visualisation example. The
script's inline metadata declares NumPy, Matplotlib, and the local Coker source.
To visualise your own model, install Matplotlib and import
``KinematicsVisualiser`` from ``coker.toolkits.kinematics.visualiser``. The main
kinematics module does not import Matplotlib.

System modelling
----------------

The system-modelling toolkit supports component metadata and port wiring, not
an executable simulation/compiler pipeline. Import ``BlockContainer``,
``Signal`` and ``list_components`` from ``coker.toolkits.system_modelling``;
standard blocks live in
``coker.toolkits.system_modelling.std_lib.common.signal_processing``.
Import component libraries before discovery: ``list_components()`` lists
registered classes as ``(name, slash_separated_module_path, None)`` tuples.
See :doc:`api/modelling` for wiring examples and precise limitations.

Mathematical programs
---------------------

:mod:`coker.toolkits.codesign` builds optimisation problems over symbolic
variables. Use :class:`~coker.toolkits.codesign.ProblemBuilder` for algebraic
programs and :class:`coker.dynamics.VariationalProblemBuilder` for trajectory
problems. ``ProblemBuilder.build`` returns a public
:class:`~coker.toolkits.codesign.MathematicalProgram` callable with the declared
runtime arguments.

Declare runtime inputs separately from optimisation decisions, then set a
scalar ``Minimise`` objective and a nonempty output list. The builder context
does not solve or consume it; calling the built program performs the solve.
``build()`` uses the current backend when none is named, without changing that
selection. See :doc:`api/optimisation` for results and backend limits.

These examples require ``pip install "coker[casadi]"``, which includes NumPy.

.. testcode:: toolkits_runtime_target

   import numpy as np
   from coker import VectorSpace
   from coker.backends.casadi import CasadiNLPSolverOptions
   from coker.toolkits.codesign import Minimise, ProblemBuilder, bounded

   options = CasadiNLPSolverOptions(
       optimiser_options={
           "ipopt.print_level": 0,
           "ipopt.sb": "yes",
           "print_time": False,
       }
   )
   with ProblemBuilder(
       arguments=[VectorSpace("target", 2)], solver_options=options
   ) as builder:
       (target,) = builder.arguments
       x = builder.new_variable(name="x", shape=(2,), initial_value=np.ones(2))
       delta = x - target
       builder.objective = Minimise(np.dot(delta, delta))
       builder.constraints = [bounded(x[0], target[0], target[0])]
       builder.outputs = [x]
       problem = builder.build("casadi")
   assert problem.parameters == {} and problem.solve_info is None

   objective, x_val = problem(np.array([3.0, -1.0]))
   np.testing.assert_allclose(objective, 0.0, atol=1e-8)
   assert x_val.shape == (2,)
   np.testing.assert_allclose(x_val, [3.0, -1.0], atol=1e-6)
   np.testing.assert_allclose(problem.parameters["x"], x_val, atol=1e-6)
   assert isinstance(objective, float)
   assert problem.solve_info.success

Use ``bounded(residual, lower, upper)`` for inclusive componentwise bounds.
Equal bounds express equality, and infinite bounds leave a side unbounded.
Bounds may depend on runtime parameters; CasADi evaluates them as residual
constraints at solve time. Symbolic ``<`` and ``>`` constraints are implemented
as non-strict numerical bounds.

Use ``new_function_parameter`` to declare a function-valued decision with a
``FunctionSpace`` and finite ``FunctionParameter`` realisation. Use the returned
callable in objectives and constraints:

.. testcode:: toolkits_function_parameter

   import numpy as np

   from coker import FunctionSpace, Scalar
   from coker.backends.casadi import CasadiNLPSolverOptions
   from coker.parameters.function_parameters import FittedFunction, RadialBasisFunction
   from coker.toolkits.codesign import Minimise, ProblemBuilder

   rate = FunctionSpace("rate", [Scalar("t")], [Scalar("rate")])
   declaration = RadialBasisFunction([0.0], 1.0, name="response")
   options = CasadiNLPSolverOptions(
       optimiser_options={
           "ipopt.print_level": 0,
           "ipopt.sb": "yes",
           "print_time": False,
       }
   )
   with ProblemBuilder(solver_options=options) as builder:
       response = builder.new_function_parameter(rate, declaration)
       value = response(0.0)
       builder.objective = Minimise(
           (value - 0.5) ** 2 + (response(1.0) - 0.25) ** 2
       )
       builder.outputs = [value]
       problem = builder.build("casadi")

   objective, value = problem()
   fitted_response = problem.parameters["response"]
   assert isinstance(fitted_response, FittedFunction)
   assert fitted_response.specification is declaration
   assert fitted_response.space is rate
   np.testing.assert_allclose(objective, 0.0, atol=1e-8)
   np.testing.assert_allclose(value, 0.5, atol=1e-6)
   assert fitted_response.parameters[0].shape == (2,)
   np.testing.assert_allclose(fitted_response(0.0), value, atol=1e-6)
   np.testing.assert_allclose(fitted_response(1.0), 0.25, atol=1e-6)
   assert value.shape == (1,)
   assert problem.solve_info.success

After a numeric solve, ``problem.parameters`` maps named decisions to their
reconstructed values. NumPy and CasADi return floats for scalar decisions and
shaped NumPy arrays for vectors/tensors; PyTorch preserves native tensors.
Function-valued decisions become ``FittedFunction`` instances
with these attributes:

- ``specification``: the original parameter declaration;
- ``space``: the declared ``FunctionSpace``;
- ``function``: the native callable;
- ``parameters``: the fitted basis.

Private solver captures are not part of the public ``(objective, *outputs)``
return tuple.
The mapping and result arrays have no copy/isolation guarantee. Copy values
when keeping a snapshot. ``solve_info`` starts as ``None`` and is updated on
solver success or ``SolveFailure``. A failed solve returns no tuple and keeps
the previous successful parameters; configuration or argument errors may leave
the previous status unchanged. Success does not guarantee global optimality.

``MathematicalProgram.input_shape()`` describes the runtime arguments.
``output_shape()`` excludes the objective; ``result_shape`` includes it.
``backend`` names the solver backend. ``lower()`` returns a solver-pinned
``Function``, not a ``LoweredFunction``. Calling it solves the program and
returns ``(objective, *outputs)``.

``OptimisationModule`` is a supported solver implementation interface, not a
model builder. ``WeightedNorm`` is backend recognition metadata produced by
``weighted_norm``; model code should use the helper, not construct the record.

Pass flat IPOPT options through ``solver_options`` for CasADi/IPOPT solves:

.. testcode:: toolkits_solver_options

   import numpy as np

   from coker.backends.casadi import CasadiNLPSolverOptions
   from coker.toolkits.codesign import Minimise, ProblemBuilder, bounded

   with ProblemBuilder(
       solver_options=CasadiNLPSolverOptions(
           optimiser_options={
               "ipopt.max_iter": 50,
               "ipopt.max_cpu_time": 30.0,
               "ipopt.print_level": 0,
               "ipopt.sb": "yes",
               "print_time": False,
           }
       )
   ) as builder:
       x = builder.new_variable("x", initial_value=0.0)
       builder.constraints = [bounded(x, -1.0, 1.0)]
       builder.objective = Minimise((x - 0.5) ** 2)
       builder.outputs = [x]
       problem = builder.build("casadi")

   objective, value = problem()
   np.testing.assert_allclose(objective, 0.0, atol=1e-8)
   assert value.shape == (1,)
   np.testing.assert_allclose(value, [0.5], atol=1e-6)

Further reading
----------------

- :doc:`how_to/robot_arm` for a robot-arm workflow.
- :doc:`dynamics` for trajectory problems and solver contracts.
- :doc:`examples` for repository examples and backend selection.
