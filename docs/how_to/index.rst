How-to guides
=============

Choose a recipe for your task. Each gives the dependencies, steps, and checks
needed to adapt the example.

The recipes assume you can construct and evaluate a Coker function. Start with
:doc:`../tutorials/index` if you are new to Coker.

Backend interoperability
------------------------

Bring a native model into a Coker computation, or expose a Coker model to
another library without rewriting its equations.

.. toctree::
   :maxdepth: 1

   backend_interoperability

Solver configuration
--------------------

Configure the solver for an existing optimisation problem and inspect its
termination status and numerical residuals.

.. toctree::
   :maxdepth: 1

   casadi_solver_tuning

Modelling and optimisation
--------------------------

Simulate a system, fit its parameters, plan a path, or model a robot arm.

.. toctree::
   :maxdepth: 1

   simulation
   system_identification
   path_planning
   robot_arm

Related reference guides
------------------------

Use these guides to look up APIs when adapting a recipe:

.. toctree::
   :maxdepth: 1

   ../backends
   ../dynamics
   ../toolkits


.. Author notes

   - Keep recipes focused on common tasks and patterns for users who know Coker.
   - Future additions, not yet published recipes:

     1. How to build an extended Kalman filter (EKF).
     2. How to tune a PID controller automatically using data.

