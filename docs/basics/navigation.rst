How to use this documentation
=============================

After :doc:`getting_started`, continue with
:doc:`../tutorials/function_evaluation` and follow the tutorials in order.
They assume basic Python functions and NumPy arrays, but no symbolic-programming
experience. They introduce scalar functions, symbolic arrays, conditionals,
and backend selection.

Use the other sections as needed:

* :doc:`../how_to/index` gives task-focused recipes for modelling, optimisation,
  and robotics. It assumes you can build and evaluate a Coker function.
* :doc:`../architecture/index` explains the execution model and design tradeoffs.
* :doc:`../api/index` lists public interfaces and their contracts.

Reading routes
--------------

Choose a route once you can construct and evaluate a Coker function:

* **Numerical modelling:** complete :doc:`../tutorials/index`, then use
  :doc:`../how_to/simulation` to build and integrate a dynamical model.
* **Optimisation:** read :doc:`../api/parameters` for decision-variable
  declarations, then follow :doc:`../how_to/system_identification` to fit model
  parameters or :doc:`../how_to/path_planning` to optimise a trajectory.
* **Robotics:** follow :doc:`../how_to/robot_arm`, then use
  :doc:`../toolkits` to look up spatial algebra and kinematics helpers.

.. Author notes

   - Suggest a reading path for numerical modelling, optimisation, and robotics.
   - Explain which prerequisites tutorials assume and how they differ from recipes.
   - Identify the next page after the first successful application.
