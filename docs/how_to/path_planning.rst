Plan a path through a corridor
==============================

Plan seven waypoints between fixed endpoints, inside a rectangular corridor
and through an elevated gate. This recipe assumes basic constrained
optimisation and Coker function knowledge. It uses the codesign API, CasADi's
IPOPT solver, NumPy and Matplotlib. Install
``pip install "coker[casadi]" matplotlib``.

The objective is the sum of squared segment lengths: discrete path energy,
not Euclidean arc length. Fixed horizontal sampling makes this a small convex
quadratic problem.

Build and solve the problem
---------------------------

1. Declare the waypoint coordinates and initial guesses with ``ProblemBuilder``.
2. Minimise path energy and constrain the endpoints, corridor and middle gate.
3. Build with the CasADi backend, solve, then check status and feasibility
   before plotting.

.. plot::
   :include-source: true

   import matplotlib.pyplot as plt
   import numpy as np

   from coker.backends.casadi import CasadiNLPSolverOptions
   from coker.toolkits.codesign import Minimise, ProblemBuilder

   count = 7
   start = np.array([0.0, 0.0])
   goal = np.array([4.0, 0.0])
   x_guess = np.linspace(start[0], goal[0], count)
   y_guess = np.array([0.0, 0.4, 0.7, 1.0, 0.7, 0.4, 0.0])
   options = CasadiNLPSolverOptions(
       optimiser_options={
           "ipopt.tol": 1e-9,
           "ipopt.max_iter": 100,
           "ipopt.print_level": 0,
           "print_time": False,
       }
   )
   with ProblemBuilder(solver_options=options) as builder:
       x = builder.new_variable("x", shape=(count,), initial_value=x_guess)
       y = builder.new_variable("y", shape=(count,), initial_value=y_guess)
       builder.objective = Minimise(
           sum(
               (x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2
               for i in range(count - 1)
           )
       )
       builder.constraints = [
           x[0] == start[0],
           y[0] == start[1],
           x[count - 1] == goal[0],
           y[count - 1] == goal[1],
           y[3] >= 0.8,
           y[3] <= 1.1,
       ]
       builder.constraints += [x[i] == x_guess[i] for i in range(1, count - 1)]
       for i in range(count):
           builder.constraints += [
               x[i] >= 0.0, x[i] <= 4.0,
               y[i] >= -0.2, y[i] <= 1.2,
           ]
       builder.outputs = [x, y]
       problem = builder.build("casadi")

   objective, x_value, y_value = problem()
   assert problem.solve_info is not None
   assert problem.solve_info.success
   path = np.column_stack((x_value, y_value))
   tolerance = 1e-6
   np.testing.assert_allclose(path[0], start, atol=tolerance)
   np.testing.assert_allclose(path[-1], goal, atol=tolerance)
   np.testing.assert_allclose(x_value, x_guess, atol=tolerance)
   assert np.all(path >= np.array([0.0, -0.2]) - tolerance)
   assert np.all(path <= np.array([4.0, 1.2]) + tolerance)
   assert 0.8 - tolerance <= y_value[3] <= 1.1 + tolerance
   expected_y = np.array([0.0, 0.8 / 3, 1.6 / 3, 0.8, 1.6 / 3, 0.8 / 3, 0.0])
   np.testing.assert_allclose(y_value, expected_y, atol=tolerance)
   measured_energy = np.sum(np.diff(path, axis=0) ** 2)
   expected_energy = 6 * (4.0 / 6) ** 2 + 6 * (0.8 / 3) ** 2
   np.testing.assert_allclose(objective, measured_energy, atol=tolerance)
   np.testing.assert_allclose(objective, expected_energy, atol=tolerance)
   initial_energy = np.sum(np.diff(x_guess) ** 2 + np.diff(y_guess) ** 2)
   assert objective < initial_energy

   fig, ax = plt.subplots()
   ax.fill_between([0.0, 4.0], -0.2, 1.2, color="lightgray", label="Corridor")
   ax.hlines([-0.2, 1.2], 0.0, 4.0, colors="gray")
   ax.plot([2.0, 2.0], [0.8, 1.1], color="tab:green", linewidth=7, label="Via gate")
   ax.plot(path[:, 0], path[:, 1], "o-", label="Optimised waypoints")
   ax.scatter(*start, marker="s", s=70, label="Start")
   ax.scatter(*goal, marker="*", s=110, label="Goal")
   ax.set(xlabel="x [m]", ylabel="y [m]", title=f"Path energy: {float(objective):.4f}")
   ax.set_aspect("equal", adjustable="box")
   ax.legend()
   fig.tight_layout()
   plt.show()

Inspect the path
----------------

The middle waypoint lies at the gate's lower edge. Equal increments on either
side give an energy of approximately ``3.093333``. The assertions check the
endpoints, corridor bounds, gate and objective within ``1e-6``.

Constraints apply to waypoints. Straight interpolation stays inside this
corridor because the rectangle is convex, and crosses the gate at the middle
waypoint. This does not establish collision clearance for arbitrary obstacles,
curved interpolation or a robot with nonzero width.

Adapt the geometry
------------------

Change the endpoints, corridor bounds or gate constraints, then update the
feasibility checks and expected energy. Different obstacle geometry may need
a different formulation, not just different bounds.

.. Author notes

   - Assume constrained-optimisation knowledge and experience evaluating Coker functions.
   - Keep explicit endpoints, waypoint variables, geometry and constraints.
   - Show declaration, objective and constraints, build, solve and path extraction with coker.toolkits.codesign.
   - Show the objective and labelled path plot; distinguish waypoint feasibility from continuous-path safety.
   - Identify the backend, solver options, optional dependencies and plotting requirements.
   - Retain endpoint and constraint tolerances, solve-status checks and checks against the stated model.
