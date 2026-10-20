How to fit a dynamical model to data
====================================

Estimate a dynamical parameter from observations. This recipe assumes you can
:doc:`simulate a model <simulation>` and understand least squares.
See :doc:`../dynamics` for the modelling API and
:doc:`../dynamics/system_analysis` for structural identifiability, a separate
question from parameter fitting.

1. Prepare the fit
------------------

Install ``coker[casadi]`` and ``matplotlib``. CasADi supplies the model backend
and the variational solver, which uses IPOPT. NumPy prepares observations.

Fit the unknown rate :math:`k` in :math:`\dot{x}=-kx`, with known
:math:`x(0)=2`. Eight noise-free synthetic measurements come from
:math:`x(t)=2\exp(-0.7t)` at distinct positive times. Time is in seconds and
the state is dimensionless. The true rate is used only to generate data and
verify the result, not supplied to the optimiser.

2. Fit and plot
----------------

Define the observations, parameterised ODE, and squared-error loss.
``VariationalProblem`` transcribes the ODE and minimises the sum of squared
observation residuals, with :math:`0.05 \leq k \leq 2` and an initial guess of
``0.3``. Its parameter-only autonomous loss callback evaluates the trajectory
at each observation time.

.. plot::
   :include-source: true

   import matplotlib.pyplot as plt
   import numpy as np

   from coker import VectorSpace
   from coker.dynamics import VariationalProblem, create_autonomous_ode
   from coker.parameters import BoundedVariable

   initial_state = 2.0
   true_rate = 0.7  # inverse seconds; used for synthetic data and checks
   observation_times = np.linspace(0.5, 4.0, 8)
   observations = initial_state * np.exp(-true_rate * observation_times)

   def rate(state, parameters):
       return -parameters[0] * state

   system = create_autonomous_ode(
       x0=np.array([initial_state]),
       xdot=rate,
       parameters=VectorSpace("p", 1),
       backend="casadi",
   )

   def squared_error(trajectory, parameters):
       total = 0.0
       for time, observed in zip(observation_times, observations):
           predicted = trajectory(float(time), parameters)
           total += (predicted - float(observed)) ** 2
       return total

   problem = VariationalProblem(
       system=system,
       loss=squared_error,
       parameters=[
           BoundedVariable(
               "decay_rate",
               lower_bound=0.05,
               upper_bound=2.0,
               guess=0.3,
           ),
       ],
       t_final=4.0,
       backend="casadi",
   )
   fit = problem()
   recovered_rate = float(fit.parameters["decay_rate"])

   assert fit.solve_info is not None and fit.solve_info.success
   assert abs(recovered_rate - true_rate) < 1e-4
   assert fit.cost < 1e-8
   # Validate the recovered parameter by re-simulating the original ODE.
   # This checks predictions independently of the collocation interpolant.
   fitted_parameters = np.array([recovered_rate])

   def predict(times):
       return np.array([
           np.asarray(system(float(time), fitted_parameters)).item()
           for time in times
       ])

   fitted_observations = predict(observation_times)
   np.testing.assert_allclose(
       fitted_observations, observations, rtol=0.0, atol=1e-4
   )

   # Check predictions at times that were not included in the objective.
   held_out_times = np.array([0.25, 1.25, 2.75, 3.75])
   held_out_predictions = predict(held_out_times)
   held_out_truth = initial_state * np.exp(-true_rate * held_out_times)
   np.testing.assert_allclose(
       held_out_predictions, held_out_truth, rtol=0.0, atol=1e-4
   )

   plot_times = np.linspace(0.0, 4.0, 101)
   fitted_states = predict(plot_times)
   fig, ax = plt.subplots()
   ax.scatter(
       observation_times, observations, label="Synthetic observations", zorder=3
   )
   ax.plot(
       plot_times,
       fitted_states,
       label=f"Fitted trajectory: k = {recovered_rate:.4f} / s",
   )
   ax.set(
       xlabel="Time [s]",
       ylabel="State x [dimensionless]",
       title="Estimating an unknown decay rate",
   )
   ax.legend()
   ax.grid(True)
   fig.tight_layout()
   plt.show()

3. Check and adapt
------------------

The legend should report a rate near ``0.7000`` per second. The assertions check:

* successful solver termination;
* parameter recovery within ``1e-4`` absolute error;
* a squared residual sum below ``1e-8`` for these noise-free data;
* predictions within ``1e-4`` state error at measurement and held-out times.

Prediction checks re-simulate the original ODE with the recovered parameter,
rather than evaluating the solver's collocation interpolant. This checks
numerical recovery for this model and data; it does not prove structural
identifiability.

This objective gives equal weight to noise-free scalar observations. For measured
data, replace the observations and choose suitable parameter bounds and initial
guesses. When outputs have different units or uncertainties, divide each
residual by a meaningful measurement scale or estimated standard deviation
before squaring; use covariance whitening if measurement errors are correlated.
Retain held-out observations and choose prediction-error thresholds from
measurement uncertainty, integration error and the accuracy the task requires,
not the noise-free tolerances above.

.. Author notes

   - Audience and prerequisites: describe the required simulation, least-squares, and measurement-data knowledge.
   - Concrete model: select a dynamical model with named unknown parameters, known inputs and initial conditions, observation times, and a documented data source or reproducible synthetic-data procedure.
   - Steps: outline defining observations, constructing predictions and residuals, choosing the fitting objective and parameter bounds, solving, and evaluating the fitted model.
   - Expected result: specify reference fitted parameters, residual statistics, and a plot comparing measurements with model predictions, including noise assumptions.
   - Backend and dependencies: identify the simulation and optimisation backends, required extras, and plotting or data-loading dependencies.
   - Verification: define parameter or prediction tolerances and a held-out or known-parameter check; explain that a successful numerical fit does not establish structural identifiability and link the separate system-analysis guide where appropriate.
