Backend configuration
=====================

CasADi and NumPy execution and solver configuration classes are listed below.
CasADi supports optimisation and large-scale numerical work. NumPy provides
direct host-side evaluation and a small implicit residual integrator.

See :doc:`../backends` for the capability matrix and backend-selection rules,
:doc:`../tutorials/backends` for a worked comparison, and
:doc:`../how_to/backend_interoperability` for native import and export recipes.

.. autoclass:: coker.backends.casadi.CasadiBackend

.. autoclass:: coker.backends.casadi.CasadiResidualSolver

.. autoclass:: coker.backends.casadi.CasadiResidualSolverOptions

.. autoclass:: coker.backends.casadi.CasadiVariationalOptions

.. autoclass:: coker.backends.numpy.NumpyBackend

.. autoclass:: coker.backends.numpy.NumpySolverParameters

