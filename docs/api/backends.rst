Backend configuration
=====================

Choose a backend when tracing a function or constructing a dynamics system.
The CasADi backend is the preferred backend for optimisation and large-scale
numerical work. The NumPy backend provides direct host-side evaluation and a
small implicit residual integrator.

.. autoclass:: coker.backends.casadi.CasadiBackend

.. autoclass:: coker.backends.casadi.CasadiResidualSolver

.. autoclass:: coker.backends.casadi.CasadiResidualSolverOptions

.. autoclass:: coker.backends.casadi.CasadiVariationalOptions

.. autoclass:: coker.backends.numpy.NumpyBackend

.. autoclass:: coker.backends.numpy.NumpySolverParameters

