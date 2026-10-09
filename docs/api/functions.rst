Functions
=========

The :func:`coker.function` factory is the primary entry point for compiling
Python callables into Coker functions.

.. autofunction:: coker.function

.. autoclass:: coker.algebra.function.Function
   :members:
   :special-members: __call__

Tensor helpers
--------------

.. autofunction:: coker.zeros

.. autoclass:: coker.SymbolicVector

Sparse matrices
---------------

.. autoclass:: coker.SparseMatrixBuilder
   :members:

.. autoclass:: coker.SparseMatrixPattern

The numeric path retains CSC storage:

.. doctest:: residual_dae

   >>> import numpy as np
   >>> from coker import SparseMatrixBuilder
   >>> builder = SparseMatrixBuilder(np.array([[True, False], [False, True]]))
   >>> builder.matrix(np.array([2.0, 3.0])).toarray().tolist()
   [[2.0, 0.0], [0.0, 3.0]]
