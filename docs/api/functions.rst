Functions
=========

The :func:`coker.function` factory is the primary entry point for compiling
Python callables into Coker functions.
See :doc:`../tutorials/function_evaluation` for a step-by-step introduction.

Call and shape contracts
------------------------

Calls are positional in declaration order, including explicitly absent slots.
Output structure follows the implementation: a bare expression returns a bare
value, while a list or tuple returns an ordered tuple, including ``None``
slots. Shape and signature metadata describe those ordered inputs and outputs:

.. testcode:: function_metadata

   import numpy as np
   from coker import Scalar, function

   pair = function([Scalar("x")], lambda x: (x, x + 1), name="pair")
   assert pair.arguments == ["x"]
   assert len(pair.input_shape()) == 1
   assert len(pair.output_shape()) == 2
   np.testing.assert_allclose(pair(2.0), (2.0, 3.0))

Lowering and inline calls
-------------------------

Use :meth:`~coker.Function.lower` for a backend execution handle; see
:ref:`backend-lowering-contract` for backend selection and the execution ABI.
Use :meth:`~coker.Function.call_inline` when symbolic higher-order evaluation
should expand through the NumPy tracing interpreter rather than leave a
composed call to the enclosing backend.

Inequalities
------------

Function comparisons create elementwise bound expressions, not Boolean
predicates. They require matching output and bound shapes without broadcasting.
Strict and non-strict operators retain the same bound interval. These
whole-function bounds are distinct from the symbolic expression comparisons
used inside :doc:`optimisation` builders.

.. testcode:: function_bounds

   import numpy as np
   from coker import VectorSpace, function

   identity = function([VectorSpace("x", 2)], lambda x: x)
   lower_bound = identity > np.zeros(2)
   assert lower_bound.value is identity
   np.testing.assert_array_equal(lower_bound.lower, np.zeros(2))
   assert np.isposinf(lower_bound.upper).all()

Reference
---------

.. autofunction:: coker.function

.. autoclass:: coker.Function
   :members:
   :special-members: __call__, __le__, __ge__, __lt__, __gt__

.. autoclass:: coker.algebra.function.BoundCallable
   :members:
   :special-members: __call__

.. autoclass:: coker.algebra.function.InequalityExpression

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
