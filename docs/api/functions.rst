Functions
=========

The :func:`coker.function` factory is the primary entry point for compiling
Python callables into Coker functions.
See :doc:`../tutorials/function_evaluation` for a step-by-step introduction.

Call and shape contracts
------------------------

Arguments follow declaration order and calls require exactly the declared
number of arguments. An arity mismatch raises :class:`ValueError`; argument
slots are not optional merely because their declaration is ``None``.
Function-space arguments accept Coker functions or traceable Python
callables. A non-callable implementation passed to :func:`coker.function`
raises :class:`TypeError`.

The factory preserves the implementation's output structure. A bare result
produces a bare return value; a list or tuple produces a tuple of output
values, including ``None`` slots. The shape methods always return tuples
in declaration order. ``arguments`` returns a copy of the input names,
and ``signature`` contains named input and output metadata.

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

``lower(options=None)`` returns a cached backend-specific executable handle.
It accepts only :class:`~coker.backends.lowered.LoweringOptions` or ``None``;
other types raise :class:`TypeError`. An explicit options backend overrides
the function's backend, except that a native callable cannot be lowered for
a different backend. That mismatch raises :class:`RuntimeError`.

``call_inline(*args)`` explicitly uses the NumPy tracing interpreter.
Use it for symbolic higher-order evaluation when the computation should be
expanded; ordinary traced calls leave composition to the enclosing backend.

Inequalities
------------

The comparison operators construct bound expressions; they do not evaluate
the function or return a Boolean. Each requires exactly one numeric output
and a bound with the same shape. There is no broadcasting. Output-count and
shape mismatches raise :class:`ValueError`.

The resulting expression stores the function and its lower and upper bounds.
Strictness is not retained as separate metadata: ``<`` and ``<=`` describe
the same upper-bound interval, and ``>`` and ``>=`` the same lower-bound
interval.

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
