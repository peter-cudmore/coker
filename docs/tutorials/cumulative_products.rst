3. Building a vector with a loop
=================================

Build a vector one entry at a time with a Python ``for`` loop and
``SymbolicVector``. For an input ``p`` of length ``n``, compute the cumulative
products:

.. math::

   s_i = \prod_{j=0}^{i} p_j, \qquad i = 0, \ldots, n-1.

Accumulate, then assign
-----------------------

An ordinary floating-point NumPy array is not an output buffer for symbolic
expressions. Instead, create ``SymbolicVector((n,))`` inside the implementation.
It can hold symbolic entries as well as constants while you fill it.

Start the running product at ``1.0``. Multiply by each input in turn and assign
the result to an output entry. Return the builder itself: Coker collapses it
into a single vector expression when tracing the return value.

.. testcode:: cumulative_products

   import numpy as np
   from coker import SymbolicVector, VectorSpace, function

   n = 4

   def build_cumulative_products(p):
       sequence = SymbolicVector((n,))
       product = 1.0
       for index in range(n):
           product = product * p[index]
           sequence[index] = product
       return sequence

   cumulative_products = function(
       arguments=[VectorSpace("p", n)],
       implementation=build_cumulative_products,
       backend="numpy",
   )

   p = np.array([2.0, 3.0, -1.0, 0.5])
   result = cumulative_products(p)
   assert result.shape == (n,)
   np.testing.assert_allclose(result, [2.0, 6.0, -6.0, -3.0])
   np.testing.assert_allclose(result, np.cumprod(p))

   with_zero = cumulative_products(np.array([2.0, 0.0, 7.0, 8.0]))
   assert with_zero.shape == (n,)
   np.testing.assert_allclose(with_zero, [2.0, 0.0, 0.0, 0.0])
   print("Cumulative products:", result.tolist())
   print("After a zero:", with_zero.tolist())

.. testoutput:: cumulative_products

   Cumulative products: [2.0, 6.0, -6.0, -3.0]
   After a zero: [2.0, 0.0, 0.0, 0.0]

Understand when the loop runs
-----------------------------

The Python loop runs during tracing. Its bound ``n`` is an ordinary integer,
so Coker records a fixed sequence of multiplications and assignments.
Evaluation reuses that computation for new vectors of the declared length.
The iteration count cannot depend on a symbolic input; construct a new
function with a different ``n`` to change the vector length.

Next, :doc:`conditionals` shows how to make a result depend on an input value
without using a native Python ``if`` on a symbolic condition.
