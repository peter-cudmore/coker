1. Creating and evaluating a function
=====================================

Build a function with one scalar input and two scalar outputs using the NumPy
backend:

.. math::

   a(x) = 3(x + 2) - x/2, \qquad b(x) = \sin(x) + \exp(-x).

Use ``numpy.sin`` and ``numpy.exp`` rather than Python's ``math`` functions:
Coker records these supported NumPy operations during tracing.

Declare, trace, then evaluate
-----------------------------

Write the implementation as a Python function, then pass it to ``function``
with an ordered list of input spaces. ``Scalar("x")`` declares a scalar
argument named ``x`` without assigning a numerical value.

Coker calls the implementation with a symbolic argument and records its
operations. The returned function evaluates that computation with numerical
inputs. ``backend="numpy"`` selects the execution backend.

.. testcode:: function_evaluation

   import numpy as np
   from coker import Scalar, function

   def evaluate_expressions(x):
       arithmetic = 3.0 * (x + 2.0) - x / 2.0
       nonlinear = np.sin(x) + np.exp(-x)
       return arithmetic, nonlinear

   expressions = function(
       arguments=[Scalar("x")],
       implementation=evaluate_expressions,
       backend="numpy",
   )

   arithmetic, nonlinear = expressions(2.0)
   np.testing.assert_allclose(arithmetic, 11.0)
   np.testing.assert_allclose(nonlinear, 1.0446327100622944, rtol=1e-12)
   assert np.asarray(arithmetic).shape == ()
   assert np.asarray(nonlinear).shape == ()

   at_zero = expressions(0.0)
   np.testing.assert_allclose(at_zero, (6.0, 1.0), rtol=1e-12)
   print(f"Arithmetic at x=2: {arithmetic:.1f}")
   print(f"Nonlinear at x=2: {nonlinear:.6f}")
   print("At x=0:", tuple(float(value) for value in at_zero))

.. testoutput:: function_evaluation

   Arithmetic at x=2: 11.0
   Nonlinear at x=2: 1.044633
   At x=0: (6.0, 1.0)

Inspect the result
------------------

Scalar outputs have shape ``()``, not ``(1,)``. The implementation returns a
tuple, so calls return a tuple in the same order. A single returned expression
would instead produce one output directly.

Call ``expressions`` with new numerical inputs to reuse the recorded
computation without tracing again.

Next, :doc:`parameterized_matrix` introduces vector inputs and a matrix whose
entries are supplied at evaluation time.
