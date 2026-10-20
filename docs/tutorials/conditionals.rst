4. Choosing between two expressions
====================================

After :doc:`cumulative_products`, you have seen a Python loop run while Coker
traces an implementation. Use ``coker.if_then_else`` for a choice that depends
on the input at evaluation time. This lesson uses the NumPy backend and needs
only the base installation.

We will evaluate this piecewise scalar function:

.. math::

   f(x) =
   \begin{cases}
     x^2, & x < 0, \\
     x + 1, & x \geq 0.
   \end{cases}

Use a symbolic conditional
---------------------------

Call ``if_then_else(condition, true_value, false_value)`` with a scalar
comparison and two branch expressions of the same shape. Here, the condition
is ``x < 0.0`` and both branches are scalars.

Native Python ``if`` and conditional expressions such as
``x * x if x < 0 else x + 1`` do not record a symbolic choice. They cannot
select a branch from the input at each later evaluation.

.. testcode:: conditionals

   import numpy as np
   from coker import Scalar, function, if_then_else

   def piecewise_expression(x):
       return if_then_else(x < 0.0, x * x, x + 1.0)

   piecewise = function(
       arguments=[Scalar("x")],
       implementation=piecewise_expression,
       backend="numpy",
   )

   samples = (-2.0, 0.0, 3.0)
   expected = (4.0, 1.0, 4.0)
   for value, reference in zip(samples, expected):
       result = piecewise(value)
       assert np.asarray(result).shape == ()
       np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)
       print(f"f({value:.1f}) = {float(result):.1f}")

.. testoutput:: conditionals

   f(-2.0) = 4.0
   f(0.0) = 1.0
   f(3.0) = 4.0

Check both sides and the boundary
----------------------------------

Negative inputs select ``x * x``; positive inputs select ``x + 1``. At zero,
``x < 0`` is false, so the result is ``1``, not ``0``. Check the boundary to
confirm that the comparison matches the piecewise definition.

``if_then_else`` does not provide Python's short-circuit execution. See
:doc:`../basics/sharp_edges` for branch-evaluation limits, a guarded reciprocal,
and the distinction between symbolic choices and fixed Python configuration.

Next, :doc:`backends` keeps a single implementation and selects different
execution backends, then obtains backend-native functions.
