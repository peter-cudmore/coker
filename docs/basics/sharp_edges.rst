Sharp edges when tracing Python
================================

Coker records supported operations by calling your implementation with symbolic
inputs when you construct a function. It does not compile arbitrary Python
control flow.

Input-dependent ``if-else``
----------------------------

A native Python ``if``, ``elif``, or conditional expression cannot select a
branch from a symbolic condition. It chooses a path during tracing rather than
recording branches for later numerical evaluation. Do not write:

.. code-block:: python

   def choose_value(x):
       if x < 0:
           return -x
       else:
           return x

Use ``if_then_else(condition, true_value, false_value)`` for symbolic selection.
The condition must be a scalar comparison, such as ``x < 0``, ``x <= 0``, or
``x == 0``, and both branches must have the same shape. A raw symbolic number
is not a condition: use ``x == 0`` rather than passing ``x`` itself.

.. testcode:: sharp_edges_selection

   import numpy as np
   from coker import Scalar, function, if_then_else

   magnitude = function(
       [Scalar("x")],
       lambda x: if_then_else(x < 0.0, -x, x),
       backend="numpy",
   )
   for value, expected in ((-2.0, 2.0), (0.0, 0.0), (3.0, 3.0)):
       np.testing.assert_allclose(magnitude(value), expected)
   print("Magnitude at -2, 0, 3:", [float(magnitude(x)) for x in (-2, 0, 3)])

.. testoutput:: sharp_edges_selection

   Magnitude at -2, 0, 3: [2.0, 0.0, 3.0]

Python's ``and``, ``or``, ``not``, and chained comparisons such as
``0 < x < 1`` also request truth values; they are not symbolic Boolean
combinators. To express an ``elif``-style choice, nest ``if_then_else`` calls
with individual scalar comparisons. See :doc:`../tutorials/conditionals` for
the introductory walkthrough and :doc:`../api/conditional` for the API contract.

Selection is not short-circuit evaluation
------------------------------------------

The branch arguments are expressions, not callbacks. Python constructs both
before calling ``if_then_else``. With a symbolic input, ``1 / x`` records a
division rather than numerically dividing by zero.

An inactive branch may still be evaluated during numerical execution. In the
NumPy backend, division by zero produces ``NaN``, but selection can return a
valid value from the other branch. If the division itself is the output, its
result remains ``NaN``. Other backends may handle inactive divisions differently.

.. testcode:: sharp_edges_guard

   import numpy as np
   from coker import Scalar, function, if_then_else

   reciprocal = function(
       [Scalar("x")],
       lambda x: if_then_else(x == 0.0, 0.0, 1.0 / x),
       backend="numpy",
   )
   values = [float(reciprocal(x)) for x in (-2.0, 0.0, 2.0)]
   np.testing.assert_allclose(values, [-0.5, 0.0, 0.5])
   print("Reciprocal with zero mapped to zero:", values)

.. testoutput:: sharp_edges_guard

   Reciprocal with zero mapped to zero: [-0.5, 0.0, 0.5]

This function is discontinuous at zero and has no derivative there, despite its
defined value of zero. A valid selected value does not imply a defined gradient.

With an ordinary Python zero, ``1 / x`` raises ``ZeroDivisionError`` before
``if_then_else`` is called. Symbolic selection does not protect arbitrary branch
operations from exceptions or make Python side effects conditional. Check the
backend's behaviour for other domain errors.

Tracing-time configuration and fixed shapes
--------------------------------------------

A Python ``if`` based on an ordinary configuration value can choose which
operations are recorded. That choice is fixed for the constructed function.
Likewise, a ``for`` loop over ``range(n)`` runs during tracing when ``n`` is an
ordinary integer; its iteration count cannot depend on symbolic inputs.
Construct another function to change configuration or input/output shapes.
See :doc:`../tutorials/cumulative_products`.

Keep implementations free of side effects that you expect to happen on every
numerical evaluation: printing, mutating external state, and drawing random
constants during tracing do not become recorded mathematical operations.
Use supported NumPy operations such as ``np.sin`` rather than ``math.sin`` on
symbolic inputs, as shown in :doc:`../tutorials/function_evaluation`.

Backend selection does not remove these tracing constraints. Available
operations and solvers also vary; check :doc:`../tutorials/backends` and the
:doc:`backend capability guide <../backends>` before switching backends.
