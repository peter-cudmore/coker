2. Reshaping parameters into a matrix
=====================================

Reshape a vector of runtime parameters into a matrix, then multiply it by a
vector:

.. math::

   f(x, p) = A(p)x,
   \qquad
   A(p) = \operatorname{reshape}(p, (n, n)).

Here ``x`` is a length-``n`` vector and ``p`` is a length-``n * n`` vector.
Both are ordinary function inputs; the parameter name is a convention.

Declare the vector shapes
-------------------------

Set ``n = 2`` before tracing. ``VectorSpace("x", n)`` declares a vector, while
``VectorSpace("p", n * n)`` declares the flattened matrix entries. Shapes are
fixed when the function is constructed; they are not extra numerical inputs.

Inside the implementation, ``np.reshape`` turns the symbolic parameter vector
into a square matrix. The default order is row-major: ``[2, -1, 1, 3]`` becomes
``[[2, -1], [1, 3]]``. Use ``@`` for matrix-vector multiplication; ``*`` would
mean elementwise multiplication instead.

.. testcode:: parameterized_matrix

   import numpy as np
   from coker import VectorSpace, function

   n = 2

   def multiply_by_parameters(x, p):
       matrix = np.reshape(p, shape=(n, n))
       return matrix @ x

   model = function(
       arguments=[VectorSpace("x", n), VectorSpace("p", n * n)],
       implementation=multiply_by_parameters,
       backend="numpy",
   )

   x = np.array([1.0, 2.0])
   p = np.array([2.0, -1.0, 1.0, 3.0])
   result = model(x, p)
   assert result.shape == (n,)
   np.testing.assert_allclose(result, [0.0, 7.0], rtol=1e-12, atol=1e-12)

   identity_parameters = np.array([1.0, 0.0, 0.0, 1.0])
   unchanged = model(x, identity_parameters)
   assert unchanged.shape == (n,)
   np.testing.assert_allclose(unchanged, x, rtol=1e-12, atol=1e-12)
   print("First matrix:", result.tolist())
   print("Identity matrix:", unchanged.tolist())

.. testoutput:: parameterized_matrix

   First matrix: [0.0, 7.0]
   Identity matrix: [1.0, 2.0]

Change the matrix
-----------------

Pass new entries in ``p`` to change the matrix without tracing again.
The identity-matrix call returns ``x`` unchanged. Supply ``x`` and ``p`` in
the order declared by ``arguments``.

Next, :doc:`cumulative_products` constructs a vector one entry at a time using
a Python loop.
