Argument Spaces
===============

Argument spaces describe the domain of a :func:`coker.function`. Every
argument passed to ``function()`` must be one of the types below.

``Scalar`` and ``VectorSpace`` are named argument domains. ``Dimension`` is
unnamed shape metadata: ``Dimension(None)`` means scalar, whereas
``Dimension((1,))`` is a one-component vector. The historical scalar ``shape``
property is ``(1,)`` even when a backend returns a Python scalar; inspect
``is_scalar()`` to distinguish them. ``VectorSpace(name, (r, c))`` describes one
matrix argument, not ``r`` separate vector arguments.

``FunctionSpace(name, arguments, output, signature=None)`` describes one callable
argument with ordered domain/codomain declarations. ``signature`` is optional
smoothness metadata, not a backend ``FunctionSignature``; it does not implement
automatic differentiation by itself.

.. doctest::

   >>> from coker import FunctionSpace, Scalar, function
   >>> space = FunctionSpace("law", [Scalar("t")], [Scalar("value")])
   >>> evaluate = function([space, Scalar("t")], lambda law, t: law(t), backend="numpy")
   >>> assert evaluate(lambda t: 2 * t, 3.) == 6.

.. autoclass:: coker.Scalar

.. autoclass:: coker.VectorSpace

.. autoclass:: coker.FunctionSpace
   :members: input_dimensions, output_dimensions, matches_signature

.. autoclass:: coker.Dimension
   :members: flat, to_space, shape, scalar, is_scalar, is_vector, is_matrix
