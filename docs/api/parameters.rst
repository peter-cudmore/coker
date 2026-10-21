Parameter declarations
======================

Import scalar and array decisions from ``coker.parameters``. Callable
declarations and fitted results belong to
``coker.parameters.function_parameters``. These declarations describe solver
decisions; they are not themselves fitted numeric values. See
:doc:`optimisation` for ordinary programs and :doc:`dynamics` for dynamics
parameter ordering and variational results.

.. Author notes

   - Audience: model authors selecting finite scalar, array and callable
     decision declarations, and inspecting reconstructed fitted values.
   - State target shapes, concrete block order, validation and backend limits;
     keep task walkthroughs in the cookbook and custom declaration hooks here.

Scalar and array decisions
--------------------------

.. autoclass:: coker.parameters.BoundedVariable
   :members: degrees_of_freedom

Declares one scalar with ``name``, ``lower_bound``, ``upper_bound`` and
``guess=0``. Its decision count is one. The dataclass does not validate bounds
or the guess at construction; validation depends on the consuming builder.

.. autoclass:: coker.parameters.UnboundedVariable
   :members: degrees_of_freedom, lower_bound, upper_bound

Declares one scalar with ``name`` and ``guess=0``. Its bounds are negative and
positive infinity. Construction does not check that the guess is finite.

.. autoclass:: coker.parameters.BoundVector
   :members: shape, size, degrees_of_freedom

Converts the guess and component-wise bounds to floating-point NumPy arrays.
All three must have the same one-dimensional shape; the guess must be finite,
bounds must be ordered, and the guess must lie within them. Invalid values
raise ``ValueError``. ``shape`` is the guess shape and ``size`` is the scalar
decision count.

.. autoclass:: coker.parameters.DenseTensorVariable
   :members: shape, size, degrees_of_freedom

Converts ``guess`` to a floating-point NumPy array. A scalar or non-finite guess
raises ``ValueError``. The array may have any positive rank; its shape is
retained during reconstruction, and its size is the scalar decision count.
There are no explicit bounds.

Callable declarations
---------------------

A callable declaration supplies finite scalar or dense decision blocks for a
:class:`~coker.FunctionSpace`. A builder validates that space and inserts the
blocks into its optimisation graph. Set a non-empty ``name`` when the consuming
builder requires named decisions, including ``ProblemBuilder``. An optional
name does not itself guarantee that the builder accepts an unnamed declaration.

The scalar families below require exactly one scalar input and one scalar
output. Passing a non-``FunctionSpace`` target raises ``TypeError``; a target
with incompatible input or output dimensions raises ``ValueError``. Numeric
calls and fitted value types depend on the backend that materialises the
declaration. Declaring a function does not imply that every backend can solve
the containing optimisation problem; see :doc:`backends`.

Closures and bound inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: coker.parameters.function_parameters.ClosureParameter
   :members: bind_callable

``ClosureParameter(function, parameters)`` binds a non-empty sequence of scalar
or dense declarations to a Coker ``Function``. The source function's inputs must
be the concrete blocks, in declaration order, followed by the public target
input; its output dimensions must match the target. Use one public input:
``evaluate(parameters, argument)`` and the inherited function builder pass a
single public argument. A non-Coker function or unsupported declaration raises
``TypeError``; an empty parameter sequence raises ``ValueError``. The public
name comes from ``function.name``.

``bind_callable(implementation, parameters, inputs, *, name=None,
backend="numpy")`` traces the source function and returns a
``ClosureParameter``. Its implementation receives concrete blocks first, then
the public input. The default name is ``"bound_" + implementation.__name__``.

.. autoclass:: coker.parameters.function_parameters.BoundFunctionParameter
   :members: validate_target, list_concrete_parameters, evaluate, build_function

Use this declaration to bind selected source input positions rather than a
leading block. ``bindings`` is a mapping or sequence of
``(input_index, declaration)`` pairs, normalised into source-input order.
Unbound inputs remain public in their original order, and output dimensions
remain unchanged. Each bound declaration must match its source input's shape.
Invalid types raise ``TypeError``; empty bindings, repeated or out-of-range
indices and incompatible shapes raise ``ValueError``. ``name`` defaults to the
source function's name.

For direct ``evaluate(parameters, argument)`` calls, pass a single public value
when one input remains, a sequence of public values for multiple inputs, or
``None`` for a fully bound function. ``build_function`` instead exposes the
public inputs as separate arguments followed by concrete blocks. Reconstruction
and calls of fitted multi-input or fully bound functions also depend on the
backend adapter; the declaration's shape validation alone does not establish
backend support.

Bind a slope input while retaining time as the public argument. This example
uses only the base NumPy dependency; solving and fitted-callable inspection are
covered in :doc:`optimisation`.

.. testcode:: bound_parameter_contract

   from coker import FunctionSpace, Scalar, function
   from coker.parameters import UnboundedVariable
   from coker.parameters.function_parameters import BoundFunctionParameter

   affine = function(
       [Scalar("time"), Scalar("slope")],
       lambda time, slope: slope * time,
   )
   declaration = BoundFunctionParameter(
       affine, {1: UnboundedVariable("slope", 2.)}, name="response",
   )
   target = FunctionSpace("response", [Scalar("time")], [Scalar("value")])
   declaration.validate_target(target)
   assert declaration.evaluate((3.,), 2.) == 6.

Dense layers
~~~~~~~~~~~~

.. autoclass:: coker.parameters.function_parameters.DenseLayer

``DenseLayer(input_size, activation, name=None)`` represents
``activation(weight @ argument + bias)`` for a vector target, or
``activation(weight * argument + bias)`` for a scalar target.
``activation`` must be a Coker ``Function`` or ``BoundCallable``.

For a vector layer, the activation has one one-dimensional vector input and
one one-dimensional vector output, both of positive width. The target input
width is ``input_size`` and its output width matches the activation output.
The concrete blocks, in order, are a weight matrix of shape
``(activation_input_width, input_size)`` initialised with ``numpy.eye``, and a
zero bias vector of length ``activation_input_width``. ``hidden_size`` and
``output_size`` report the activation input and output widths for vector
activations.

For a scalar activation, ``input_size`` must be one and the target must be
scalar-to-scalar. Its concrete blocks are scalar weight and bias, with guesses
``1.0`` and ``0.0``. Invalid activation types raise ``TypeError``; a non-positive
or non-integral ``input_size`` (including ``bool``), incompatible activation
signature or incompatible target raises ``ValueError``. All decisions are
unbounded.

Monotone functions
~~~~~~~~~~~~~~~~~~

.. autoclass:: coker.parameters.function_parameters.MonotonePiecewiseLinear

``MonotonePiecewiseLinear(domain_knots, lower_bound, upper_bound, guess=None,
name=None)`` defines a bounded, increasing scalar piecewise-linear function.
It interpolates between fitted knot values and holds the endpoint values
constant outside the knot range. Knots are used in the supplied coordinates;
no input normalisation is applied.

The concrete block is one dense vector, with one unconstrained coefficient per
knot. ``guess`` supplies these coefficients, not output knot values, and defaults
to zeros. An exponential cumulative parameterisation places the knot values
between the bounds in increasing order. This is a monotonicity and output-bound
parameterisation, not a component-wise bound on its coefficients.

Knots must be a finite, strictly increasing vector with at least two entries.
Bounds must be finite real scalars with ``lower_bound < upper_bound``. The guess
must be a finite vector of knot count. Non-numeric sequences or non-real bounds
raise ``TypeError``; invalid shapes, ordering, finiteness or guess length raise
``ValueError``.

Radial basis functions
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: coker.parameters.function_parameters.RadialBasisFunction

``RadialBasisFunction(centers, width, lower_bound=None, upper_bound=None,
guess=None, name=None)`` represents the scalar Gaussian expansion

.. math::

   f(x) = c_n + \sum_{i=0}^{n-1} c_i
          \exp\left(-\tfrac12 ((x - \mu_i) / w)^2\right).

``centers`` fixes the ``n`` centres and ``width`` fixes the positive common
width ``w``. One vector block contains the ``n`` coefficients followed by the
constant offset. Its guess has length ``n + 1`` and defaults to zeros.
Omitting both bounds creates an unbounded dense block; specifying both creates
a ``BoundVector`` with those bounds on every coefficient, including the offset.
These are coefficient bounds, not bounds on the function's output.

Centres must form a non-empty finite vector. Width must be a finite real scalar
and positive. Guesses must be finite and have the correct length. Bounds must
both be supplied or both omitted; when supplied they must be ordered real
scalars, may be infinite, and must contain the guess. Non-numeric sequences,
non-real bounds and a non-finite or non-real width raise ``TypeError``; invalid
shapes, a non-positive width, inconsistent bounds or an invalid guess raise
``ValueError``.

Fitted results
--------------

.. autoclass:: coker.parameters.function_parameters.FittedFunction
   :members: input_shape, output_shape
   :special-members: __call__

Solvers reconstruct callable decisions as ``FittedFunction`` values. Ordinary
programs expose them through ``program.parameters[name]``; variational results
expose fitted declarations through their parameter results. The wrapper has
four public attributes:

``specification``
   The original ``FunctionParameter`` declaration.
``space``
   The validated public ``FunctionSpace``.
``function``
   The concrete backend callable, not necessarily a Coker ``Function``.
``parameters``
   The reconstructed concrete blocks in declaration order, retaining scalar
   or dense shapes. Solver-produced values use a tuple of blocks; their numeric
   types are backend-specific.

``fitted(*arguments, **kwargs)`` forwards arguments, keywords, return values and
errors directly to ``fitted.function``. It does not flatten inputs, convert
outputs to NumPy, guarantee a scalar return type, or add shape validation.
``input_shape()`` and ``output_shape()`` return tuples of dimensions from
``space``, excluding private concrete blocks. A fitted function is callable,
but does not itself provide ``Function.lower()`` or other compilation methods.

.. _custom-parameter-declarations:

Custom declarations
-------------------

.. autoclass:: coker.parameters.function_parameters.FunctionParameter
   :members: validate_target, list_concrete_parameters, evaluate, build_function

Implement ``validate_target(target)`` to return a compatible ``FunctionSpace``
or raise an error, ``list_concrete_parameters()`` to return an ordered tuple of
named scalar or dense declarations, and ``evaluate(parameters, argument)`` to
return the declared output from reconstructed blocks and a public argument.
The default ``build_function(target, backend)`` returns a Coker ``Function``
whose inputs are the target argument followed by concrete blocks; it evaluates
only the first public argument. Override it for a declaration requiring another
argument-passing convention, as ``BoundFunctionParameter`` does.

.. autoclass:: coker.parameters.ParameterMixin
   :members: degrees_of_freedom

Scalar and dense declarations implement this separate decision-count interface.
Callable declarations use ``FunctionParameter`` rather than ``ParameterMixin``.
