5. Choosing a backend
=====================

.. Author notes

   - Assume the reader has completed the four preceding lessons and can construct
     and call a Coker function. Keep this as the fifth step, not a registry tutorial.
   - Compare ordinary calls first. Keep native export optional and distinguish it
     from direct execution before introducing backend-specific extraction.
   - Acknowledge all five bundled backends; execute NumPy, CasADi, SymPy and
     PyTorch here because the documentation environment installs those dependencies.

Evaluate the same implementation with NumPy, CasADi, SymPy and PyTorch using
ordinary Coker calls. Then optionally extract native functions for integration
with other libraries. This final lesson builds on :doc:`conditionals`.

Install the optional dependencies for CasADi and PyTorch:

.. code-block:: console

   pip install "coker[casadi,pytorch]"

NumPy and SymPy are included in the base installation. The PyTorch examples
use CPU tensors and do not require CUDA.

JAX is another bundled backend. It requires Python 3.11 or newer and
``pip install "coker[jax]"``. This lesson does not include a JAX execution or
native-extraction walkthrough; see :doc:`../backends` for its capabilities.

Choose a backend
----------------

Choose a backend for the environment in which you want to use the function:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Backend name
     - Choose it for
     - Native interface used below
   * - ``numpy``
     - Familiar numerical evaluation and reference results.
     - Call the Coker function directly.
   * - ``casadi``
     - Symbolic numerical functions, derivatives, and optimisation workflows.
     - A ``casadi.Function``.
   * - ``sympy``
     - Inspecting and manipulating symbolic expressions.
     - Symbolic arguments and an expression, wrapped in ``sympy.Lambda``.
   * - ``pytorch``
     - Tensor execution and PyTorch autograd.
     - An eager ``torch.nn.Module``.
   * - ``jax``
     - Array evaluation with JAX-native values.
     - Not covered in this walkthrough; see :doc:`../backends`.

Evaluate one implementation four ways
-------------------------------------

Define ``f(x) = x*x + 2*x + 1`` and evaluate it at ``x = 3`` with each backend.
Change the ``backend`` argument to ``function``; keep the implementation and
input declaration the same.

.. testcode:: backend_evaluation

   import numpy as np
   from coker import Scalar, function

   def evaluate_polynomial(x):
       return x * x + 2 * x + 1

   models = {}
   for backend_name in ("numpy", "casadi", "sympy", "pytorch"):
       model = function(
           arguments=[Scalar("x")],
           implementation=evaluate_polynomial,
           backend=backend_name,
       )
       models[backend_name] = model
       result = model(3.0)
       # Convert only to compare and display these numerical sample results.
       numeric = np.asarray(result, dtype=float)
       assert numeric.size == 1
       np.testing.assert_allclose(numeric, 16.0, rtol=1e-12, atol=1e-12)
       print(f"{backend_name}: {float(numeric.reshape(())):.1f}")

.. testoutput:: backend_evaluation

   numpy: 16.0
   casadi: 16.0
   sympy: 16.0
   pytorch: 16.0

Each backend accepts the ordinary scalar input ``3.0``; no manual input
conversion or backend lookup is needed. The NumPy conversion above is only
for the assertions and display, not for execution. A CasADi scalar can be a
``1 x 1`` matrix, while a PyTorch scalar tensor has shape ``()``. The
assertion checks for one value rather than a shared native shape.

Optional native export
----------------------

Direct calls such as ``model(3.0)`` are sufficient for Coker execution on
every backend above. Export a native object only when another library needs
its own interface. The following subsections reuse the models already built.
NumPy needs no separate native extraction.

Get a CasADi function
~~~~~~~~~~~~~~~~~~~~~

Call ``lower()`` to obtain an execution handle, then access its
``ca_function`` attribute:

.. testcode:: backend_evaluation

   import casadi as ca

   casadi_handle = models["casadi"].lower()
   native_casadi = casadi_handle.ca_function
   assert isinstance(native_casadi, ca.Function)
   casadi_result = native_casadi(3.0)
   assert casadi_result.shape == (1, 1)
   np.testing.assert_allclose(np.asarray(casadi_result), [[16.0]])
   print(f"Native CasADi: {float(casadi_result):.1f}")

.. testoutput:: backend_evaluation

   Native CasADi: 16.0

Pass ``native_casadi`` directly to CasADi APIs. Native extraction requires a
supported graph: a fallback execution handle can have ``ca_function is None``.
The polynomial above supports native extraction.

Get a SymPy function
~~~~~~~~~~~~~~~~~~~~

Call the SymPy backend's ``lower_to_symbolic`` method to obtain symbolic
arguments and a formula, then wrap them in ``sympy.Lambda``. Unlike ``lower()``,
this returns symbolic objects rather than an execution handle. For a
single-output model, the output is the expression itself, not a list.

.. testcode:: backend_evaluation

   import sympy as sp
   from coker.backends import get_backend_by_name

   sympy_backend = get_backend_by_name("sympy")
   arguments, expression = sympy_backend.lower_to_symbolic(models["sympy"])
   (symbolic_x,) = arguments
   assert sp.simplify(expression - (symbolic_x + 1) ** 2) == 0
   native_sympy = sp.Lambda(tuple(arguments), expression)
   assert isinstance(native_sympy, sp.Lambda)
   assert sp.simplify(native_sympy(3) - 16) == 0
   print(f"Native SymPy: {float(native_sympy(3)):.1f}")

.. testoutput:: backend_evaluation

   Native SymPy: 16.0

``native_sympy`` accepts symbolic or numerical arguments. To generate a
numerical Python callable instead, use
``sp.lambdify(arguments, expression, modules="numpy")`` with the same formula.

Get a PyTorch module
~~~~~~~~~~~~~~~~~~~~

Call the PyTorch backend's ``as_module`` method and pass a tensor to the
returned module. Its operations remain connected to autograd:

.. testcode:: backend_evaluation

   import torch

   pytorch_backend = get_backend_by_name("pytorch")
   native_pytorch = pytorch_backend.as_module(models["pytorch"])
   assert isinstance(native_pytorch, torch.nn.Module)
   tensor_x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
   tensor_result = native_pytorch(tensor_x)
   assert tensor_result.shape == torch.Size([])
   torch.testing.assert_close(tensor_result, torch.tensor(16.0, dtype=torch.float64))
   tensor_result.backward()
   torch.testing.assert_close(tensor_x.grad, torch.tensor(8.0, dtype=torch.float64))
   print(f"Native PyTorch: {tensor_result.item():.1f}")
   print(f"Derivative at x=3: {tensor_x.grad.item():.1f}")

.. testoutput:: backend_evaluation

   Native PyTorch: 16.0
   Derivative at x=3: 8.0

The derivative is ``2*x + 2``, giving ``8`` at ``x = 3``. Calling ``backward()``
sets the input tensor's gradient. The exported module is eager, not a
TorchScript or ``torch.compile`` target; closed-over Coker constants do not
become trainable PyTorch parameters.

The backend changes execution and native representations, not the formula.
Supported operations and solvers still differ between backends.

You have completed the introductory sequence. Continue with
:doc:`../how_to/backend_interoperability` for backend-specific interoperability
recipes, or consult :doc:`../backends` for the wider capability reference.
