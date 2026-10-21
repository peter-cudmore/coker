Tracing and execution
=====================

Coker separates the definition of a computation from its numerical execution.
It calls a Python implementation with symbolic inputs to record supported
operations, then delegates the recorded computation to a backend.

The pipeline is::

   Python callable + argument spaces
                  |
                tracing
                  |
        Function (tape + signature + outputs)
                  |
           backend lowering
                  |
          execution handle + concrete inputs
                  |
                results

Tracing records operations
--------------------------

Argument spaces describe the inputs. During tracing, symbolic values stand in
for concrete data. Their NumPy hooks intercept supported operations such as
``np.dot`` and record them on a *tape*: an ordered representation of operations
and their dependencies.

The resulting ``Function`` holds the tape, its outputs and its input/output
signature. It also records the backend selected for concrete evaluation.
The tape describes the mathematical computation, not the Python source code
or every action performed while tracing.

The following example uses only the base installation. It records a squared
norm, then evaluates it through both the ``Function`` and a lowered handle.

.. testcode:: evaluation_pipeline

   import numpy as np
   from coker import VectorSpace, function

   def evaluate_energy(x):
       return np.dot(x, x)

   # Tracing calls evaluate_energy with symbolic arguments and records its ops.
   energy = function(
       [VectorSpace("x", 2)],
       evaluate_energy,
       backend="numpy",
       name="energy",
   )

   # Lowering produces a backend execution handle for the recorded computation.
   executable = energy.lower()
   sample = np.array([3.0, 4.0])

   # Function calls unwrap a single output; execute() always returns a tuple.
   direct = energy(sample)
   (packed,) = executable.execute((sample,))
   positional = executable(sample)
   np.testing.assert_allclose([direct, packed, positional], [25.0, 25.0, 25.0])
   print(f"Function: {direct:.1f}; lowered: {positional:.1f}")

.. testoutput:: evaluation_pipeline

   Function: 25.0; lowered: 25.0

Here, tracing records ``np.dot(x, x)`` without knowing the numerical value of
``x``. Passing ``[3, 4]`` later evaluates that recorded operation.

Lowering prepares execution
---------------------------

Lowering asks a backend to prepare an execution handle for the ``Function``.
The handle may contain a reusable execution plan or a native backend function;
lowering does not necessarily mean generating machine code.

``Function.lower()`` caches handles by backend instance and lowering options.
A concrete ``Function`` call uses this lowering path, so repeated calls reuse
the handle rather than retracing the Python implementation. The example's
three calls therefore evaluate the same recorded computation. The packed
``execute(inputs)`` interface always returns a tuple; positional handle calls
and ordinary ``Function`` calls unwrap a single output.

Backends own their native representations and calls. A model built from
supported operations can be lowered to another backend, but an imported native
call remains tied to its owning backend. See the :doc:`backend guide
<../backends>` for lowering interfaces and backend capabilities.

Design tradeoffs
----------------

Tracing lets a model use familiar Python and supported NumPy operations
without requiring Coker to compile arbitrary Python. The cost is that Python
control flow runs while the graph is constructed. A fixed-length loop can
record repeated operations, but a native Python ``if`` cannot record a choice
based on a symbolic input. Use supported symbolic selection instead; see
:doc:`../basics/sharp_edges`.

Separating the tape from backend execution lets the same mathematical model
serve numerical and symbolic workflows. It does not make the backends
interchangeable: supported operations, solvers and numerical behaviour differ.
Backend-owned native calls provide access to specialised tools at the cost
of portability.

The tape and tracing machinery are compiler internals. Most users need only
argument spaces, ``function`` and the resulting ``Function``; see
:doc:`../api/index` for the public interface boundary.

.. Author notes

   - Walk through one small expression from Python callable to evaluated result.
   - Explain what happens during tracing versus concrete evaluation.
   - Describe NumPy hooks and how symbolic operations are recorded on the tape.
   - Explain the relationship between the dynamic AST/tape and a Function.
   - Describe lowering, execution handles, caching, and backend-owned native calls.
   - Explain why this design supports multiple backends and what it restricts.
   - Include a pipeline diagram and justify the main design tradeoffs.
   - Distinguish core Coker from the separately packaged native runtime.
