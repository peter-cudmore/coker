Your first Coker application
============================

After :doc:`installation`, define a Celsius-to-Fahrenheit conversion using
the NumPy backend:

.. testcode:: first_application

   from coker import Scalar, function

   fahrenheit = function(
       arguments=[Scalar("celsius")],
       implementation=lambda celsius: 1.8 * celsius + 32.0,
       backend="numpy",
       name="fahrenheit",
   )

   freezing = fahrenheit(0.0)
   boiling = fahrenheit(100.0)
   assert freezing == 32.0
   assert boiling == 212.0
   print(f"Freezing: {freezing:.1f} F; boiling: {boiling:.1f} F")

.. testoutput:: first_application

   Freezing: 32.0 F; boiling: 212.0 F

``Scalar("celsius")`` declares a scalar input. ``implementation`` defines the
calculation, and ``backend="numpy"`` selects NumPy for numerical execution.
Calling ``fahrenheit`` evaluates the function.

Continue with :doc:`../tutorials/function_evaluation`, then follow the tutorials
in order to learn about symbolic arrays, conditionals, and backend selection.

.. Author notes

   - Choose the hello-world computation and explain why it is useful.
   - Include every import and use the base NumPy backend.
   - Introduce argument spaces, the Python implementation, and function calls.
   - Show the expected numeric result and how to check it.
   - Leave backend internals and advanced composition to later pages.
   - End with a link to the function-evaluation tutorial.
