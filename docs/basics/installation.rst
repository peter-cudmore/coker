Installation
============

Use Python 3.10 or newer. Create an isolated environment before installing Coker.
On Linux or macOS:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install coker

On Windows PowerShell:

.. code-block:: powershell

   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install coker

Check the installation with this Python example. It uses NumPy, which is included
in the base installation, and prints ``9``:

.. testcode:: installation

   from coker import Scalar, function

   square = function([Scalar("x")], lambda x: x**2, backend="numpy")
   result = square(3)
   assert result == 9
   print(result)

.. testoutput:: installation

   9

Install optional backends as needed:

.. code-block:: bash

   python -m pip install "coker[casadi]"
   python -m pip install "coker[pytorch]"
   python -m pip install "coker[jax]"

JAX requires Python 3.11 or newer. Plotting examples also require
``python -m pip install matplotlib``; Coker itself does not require Matplotlib.
External backend plugins are installed separately. See :doc:`../backends` for
backend capabilities and dependencies.

.. Author notes

   - State the supported Python versions and the recommended environment setup.
   - Start with the base installation and a command that verifies it works.
   - Introduce optional CasADi, PyTorch, and JAX dependencies only as needed.
   - Distinguish base dependencies, optional extras, and external plugins.
   - Explain plotting dependencies without making them mandatory for all users.
