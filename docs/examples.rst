Examples and workflows
======================

Find runnable repository programs and model examples here. For guided lessons,
use :doc:`tutorials/index`; for task-focused instructions, use
:doc:`how_to/index`. The :doc:`backends` guide lists backend capabilities.

README examples
---------------

The `README <https://github.com/pneumatica-bio/coker/blob/develop/README.md#install-and-try>`_
demonstrates reusing a spring-force model for parameter fitting and inspecting
symbolic vector/matrix maths before differentiating it with PyTorch. Each Python
block is standalone and includes numerical checks. The documentation doctest
job executes those blocks directly from the README, so the examples being checked
are the ones readers copy.

Run documentation commands from the repository root, as described in
``CONTRIBUTING.md``. The documentation environment includes the CasADi and
PyTorch extras needed by the examples.

.. testcode:: readme_examples
   :hide:

   from contextlib import redirect_stdout
   from io import StringIO
   from pathlib import Path
   import re

   readme = Path("README.md")
   sources = re.findall(
       r"```python\n(.*?)```", readme.read_text(encoding="utf-8"), re.DOTALL
   )
   # The examples assert numerical behavior, not solver log formatting.
   with redirect_stdout(StringIO()):
       for source in sources:
           exec(compile(source, str(readme), "exec"), {})

Repository entry points
-----------------------

``examples/pytorch_explicit_network_training.py``
   Trains a Coker network with explicit weight and bias inputs.
   Caller-owned ``torch.nn.Parameter`` tensors receive gradients through the
   lowered PyTorch graph.

``examples/pytorch_imported_module_training.py``
   Imports a native ``torch.nn.Module`` with an explicit
   ``FunctionSignature`` and trains its parameters through the returned
   Coker function.

``examples/pytorch_mathematical_program_training.py``
   Fits an explicit network as a ``MathematicalProgram`` using Adam and
   ``PytorchNLPSolverOptions``. It selects CUDA float32 when available,
   otherwise CPU float64; the program owns the packed decision vector.

``scripts/double_pendulum.py``
   Builds a two-link rigid-body model with ``RigidBody``, ``Revolute``,
   ``Inertia``, :class:`coker.toolkits.spatial.Isometry3`, and
   :class:`coker.toolkits.spatial.Screw`, then drives a visualiser sweep.

``tests/dynamical_systems/test_variational_solver.py``
   Demonstrates dynamics and parameter fitting, including constraints and
   regularisation.

``tests/toolkits/kinematics/test_kinematics.py``
   Exercises joint coordinate contracts, two-link and branched-tree forward
   kinematics and Jacobians, single-pendulum energy and inverse dynamics, and
   hexapod-leg symbolic forward kinematics.

``tests/benchmarks/benchmark_backends.py``
   Compares backend behaviour for function evaluation, ODE integration, and a
   variational parameter-fitting workload.

For worked dynamics recipes, see :doc:`how_to/simulation`,
:doc:`how_to/system_identification`, and :doc:`how_to/path_planning`.
See :doc:`how_to/robot_arm` for a kinematics recipe.

PyTorch examples
----------------

The PyTorch examples require the optional extra:

.. code-block:: console

   uv run --extra pytorch python examples/pytorch_explicit_network_training.py
   uv run --extra pytorch python examples/pytorch_imported_module_training.py
   uv run --extra pytorch python examples/pytorch_mathematical_program_training.py

The direct explicit-parameter and imported-module examples use ordinary
PyTorch execution.

Double-pendulum visualisation
-----------------------------

Run the script with ``uv`` from the repository root:

.. code-block:: console

   uv run scripts/double_pendulum.py

Inline metadata declares NumPy and Matplotlib and resolves Coker from the
repository via a path relative to the script (``..``). ``uv`` installs these
dependencies in a script-specific environment; no package-wide plotting extra
or manual environment setup is needed.

The script imports ``KinematicsVisualiser`` from
``coker.toolkits.kinematics.visualiser``. The main kinematics package does not
import Matplotlib. No Qt backend is forced: Matplotlib uses its configured
backend, or one selected via ``MPLBACKEND``. An interactive backend needs its
usual GUI dependencies and a display.

For a headless run without a GUI:

.. code-block:: console

   MPLBACKEND=Agg uv run scripts/double_pendulum.py

The non-interactive Agg backend does not open a window or play the animation.
