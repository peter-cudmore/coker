Tutorials
=========

Work through these five lessons in order: scalar functions, parameterised
matrices, vectors built with loops, conditionals, and backend selection.

Prerequisites
-------------

You should be comfortable defining Python functions and using basic NumPy
arrays. No previous symbolic-programming experience is required. The first
four lessons use the base installation, ``pip install coker``. The fifth
explains how to install the CasADi and PyTorch extras.

Beginner sequence
-----------------

.. toctree::
   :maxdepth: 1

   function_evaluation
   parameterized_matrix
   cumulative_products
   conditionals
   backends

Put the basics to work
-----------------------

Use :doc:`../how_to/index` for task-specific recipes. See
:doc:`../basics/sharp_edges` before adapting Python control flow or changing
model shapes. The example catalogue offers further applications:

.. toctree::
   :maxdepth: 1

   ../examples


.. Author notes

   - Entry prerequisites are Python functions and basic NumPy arrays, not prior
     symbolic programming. Keep dependency guidance with the lesson that needs it.
   - Preserve the five-step order: scalar functions, parameterised matrices,
     loop-built SymbolicVectors, conditionals, then backend selection.
   - Teach direct Coker calls before optional native export in the final lesson;
     do not make registry lookup or manual input conversion a beginner prerequisite.
   - Keep this sequence focused on everyday construction and evaluation. Link to
     the cookbook for tasks and the API reference for precise contracts.
