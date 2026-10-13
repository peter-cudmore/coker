API Reference
=============

Use these imports for the supported user workflows in this checkout. Reference
pages name the actual exported/owning paths; internal modules are not implicitly
stable because they are importable. Boundary changes tracked by issue #88 are
not presumed to have shipped: current ambient-backend and constructor-export
limitations are called out at the affected signatures.

.. list-table:: Public API inventory
   :header-rows: 1

   * - Domain
     - Canonical imports
     - Reference
   * - Callables, spaces and tensor construction
     - ``coker``
     - :doc:`functions`, :doc:`spaces`, :doc:`conditional`
   * - Numeric and function-valued parameters
     - ``coker.parameters``, ``coker.parameters.function_parameters``
     - :doc:`parameters`
   * - Backend lookup, signatures, imports and options
     - ``coker.backends``, ``coker.backends.lowered``, backend-specific modules
     - :doc:`backends`
   * - Mathematical programs and solve status
     - ``coker.toolkits.codesign``
     - :doc:`optimisation`
   * - Dynamics, variational solving and structural analysis
     - ``coker.dynamics``; general constructors in ``coker.dynamics.system``
     - :doc:`dynamics`
   * - Spatial algebra and rigid-body models
     - ``coker.toolkits.spatial``, ``coker.toolkits.kinematics``
     - :doc:`robotics`
   * - Component model wiring and discovery
     - ``coker.toolkits.system_modelling`` and documented owning modules
     - :doc:`modelling`

Ordinary user reference
-----------------------

.. toctree::
   :maxdepth: 1

   functions
   spaces
   conditional
   optimisation
   parameters
   backends
   dynamics
   robotics
   modelling

Advanced authors
----------------

.. toctree::
   :maxdepth: 1

   extensions
