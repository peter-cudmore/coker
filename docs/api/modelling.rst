System modelling
================

Working wiring versus simulation
--------------------------------

The current toolkit supports component metadata, port discovery and connection
wiring. It does **not** provide a released compiler or simulator for those
connections. ``Integrator`` declares state but does not integrate;
``Gain``, ``Sum`` and ``Difference`` declare ports but do not numerically
execute. The PID script illustrates model construction, not a supported
end-to-end simulation contract.

Import container and metadata types, including discovery, from
``coker.toolkits.system_modelling``. Import standard blocks from
``coker.toolkits.system_modelling.std_lib.common.signal_processing``.
Importing component modules registers their classes: discovery does not scan
or import libraries and its hints are always ``None``.

.. testcode:: modelling_wiring

   from coker.toolkits.system_modelling import (
       BlockContainer, Signal, list_components,
   )
   from coker.toolkits.system_modelling.std_lib.common.signal_processing import (
       Gain, Integrator,
   )

   library = "coker/toolkits/system_modelling/std_lib/common/signal_processing"
   assert ("Gain", library, None) in list_components()
   plant = BlockContainer("plant")
   gain, integrator = Gain(2.0, name="gain"), Integrator(name="state")
   plant.add_models(gain, integrator)
   external_input = plant.add_input(Signal("u"))
   external_output = plant.add_output(Signal("y"))
   plant.add_connections(
       (gain.input, external_input),
       (integrator.input, gain.output),
       (external_output, integrator.output),
   )
   assert len(plant.connections) == 3
   assert plant.connections[integrator.input.resolve()] == gain.output.resolve()
   assert gain.spec().parameters[0].value == 2.0
   assert len(plant.spec().state) == 1

Connections have one driver per input. Components must be attached before
resolving their ports, and one component cannot be attached to two parents.
Single-port blocks expose a handle; multiport blocks expose a ``PortBundle``
selectable by name or zero-based integer. No dimension, unit or clock-domain
compatibility checking is performed during wiring. Container input handles
act as internal sources; container output handles act as internal sinks.

Ordinary user API
-----------------

.. autofunction:: coker.toolkits.system_modelling.list_components

.. autoclass:: coker.toolkits.system_modelling.BlockContainer
   :members: spec, add_models, add_connection, add_connections, add_input, add_output

.. autoclass:: coker.toolkits.system_modelling.PortBundle

.. autoclass:: coker.toolkits.system_modelling.Signal
   :members: dimension

.. autoclass:: coker.toolkits.system_modelling.Angle

.. autoclass:: coker.toolkits.system_modelling.Position3D
   :members: dimension

.. autoclass:: coker.toolkits.system_modelling.Orientation3D
   :members: dimension

.. autoclass:: coker.toolkits.system_modelling.Variable
   :members: dimension

.. autoclass:: coker.toolkits.system_modelling.VectorVariable
   :members: dimension

.. autoclass:: coker.toolkits.system_modelling.Parameter
   :members: dimension, is_constant

.. autoclass:: coker.toolkits.system_modelling.ConstantParameter
   :members: dimension

.. autoclass:: coker.toolkits.system_modelling.Vector

``Signal``, ``Variable`` and ``Parameter`` are scalar metadata. Position and
orientation declarations have three components; orientation uses axis-angle.
``VectorVariable`` has an explicit dimension, but ``Vector`` currently inherits
the scalar ``Parameter.dimension`` method. ``ValueType`` is the existing
float-or-Parameter type alias, not a numeric array evaluation API.

Standard block metadata
-----------------------

.. autoclass:: coker.toolkits.system_modelling.std_lib.common.signal_processing.Gain

.. autoclass:: coker.toolkits.system_modelling.std_lib.common.signal_processing.Integrator

.. autoclass:: coker.toolkits.system_modelling.std_lib.common.signal_processing.Difference

.. autoclass:: coker.toolkits.system_modelling.std_lib.common.signal_processing.Sum

.. autoclass:: coker.toolkits.system_modelling.std_lib.common.signal_processing.Constant

.. autoclass:: coker.toolkits.system_modelling.std_lib.common.signal_processing.TransferFunction

Extension API
-------------

Subclass ``Block`` with a ``BlockSpec`` to declare ordered input/output signals,
state variables and parameters. This registers metadata for discovery, not an
evaluation implementation. Use the existing port and container methods rather
than private maps or generated-I/O internals.

.. autoclass:: coker.toolkits.system_modelling.Block
   :members: spec, input, output

.. autoclass:: coker.toolkits.system_modelling.BlockSpec

Registration and wiring internals
---------------------------------

``CokerModel``, ``model``, ``is_model`` and ``is_component_def`` are public
factory-registration utilities, not independent simulation entry points. The
model decorator wraps a factory and retains its function name; it does not
compile or run a dynamical system.

Indices, lazy handles, forwarding components, subclass-registration machinery
and ``get_component_registry`` remain in the ``modelling`` implementation
module rather than the toolkit's public exports. Lazy handles' ``resolve``
method exposes container-local ``(component_index, port_index)`` pairs for
inspection. Applications should obtain handles through blocks and containers.

``Step`` is discoverable after importing its module, but its constructor
currently passes unsupported arguments to ``Block`` and raises ``TypeError``.
``SignalGenerator`` can directly call its supplied function, but its
class-level ``spec`` shadows the base method and is not compatible with normal
``BlockContainer.add_models`` collection. Neither is presented as a working
container example. No optional solver dependency makes these incomplete
simulation paths supported.
