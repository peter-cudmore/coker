from typing import List, Union, Optional, NewType, Dict, Tuple
import inspect

from coker import Scalar
from coker.toolkits.system_modelling.modelling.coker_abc import (
    CokerListableSubclasses,
)


def get_component_registry():

    return {
        item.__name__: inspect.getsourcefile(item) for item in Block.__subclasses__()
    }


BlockIndex = NewType("BlockIndex", int)
PortIndex = NewType("PortIndex", int)


class LazyPortHandle:
    def __init__(self, component, port_index: PortIndex):
        self.component = component
        self.port_index = port_index

    def resolve(self) -> (BlockIndex, PortIndex):
        """Return container-local component and port indices.

        Returns:
            tuple[int, int]: Index of the attached component and local port.
                Resolving an unattached component raises AssertionError.
        """
        parent: BlockContainer = self.component.parent
        assert parent is not None

        index = parent.components.index(self.component)
        return index, self.port_index

    def __repr__(self):
        return f"{self.component}({self.port_index})"


class LazyInputHandle(LazyPortHandle):
    pass


class LazyOutputHandle(LazyPortHandle):
    pass


class CokerModel:
    """Wrap a model factory without compilation or numerical execution.

    Args:
        func: Callable returning a component model.

    Prefer the ``model`` decorator, for example ``factory = model(make_plant)``.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        """Call the wrapped factory with its positional and keyword arguments."""
        return self.func(*args, **kwargs)

    def name(self):
        """Return the wrapped factory's Python function name."""
        return self.func.__name__


def model(func):
    """Wrap a component factory without compiling or simulating it.

    Args:
        func: Callable constructing a model.

    Returns:
        CokerModel: Callable wrapper delegating to the original factory.
    """
    return CokerModel(func)


def is_component_def(obj) -> bool:
    """Return whether an object is a Block or BlockContainer subclass."""
    try:
        if issubclass(obj, Block):
            return True
        if issubclass(obj, BlockContainer):
            return True
    except TypeError:
        pass
    return False


def is_model(obj) -> bool:
    """Return whether an object is a model-decorated factory wrapper."""
    return isinstance(obj, CokerModel)


class PortForwarder:
    def __init__(self):
        self.ports = []
        self.parent = None

    def set_parent(self, parent: "BlockContainer"):
        assert self.parent is None
        self.parent = parent

    def create(self, signal) -> PortIndex:

        index = len(self.ports)
        self.ports.append(signal)
        return index


class BlockContainer(CokerListableSubclasses):
    """Collect component metadata and connections, not executable simulation.

    Args:
        name: Container name.

    Add components before connecting their lazy ports. Containers include two
    internal forwarding components, so user components start at index two.

    Examples:
        >>> from coker.toolkits.system_modelling import BlockContainer, Signal
        >>> plant = BlockContainer("plant")
        >>> source = plant.add_input(Signal("u"))
        >>> sink = plant.add_output(Signal("y"))
        >>> plant.add_connections((sink, source))
        >>> len(plant.connections)
        1
    """

    def __init__(self, name):
        self.name = name
        """Exposed quantities, forwarded from internal components."""
        self._inputs = PortForwarder()
        self._outputs = PortForwarder()

        self._parameters = []
        self._state = []
        self.parameter_map = {}

        self.labels: Dict[(BlockIndex, PortIndex), str] = {}

        self.components: List[Union[Block, "BlockContainer", PortForwarder]] = []
        self.add_models(self._inputs, self._outputs)

        self.parent = None

        """Dictionary of internal connections as input: output"""
        self.connections: Dict[(BlockIndex, PortIndex), (BlockIndex, PortIndex)] = {}

    def spec(self):
        """Return current inputs, outputs, accumulated parameters and state."""
        return BlockSpec(
            inputs=self._inputs.ports,
            outputs=self._outputs.ports,
            parameters=self._parameters,
            state=self._state,
        )

    def add_models(self, *components: Union["Block", "BlockContainer", PortForwarder]):
        """Attach components in order and collect their state/parameters."""
        for component in components:
            component.set_parent(self)
            self.components.append(component)
            if not (isinstance(component, PortForwarder)):
                spec = component.spec()
                self._state += spec.state if spec.state is not None else []
                self._parameters += (
                    spec.parameters if spec.parameters is not None else []
                )

    def set_parent(self, parent):
        assert (
            self.parent is None
        ), f"{self.name} is already attached to model {self.parent}"
        self.parent = parent

    def add_connections(self, *connections: Tuple):
        """Wire tuples of input/output handles, optionally with a label.

        Args:
            connections: Tuples ``(input, output)`` or ``(output, input)``;
                an optional third element labels the source output.
        """
        for port_in, port_out, *label in connections:
            if isinstance(port_in, LazyOutputHandle) and isinstance(
                port_out, LazyInputHandle
            ):
                port_in, port_out = port_out, port_in
            elif isinstance(port_in, LazyInputHandle) and isinstance(
                port_out, LazyOutputHandle
            ):
                pass
            else:
                raise TypeError("Cannot connect {} to {}", port_in, port_out)
            if len(label) == 0:
                self.add_connection(port_in, port_out)
            elif len(label) == 1:
                (name,) = label
                self.add_connection(port_in, port_out)
                self.labels[port_out.resolve()] = name

    def add_connection(self, port_in: LazyInputHandle, port_out: LazyOutputHandle):
        """Store an input-to-output connection; reject duplicate drivers."""
        in_port = port_in.resolve()
        out_port = port_out.resolve()
        assert in_port not in self.connections, (
            f"Component input {port_in.component}: "
            f"{port_in.component.spec().inputs[port_in.port_index]} is "
            "already in use"
        )
        self.connections[in_port] = out_port

    def add_input(self, signal) -> LazyPortHandle:
        """Expose a container input, returning its internal output handle."""
        index = self._inputs.create(signal)

        return LazyOutputHandle(self._inputs, index)

    def add_output(self, signal) -> LazyPortHandle:
        """Expose a container output, returning its internal input handle."""
        index = self._outputs.create(signal)
        return LazyInputHandle(self._outputs, index)


class PortBundle:
    """Select a multiport handle by signal name or insertion-order integer."""

    def __init__(self, ports: dict[str, LazyPortHandle]):
        self.ports = ports

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.ports[item]
        if isinstance(item, int):
            values = tuple(self.ports.values())
            return values[item]
        raise AttributeError(item)


class Block(CokerListableSubclasses):
    """Extension base for component port/state/parameter metadata.

    Args:
        name: Component name, or None for a generated name.
        block_spec: BlockSpec declaring ports and metadata.

    This class defines wiring, not numerical evaluation or simulation.
    """

    next_index = 0

    def __init__(self, name: Union[str, None], block_spec: "BlockSpec"):
        self._spec = block_spec
        if name is None:
            name = f"{self.__class__.__name__}_{self.next_index}"
            self.next_index += 1
        self.name = name
        self.parent = None
        self.__generate_io_from_spec()

    def __generate_io_from_spec(self):
        pass

    def spec(self) -> "BlockSpec":
        """Return the component's BlockSpec."""
        return self._spec

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def _generate_local_map(self, prefix):
        #    key    value
        #
        pass

    def set_parent(self, parent):
        assert (
            self.parent is None
        ), f"{self.name} is already attached to model {self.parent}"
        self.parent = parent

    @property
    def input(self) -> Union[LazyPortHandle, PortBundle]:
        """Return an input handle or indexable PortBundle for many inputs."""
        assert self.spec().inputs, f"Component {self} has no inputs"

        if len(self.spec().inputs) == 1:
            return LazyInputHandle(self, 0)
        else:
            return PortBundle(
                {
                    value.name: LazyInputHandle(self, idx)
                    for idx, value in enumerate(self.spec().inputs)
                }
            )

    @property
    def output(self) -> Union[LazyPortHandle, PortBundle]:
        """Return an output handle or indexable PortBundle for many outputs."""
        assert self.spec().outputs, f"Component {self} has no outputs"

        if len(self.spec().outputs) == 1:
            return LazyOutputHandle(self, 0)
        else:
            return PortBundle(
                {
                    value.name: LazyOutputHandle(self, idx)
                    for idx, value in enumerate(self.spec().outputs)
                }
            )


class Signal(Scalar):
    """Declare a scalar analog signal.

    Args:
        name: Port name.
        clock_domain: Optional clock metadata; no execution is scheduled.
    """

    def __init__(self, name: str, clock_domain=None):
        self.name = name
        self.clock_domain = clock_domain

    def __repr__(self):
        return f"Sig:{self.name}"

    @staticmethod
    def dimension():
        return 1


class Variable:
    """Declare a named scalar state with optional clock-domain metadata."""

    def __init__(self, name: str, clock_domain=None):
        self.name = name
        self.clock_domain = clock_domain

    @staticmethod
    def dimension():
        return 1


class VectorVariable(Variable):
    """Declare state with an explicit dimension and optional clock metadata."""

    def __init__(self, name: str, dimension: int, clock_domain=None):
        super().__init__(name, clock_domain)
        self._dimension = dimension

    def dimension(self):
        return self._dimension


class Parameter:
    """Declare a named scalar parameter with an optional fixed value."""

    def __init__(self, name: str, value: Optional[float] = None):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"Param:{self.name}"

    def is_constant(self) -> bool:
        """Return whether a fixed value was supplied."""
        return self.value is not None

    @staticmethod
    def dimension():
        return 1


class ConstantParameter:
    """Declare an unnamed fixed scalar parameter, defaulting to zero."""

    def __init__(self, value: float = 0.0):
        self.value = value

    @staticmethod
    def dimension():
        return 1


class Angle(Signal):
    """Declare an angular scalar signal (radians by convention)."""

    pass


class Position3D(Signal):
    """Declare a three-component position signal in consistent length units."""

    @staticmethod
    def dimension() -> int:
        return 3


class Orientation3D(Signal):
    """Orientation3D in axis-angle representation."""

    @staticmethod
    def dimension() -> 3:
        return 3


class Vector(Parameter):
    """Parameter subtype; currently inherits scalar dimension metadata."""

    pass


class BlockSpec:
    """Describe block ports, state and parameters without evaluation.

    Args:
        inputs: Ordered Signal input declarations, or None.
        state: Ordered Variable state declarations, or None.
        parameters: Ordered Parameter declarations, or None.
        outputs: Ordered Signal output declarations, or None.
    """

    def __init__(
        self,
        inputs: List[Signal] = None,
        state: List[Variable] = None,
        parameters: List[Parameter] = None,
        outputs: List[Signal] = None,
    ):

        self.inputs = inputs
        self.outputs = outputs
        self.state = state
        self.parameters = parameters


ValueType = Union[float, Parameter]
