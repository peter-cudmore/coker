import pathlib
from typing import Type, Callable, Tuple, Dict
from coker.modelling import Block
from coker.algebra.kernel import Function
from coker.helpers import get_all_subclasses

__ALL__ = [
    "Workspace",
]


class ComponentHandle:
    def __init__(self, path, in_ports, out_ports):
        self.path = path
        self.in_ports = in_ports
        self.out_ports = out_ports

    @property
    def input(self):
        if len(self.in_ports) == 1:
            return self.in_ports[0]

        return self.in_ports

    @property
    def output(self):
        if len(self.out_ports) == 1:
            return self.out_ports[0]

        return self.out_ports


class Worksheet:

    def __init__(self):
        self.components = []
        self.connections = []
        self.input_ports = set()

    def add_instance(self, constructor: Callable, *args, name=None, **kwargs):
        try:
            component: coker.Block = constructor(*args, **kwargs)
        except Exception as e:
            raise e
        if not isinstance(component, Block):
            raise TypeError("Component must be a subclass of `coker.Block`")

        index = len(self.components)
        self.components.append(component)

        component_path = (
            f"/{name}" if name else f"/{component.__class__.__name__}{index}"
        )
        in_ports = [f"{component_path}/in_{i}" for i in component.spec.inputs]
        out_ports = [
            f"{component_path}/in_{i}" for i in component.spec.outputs
        ]
        handle = ComponentHandle(component_path, in_ports, out_ports)

        return handle

    @property
    def free_parameters(self):
        for component in self.components:
            component.free_parameters()


class Simulation(Worksheet):
    def __init__(self, window: Tuple[float | int, float | int], step_size):
        super().__init__()
        self.window = window
        self.probes = []
        self.step_size = step_size

    def add_probe(self, signal, *args, renderer=None, **kwargs):
        self.probes.append(signal)

    def run(self):
        # compile model into dae
        # 0 = Af(t, x, u, p ) - u
        # 0 = Phi(t, dx, x, u, p)
        pass


class BuildRecipe(Worksheet):
    def __init__(self, name_space: str):
        self.name_space: str = name_space
        self.kernels: Dict[str, Function] = {}

    def add_function(self, name: str, function: Function):
        self.kernels[name] = function


def get_worksheets():
    build_recipes = get_all_subclasses(BuildRecipe)
    sim_recipes = get_all_subclasses(Simulation)

    return [("Deploy", name, value) for name, value in build_recipes] + [
        ("Simulation", name, value) for name, value in sim_recipes
    ]
