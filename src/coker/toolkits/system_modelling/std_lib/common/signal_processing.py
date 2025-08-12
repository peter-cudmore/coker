from coker.toolkits.system_modelling.modelling import (
    Block,
    BlockSpec,
    Signal,
    Vector,
    Parameter,
    Variable,
    ValueType,
)
from typing import Callable


class SignalGenerator(Block):
    spec = BlockSpec(
        inputs=[],
        outputs=[Signal("y")],
    )

    def __init__(self, name, parent, func: Callable[[float], float]):

        Block.__init__(self, name, SignalGenerator.spec)
        self.func = func

    def __call__(self, t, *args) -> float:
        return self.func(t)

    def free_parameters(self) -> set:
        return set()


class TransferFunction(Block):
    def __init__(self, name, numerator: Vector, denominator: Vector):
        spec = BlockSpec(
            inputs=[Signal("u")],
            outputs=[Signal("y")],
            parameters=[Vector("numerator"), Vector("denominator")],
            state=[],
        )
        Block.__init__(self, name, spec)
        self.numerator = numerator
        self.denominator = denominator


class Step(Block):
    spec = BlockSpec(
        outputs=[Signal("y")],
        parameters=[Parameter("t_step")],
    )

    def __init__(self, name, parent, t_step=None):
        if t_step is None:
            super().__init__(name, parent, self.spec)
        else:
            super().__init__(
                name,
                parent,
                BlockSpec(
                    outputs=[Signal("y")],
                    parameters=[Parameter("t_step", t_step)],
                ),
            )
            self.t_step = t_step


class Integrator(Block):
    SPEC = BlockSpec(
        inputs=[Signal("u")],
        state=[Variable("x")],
        outputs=[Signal("y")],
    )

    def __init__(self, name=None):
        if name is None:
            name = "1/s"
        super().__init__(name, Integrator.SPEC)


class Gain(Block):
    def __init__(self, gain: float, name=None):
        if name is None:
            param = Parameter(name="gain", value=gain)
        else:
            param = Parameter(name=name, value=gain)
        spec = BlockSpec(
            inputs=[Signal("u")], outputs=[Signal("y")], parameters=[param]
        )

        super().__init__(name, spec)


class Constant(Block):
    def __init__(self, value: ValueType, name=None):
        spec = BlockSpec(
            inputs=[Signal("u")],
            outputs=[Signal("y")],
            parameters=[Parameter("value", value)],
        )

        super().__init__(name, spec)


class Difference(Block):
    SPEC = BlockSpec(
        inputs=[Signal("+"), Signal("-")],
        outputs=[Signal("y")],
    )

    def __init__(self, name=None):
        super().__init__(name, Difference.SPEC)


class Sum(Block):
    def __init__(self, name=None, inputs: int = 2):
        spec = BlockSpec(
            inputs=[Signal(f"u_{i}") for i in range(inputs)],
            outputs=[Signal("y")],
        )
        super().__init__(name, spec)
