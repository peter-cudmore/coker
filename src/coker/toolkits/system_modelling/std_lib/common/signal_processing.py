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
    """Wrap a callable of time; direct calls evaluate the supplied function.

    Args:
        name: Component name.
        parent: Historical unused argument; attach using add_models.
        func: Callable accepting time and returning a scalar.

    Its class-level ``spec`` shadows Block.spec; it is not currently compatible
    with ordinary container metadata collection.
    """

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
    """Declare transfer-function coefficient metadata, without simulation.

    Args:
        name: Component name.
        numerator: Numerator coefficient metadata.
        denominator: Denominator coefficient metadata.
    """

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
    """Unreleased step component; construction currently raises TypeError."""

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
    """Declare scalar input/output and one scalar state; no integrator runs.

    Args:
        name: Optional component name, default ``1/s``.
    """

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
    """Declare gain metadata, not an executable multiplication block.

    Args:
        gain: Fixed scalar gain.
        name: Optional name, also used as the parameter name when supplied.
    """

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
    """Declare a value parameter with input/output metadata.

    Args:
        value: Scalar value or Parameter metadata.
        name: Optional component name.

    The current declaration includes an input; it is not a source evaluator.
    """

    def __init__(self, value: ValueType, name=None):
        spec = BlockSpec(
            inputs=[Signal("u")],
            outputs=[Signal("y")],
            parameters=[Parameter("value", value)],
        )

        super().__init__(name, spec)


class Difference(Block):
    """Declare two inputs named ``+`` and ``-`` and one output named ``y``."""

    SPEC = BlockSpec(
        inputs=[Signal("+"), Signal("-")],
        outputs=[Signal("y")],
    )

    def __init__(self, name=None):
        super().__init__(name, Difference.SPEC)


class Sum(Block):
    """Declare a configurable number of inputs and one scalar output.

    Args:
        name: Optional component name.
        inputs: Input count, default two; names are ``u_0``, ``u_1``, etc.
    """

    def __init__(self, name=None, inputs: int = 2):
        spec = BlockSpec(
            inputs=[Signal(f"u_{i}") for i in range(inputs)],
            outputs=[Signal("y")],
        )
        super().__init__(name, spec)
