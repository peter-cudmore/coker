import coker
from coker import BlockContainer
from coker.std_lib import *


@coker.model
def plant():
    # Motor model from https://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition&section=SimulinkModeling
    plant = coker.BlockContainer("plant")

    voltage_in = plant.add_input(Signal("voltage_in"))
    position_out = plant.add_output(Signal("position_out"))

    R = 4
    L = 2.75e-6
    K = 0.0274
    J = 3.2284e-6
    b = 3.5077e-6

    add_11 = Difference()
    add_12 = Difference()
    inductance = Gain(1 / L, name="1/L")
    Kt = Gain(K, name="K")
    Ke = Gain(K, name="K")
    resistance = Gain(R, name="R")
    add = Difference()
    inertia = Gain(1 / J, name="1/J")
    damping = Gain(b, name="b")

    integrator = Integrator("d/dt(theta)")
    integrator1 = Integrator("theta")
    integrator2 = Integrator("i")

    plant.add_models(
        add_11,
        add_12,
        inductance,
        Kt,
        add,
        integrator,
        inertia,
        damping,
        integrator1,
        integrator2,
        Ke,
        resistance,
    )

    plant.add_connections(
        (add_12.input["+"], add_11.output),
        (inductance.input, add_12.output),
        (inductance.output, integrator2.input),
        (integrator2.output, Kt.input),
        (Kt.output, add.input["+"]),
        (add.output, inertia.input),
        (inertia.output, integrator.input),
        (integrator.output, integrator1.input),
        (integrator1.output, position_out),
        (integrator.output, damping.input),
        (damping.output, add.input["-"]),
        (voltage_in, add_11.input["+"]),
        (integrator2.output, resistance.input),
        (resistance.output, add_11.input["-"]),
        (integrator.output, Ke.input),
        (Ke.output, add_12.input["-"]),
    )

    return plant


class Differentiator(Block):
    def __init__(self, name):
        spec = BlockSpec(
            inputs=[Signal("u")], outputs=[Signal("x")], state=[Variable("x")]
        )
        super(Differentiator, self).__init__(name, block_spec=spec)

    def __call__(self, clock, inputs, state):
        (u,) = inputs
        (x,) = state
        dx = (u - x) / clock.dt
        return [dx, dx]


@coker.model
def controller():
    controller = coker.BlockContainer("controller")
    position_in = controller.add_input(Signal("position_in"))
    setpoint_in = controller.add_input(Signal("setpoint_in"))
    voltage_out = controller.add_output(Signal("voltage_out"))

    e = Difference(name="e")
    k_p = Gain(1, name="k_p")
    k_i = Gain(1, name="k_i")
    i = Integrator(name="1/s")
    k_d = Gain(1, name="k_d")
    d = Differentiator(name="s")
    summer = Sum(name="summer", inputs=3)

    controller.add_models(
        e,
        k_p,
        k_i,
        i,
        k_d,
        d,
        summer,
    )
    controller.add_connections(
        (setpoint_in, e.input["+"]),
        (position_in, e.input["-"]),
        (summer.output, voltage_out),
        # Prop path
        (e.output, k_p.input),
        (k_p.output, summer.input[0]),
        # Diff Path
        (e.output, k_d.input),
        (k_d.output, d.input),
        (d.output, summer.input[1]),
        # Int Path
        (e.output, k_i.input),
        (k_i.output, i.input),
        (i.output, summer.input[2]),
    )

    return controller


def main():
    p = plant()
    c = controller()


if __name__ == "__main__":
    main()
