import numpy as np
from coker.toolkits.kinematics import (
    RigidBody,
    Revolute,
    Inertia,
    KinematicsVisualiser,
)
from coker.toolkits.spatial import Isometry3, Screw
import matplotlib

matplotlib.use("qtagg")
import matplotlib.pyplot as plt


block_inertia = Inertia(
    centre_of_mass=Isometry3(
        translation=np.array([0.0, 0, -0.5])
    ),  # 0.5 m in y
    mass=1,  # 1kg
    moments=0.01 * np.array([1, 0, 0, 1, 0, 1]),
)

w_y = Screw.from_tuple(0, 1, 0, 0, 0, 0)
w_z = Screw.from_tuple(0, 0, 1, 0, 0, 0)
double_pendulum = RigidBody()
link_1 = double_pendulum.add_link(
    parent=double_pendulum.WORLD,
    at=Isometry3.identity(),
    joint=Revolute(w_y),
    inertia=block_inertia,
)

link_2 = double_pendulum.add_link(
    parent=link_1,
    at=Isometry3(translation=np.array([0, 0, -1])),
    joint=Revolute(w_y),
    inertia=block_inertia,
)

tip = double_pendulum.add_effector(
    parent=link_2, at=Isometry3(translation=np.array([0, 0, -1]))
)

visualiser = KinematicsVisualiser(double_pendulum)
visualiser.set_view(z_lim=(-2.1, 0.1), x_lim=(-1, 1), y_lim=(-1, 1))


n = 50
q_sweep = [i * np.array([np.pi, 0]) / n for i in range(-n, n + 1)] + [
    np.array([np.pi / 2, i * np.pi / n]) for i in range(-n, n + 1)
]
# visualiser.animate_sweep(q_sweep_1, loop=False)

visualiser.animate_sweep(q_sweep, loop=False)
