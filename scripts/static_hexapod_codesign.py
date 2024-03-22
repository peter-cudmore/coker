import numpy as np
from coker.toolkits.codesign import DesignVariable, norm

from coker.algebra import Tensor


def build_hexapod_model(root_pose, q, g):
    pass


def hexapod_codesign():
    motor_max = 0.52    # Netwon Meters

    femur_length = DesignVariable()
    coxa_length = DesignVariable()
    tibia_length = DesignVariable()

    static_torques = DesignVariable(shape=(18,))

    joint_positions = DesignVariable(shape=(18, ))
    joint_velocities = np.zeros(shape=(18,))
    joint_acceleration = np.zeros(shape=(18,))
    rest_height = DesignVariable()
    body_position = Tensor.from_list([0, 0, 0, 0, 0, rest_height])
    body_velocity = np.zeros(shape=(6,))
    body_acceleration = np.zeros(shape=(6,))

    g = np.array([0, 0, -9.8])
    model = build_hexapod_model(
        root_pose=body_position,
        q=joint_positions,
        g=g
    )

    feet = model.forward_kinematics()

    constraints = [
        -0.5 < static_torques < 0.5,
        -np.pi/2 < joint_positions < np.pi / 2,
        0 < femur_length < 0.2,
        0 < coxa_length < 0.2,
        0 < tibia_length < 0.2,
        tibia_length < 2 * rest_height
    ]

    cost = 1.47 * norm(static_torques, 1)  # Amps... approximately

