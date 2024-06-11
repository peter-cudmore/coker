import matplotlib.pyplot as plt
from typing import List

import numpy as np
from coker.toolkits.codesign import ProblemBuilder, Minimise
from coker.toolkits.kinematics import RigidBody, Isometry3, Inertia, Screw, Revolute, Free
from coker.toolkits.spatial import Rotation3

coxa_intertia = Inertia(
    centre_of_mass=Isometry3(translation=np.array([7.083, 0.003, 0.217])),
    mass=127.677,
    moments=np.array([10961.79, -2.51, -31.522, 19145.553, -2.755, 17119.771])
)
femur_inertia = Inertia(
    centre_of_mass=Isometry3(translation=np.array([90.0, -0.34, 0.0])),
    mass=250.748,
    moments=np.array([20670.926, -0.378, 29.919, 30410.0, 0.0, 30940.0])
)
tibia_inertia = Inertia(
    centre_of_mass=Isometry3(translation=np.array([162.971, -0.518, -27.12])),
    mass=8.216,
    moments=np.array([16793.318, 4.209, 561.098, 15265.321, -8.506, 2889.306])
)
body_intertia = Inertia(
    centre_of_mass=Isometry3.identity(),
    mass=1000,
    moments=1e6 * np.array([1, 0, 0, 1, 0, 1])
)


def build_hexapod_leg(model, body_idx, anchor, coxa_length, femur_length, tibia_length):
    coxa = model.add_link(
        parent=body_idx,
        at=anchor,
        joint=Revolute(Screw.w_z()),
        inertia=coxa_intertia
    )
    femur_coxa_joint = Isometry3(translation=coxa_length * np.array([1., 0., 0.]))
    femur_tibia_joint = Isometry3(translation=femur_length * np.array([1., 0., 0.]))
    v = np.array([34.630337, 0.0, -156.20737])
    v = tibia_length * v / np.sqrt(v.dot(v))
    foot_transform = Isometry3(translation=v)

    femur = model.add_link(
        parent=coxa,
        at=femur_coxa_joint,
        joint=Revolute(Screw.w_y()),
        inertia=femur_inertia
    )

    tibia = model.add_link(
        parent=femur,
        at=femur_tibia_joint,
        joint=Revolute(Screw.w_y()),
        inertia=tibia_inertia
    )
    foot = model.add_effector(
        parent=tibia,
        at=foot_transform
    )

    return coxa, femur, tibia, foot


def build_hexapod_model(coxa_length, femur_length, tibia_length):
    model = RigidBody()
    anchors = get_anchors()
    body_idx = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        inertia=body_intertia,
        joint=Free()
    )
    _ = [
        build_hexapod_leg(model, body_idx, anchor, coxa_length, femur_length, tibia_length)
        for anchor in anchors
    ]
    return model


def static_hexapod_codesign():
    # static torque
    # A q, the (J^s_{st})^T maps spatial wrenches applied at the end effector
    # to joint torques.
    # If this force, F_i, is equal and opposite to the force applied to the
    # body
    #
    # For each foot, we have an external force f^i_ext
    # applied at the tip. This is a "free" force
    # that satisfies f^i_z >= 0, \sum_i f^i_{xy} = 0
    #
    # we transform that to a wrench at the base; call this w_b^i

    # For each Link, we have a force due to gravity, applied at the com
    # we transform this to a wrench applied at the base

    # by newton, we have sum of all forces and wrenches at the base
    # must balance.

    # we want to do this by minimising joint torque
    # while maximising area spanned by footprint

    # this gives an optimisation problem in q, [l_coxa, l_femur, l_tibia]

    def problem_torques(q, h, lc, lf, lt, forces):
        model = build_hexapod_model(lc, lf, lt)
        q_actual = np.concatenate([h * np.array([0, 0, 0, 0, 0, 1]), q])
        q_dot = np.zeros_like(q_actual)
        q_ddot = np.zeros_like(q_dot)
        g = np.array([0, 0, -9800])  # mm/s^2

        torques = model.inverse_dynamics(q_actual, q_dot, q_ddot, g, forces)

        foot_positions = model.forward_kinematics(q_actual)
        origin = np.zeros((3,))
        e_z = np.array([[0], [0], [1]], dtype=float)
        contact_constraints = [
            e_z.T @ f.apply(origin) > 0 for f in foot_positions
        ]
        # contact cone constraints

        # motor torque bounds
        torque_constraints = [t < 1500 for t in torques[5:]]
        torque_constraints += [t > - 1500 for t in torques[5:]]

        return torques, contact_constraints, torque_constraints

    q_test = np.zeros((18,))
    h_test = 0
    lc_test = 50 #mm
    lf_test = 50
    lt_test = 100
    f_rest = 9800   #nmm
    forces = [f_rest * np.array([0, 0, 0, 0, 0, 1])] * 6

    t_test, cc_test, tc_test = problem_torques(q_test, h_test, lc_test, lf_test, lt_test, forces)

    return


def hexapod_codesign():
    motor_max = 0.52  # Netwon Meters

    with ProblemBuilder() as builder:
        femur_length = builder.new_variable(name='l_f')
        coxa_length = builder.new_variable(name='l_c')
        tibia_length = builder.new_variable(name='l_c')
        static_torques = builder.new_variable(r'\tau', shape=(18,))
        contact_forces = builder.new_variable('f_z', shape=(6,))

        q_joints = builder.new_variable('q', shape=(18,))
        rest_height = builder.new_variable('h')
        q = np.concatenate([rest_height * np.array([0., 0., 1, 0, 0, 0]), q_joints])

        q_dot = np.zeros(shape=(24,), dtype=float)
        q_ddot = np.zeros(shape=(24,), dtype=float)

        model = build_hexapod_model(1000 * femur_length, 1000 * coxa_length, 1000 * tibia_length)
        ones = np.ones(shape=(18,), dtype=float)
        constraints = [
            -0.5 * motor_max * ones < static_torques,
            static_torques < 0.5 * motor_max * ones,
            - ones * np.pi / 2 < q_joints,
            q_joints < ones * np.pi / 2,
            0.01 < femur_length,
            femur_length < 0.2,
            0.01 < coxa_length,
            coxa_length < 0.2,
            0.05 < tibia_length,
            tibia_length < 0.2,
            rest_height > 0.2
        ]
        e_z = np.array([0, 0, 1], dtype=float)

        potential_energy = model.potential_energy(q, -9.8 * e_z)

        # static torque = partial(potential, q)

        Js_st = model.spatial_manipulator_jacobian(q)
        assert Js_st.shape == (6 * 6, 24)

        # Amps... approximately
        cost = 1.47 * np.abs(static_torques).T @ np.ones(shape=(18,)) + potential_energy

        builder.constraints = constraints
        builder.objective = Minimise(cost)
        builder.outputs = [coxa_length, femur_length, tibia_length, q_joints]

        problem = builder.build()


e_z = np.array([0, 0, 1], dtype=float)


def get_anchors(distance_from_center=50):
    angles = [np.pi / 6, np.pi / 2, 5 * np.pi / 6, -5 * np.pi / 6, -np.pi / 2, -np.pi / 6]

    return [Isometry3(rotation=Rotation3(axis=e_z, angle=a))
            @ Isometry3(translation=np.array([distance_from_center, 0, 0]))
            for a in angles]


def main():
    origin = np.zeros((3,), dtype=float)
    lines = []
    angles = np.zeros(shape=(24,), dtype=float)
    anchors = get_anchors()
    model = build_hexapod_model(50, 80, 90)
    feet_idx = range(len(model.end_effectors))
    feet = model.forward_kinematics(angles)
    for foot_idx, foot in zip(feet_idx, feet):
        joints = model.joint_locations(angles, foot_idx)

        joint_p = [p @ origin for p in joints]
        joint_p.append(foot @ origin)
        lines.append(joint_p)

    lines.append([a @ origin for a in anchors])
    ax = plt.figure().add_subplot(projection='3d')
    for line in lines:
        plot_lines(ax, line)
    ax.set_xlabel('forward')

    ax.set_xlabel('forward (mm)')
    ax.set_ylabel('left (mm)')
    ax.set_zlabel('up (mm)')
    ax.set_title('Hexapod Pose')
    plt.show()


def plot_lines(ax: plt.Axes, lines: List[np.ndarray]):
    line_array = np.vstack(lines).T
    ax.scatter(line_array[0, :], line_array[1, :], line_array[2, :], 'ko')
    ax.plot(line_array[0, :], line_array[1, :], line_array[2, :])


if __name__ == '__main__':
    static_hexapod_codesign()
