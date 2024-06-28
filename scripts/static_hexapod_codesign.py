
import numpy as np
from coker.toolkits.codesign import ProblemBuilder, Minimise
from coker.toolkits.kinematics import RigidBody, Isometry3, Inertia, Screw, Revolute, Free, KinematicsVisualiser
from coker.toolkits.spatial import Rotation3
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from typing import List

base_coxa_length = 0.05
base_femur_length = 0.08
base_tibia_length = 0.16


coxa_intertia = Inertia(
    centre_of_mass=Isometry3(translation=np.array([0.007, 0, 0])),
    mass=0.128,
    moments=np.array([
        1.096E-05, -2.510E-09,-3.152E-08, 1.915E-05, -2.755E-09, 1.712E-05
    ])
)

femur_inertia = Inertia(
    centre_of_mass=Isometry3(translation=np.array([0.04, 0.0, 0.0])),
    mass=0.251,
    moments=np.array([2.067E-05, -3.782E-10, 2.992E-08, 0.0, 0.0, 0.0])
)

tibia_inertia = Inertia(
    centre_of_mass=Isometry3(translation=np.array([0.035, 0, -0.027])),
    mass=0.008,
    moments=np.array([
        1.679E-05,
        4.209E-09,
        5.611E-07,
        1.527E-05,
        -8.506E-09,
        2.889E-06])
)

body_intertia = Inertia(
    centre_of_mass=Isometry3.identity(),
    mass=0.0001,
    moments=1e-9 * np.array([1, 0, 0, 1, 0, 1])
)


def build_hexapod_leg(coxa_length, femur_length, tibia_length):
    model = RigidBody()
    coxa = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw.w_z()),
        inertia=coxa_intertia
    )
    femur_coxa_joint = Isometry3(translation= coxa_length * np.array([1., 0., 0.]))
    femur_tibia_joint = Isometry3(translation= femur_length * np.array([1., 0., 0.]))
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
    model.add_effector(
        parent=tibia,
        at=foot_transform
    )

    return model


def build_hexapod_model(coxa_length, femur_length, tibia_length):
    model = RigidBody()
    anchors = get_anchors()
    body_idx = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        inertia=body_intertia,
        joint=Free()
    )
    leg = build_hexapod_leg(coxa_length, femur_length, tibia_length)
    for anchor in anchors:
        model.add_body(leg, anchor, body_idx)
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
    lc_test = 50  # mm
    lf_test = 50
    lt_test = 100
    f_rest = 9800  # nmm
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

        problem = builder.build(backend='casadi')


e_z = np.array([0, 0, 1], dtype=float)


def get_anchors(distance_from_center=0.050):
    angles = [np.pi / 6, np.pi / 2, 5 * np.pi / 6, -5 * np.pi / 6, -np.pi / 2, -np.pi / 6]

    return [Isometry3(rotation=Rotation3(axis=e_z, angle=a))
            @ Isometry3(translation=np.array([distance_from_center, 0, 0]))
            for a in angles]


def main():
    angles = np.zeros(shape=(24,), dtype=float)

    #    plot_model((.050, 0.080, 0.090), angles)
    params = (.050, 0.080, 0.090)
    model = build_hexapod_model(*params)
    viz = KinematicsVisualiser(model, scale=0.02)
    viz.draw()
    plt.show()



def plot_model(params, angles, angle_rates=None):
    lines = []
    origin = angles[3:6]

    model = build_hexapod_model(*params)
    feet_idx = range(len(model.end_effectors))
    feet = model.forward_kinematics(angles)
    anchors = []
    for foot_idx, foot in zip(feet_idx, feet):
        joints = model.joint_locations(angles, foot_idx)

        joint_p = [p @ origin for p in joints]
        joint_p.append(foot @ origin)
        lines.append(joint_p)
        anchors.append(joint_p[1])

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
    line_array = 1000 * np.vstack(lines).T
    ax.scatter(line_array[0, :], line_array[1, :], line_array[2, :], 'ko')
    ax.plot(line_array[0, :], line_array[1, :], line_array[2, :])


def lag_iterator(iterable):
    n = len(iterable)
    for i in range(n):
        j = (i + 1) % n
        yield iterable[j], iterable[i]


def problem_1():
    # Objective -> min_q |ddot{q}(q, 0)|
    # Constraints
    # - h(center) > h_0
    # 1e-2 > z_i > 0 for all z_i
    # normal force at foot f_i >= 0
    #

    motor_max = 0.52  # Netwon millimeters
    a_g = np.array([0, 0, -9.8])
    with ProblemBuilder() as builder:
        femur_length = 0.08
        coxa_length = .05
        tibia_length = 0.09

        contact_forces = {
            i: np.concatenate([np.zeros(3,), builder.new_variable('f_z', shape=(3,))]) for i in range(6)
        }

        q_joints = builder.new_variable('q', shape=(18,))
        rest_height = builder.new_variable('h', initial_value=0.4)
        q = np.concatenate([np.array([0., 0., 0, 0, 0, 0]), q_joints])

        q_dot = np.zeros(shape=(24,), dtype=float)
        q_ddot = np.zeros(shape=(24,), dtype=float)

        model = build_hexapod_model(coxa_length, femur_length, tibia_length)
        tau = model.inverse_dynamics(q, q_dot, q_ddot, a_g, contact_forces)

        ones = np.ones(shape=(18,), dtype=float)

        cost = 0 #np.dot(tau[6:], tau[6:]) #+ 0.01 * rest_height
        feet = model.forward_kinematics(q)
        origin = np.array([0, 0, 0])
        e_z = np.array([0, 0, 1])
        alpha = 1000
        for a, b in lag_iterator(feet):
            f_a = a @ origin
            f_b = b @ origin
            u = np.cross(f_a, f_b)
            cost -= alpha * np.dot(u, u)

        constraints = [
                      f_z[6] > 0 for f_z in contact_forces.values()
                      ] + [
                          q_joints < ones * np.pi/4,
                          - ones * np.pi/4 < q_joints,
#                          tau[6:] < ones * motor_max,
#                          tau[6:] > - ones * motor_max,
#                          rest_height > 0.3
                      ] + [
           np.dot(e_z, foot @ origin) > -rest_height for foot in feet
        ] + [
            np.dot(e_z, foot @ origin) < 0 for foot in feet
        ]

        builder.constraints = constraints
        builder.objective = Minimise(cost)
        builder.outputs = [q, tau]

        solver = builder.build('casadi')

    [q_sol, tau_sol] = solver()
    print(q_sol)
    print(tau_sol)

    plot_model((coxa_length, femur_length, tibia_length), q_sol, q_dot)

def check_angles():
    femur_length = 0.08
    coxa_length = .05
    tibia_length = 0.09

    names = ['hip', 'knee', 'ankle']
    for i in range(3):
        for theta in np.linspace(-np.pi/4, np.pi/4, 4):

            q = np.zeros((24,))
            q_dot = np.zeros(shape=(24,))
            for k in range(6):
                q[6 + k * 3 + i] = theta
            print(f"{names[i]} @ {theta}")
            plot_model((coxa_length, femur_length, tibia_length), q, q_dot)


if __name__ == '__main__':
#    check_angles()
#    problem_1()
    main()