import numpy as np
import pytest

from coker.algebra.kernel import function, VectorSpace
from coker.toolkits.kinematics import RigidBody, Revolute, Inertia
from coker.toolkits.spatial import Rotation3, Isometry3, SE3Adjoint, Screw

# Test based on 3-link open-chain manipulator
# from Murry Et. Al

from test.util import is_close, validate_symbolic_call

g = -9.8
# Parameters from the book, r being center of mass, l being next joint
l_0 = 0.5
l_1 = 0.5
l_2 = 0.5
r_0 = l_0 / 2
r_1 = l_1 / 2
r_2 = l_2 / 2

m_0 = 1
m_1 = 1
m_2 = 1
m_3 = 1

Ix_0 = 1
Ix_1 = 1
Ix_2 = 1
Ix_3 = 1

Iy_0 = 1
Iy_1 = 1
Iy_2 = 1
Iy_3 = 1

Iz_0 = 1
Iz_1 = 1
Iz_2 = 1
Iz_3 = 1

e_x = np.array([1, 0, 0])
e_y = np.array([0, 1, 0])
e_z = np.array([0, 0, 1])


class Sampler:

    def __init__(self, n: int):
        self.dimensions = n
        self.rng = np.random.default_rng()

    def sample_positions(self, count):
        vel = np.zeros((self.dimensions,))
        acc = np.zeros((self.dimensions,))
        pos = np.zeros((self.dimensions,))
        yield pos, vel, acc
        for i in range(count):
            pos = self.rng.uniform(-1, 1, size=(self.dimensions,))
            yield pos, vel, acc

    def sample_position_and_velocity(self, count):
        vel = np.zeros((self.dimensions,))
        acc = np.zeros((self.dimensions,))
        pos = np.zeros((self.dimensions,))
        yield pos, vel, acc
        for i in range(count):
            pos = self.rng.uniform(-1, 1, size=(self.dimensions,))
            vel = self.rng.uniform(-1, 1, size=(self.dimensions,))
            yield pos, vel, acc

    def sample_position_velocity_and_accel(self, count):
        vel = np.zeros((self.dimensions,))
        acc = np.zeros((self.dimensions,))
        pos = np.zeros((self.dimensions,))
        yield pos, vel, acc
        for i in range(count):
            pos = self.rng.uniform(-1, 1, size=(self.dimensions,))
            vel = self.rng.uniform(-1, 1, size=(self.dimensions,))
            acc = self.rng.uniform(-1, 1, size=(self.dimensions,))
            yield pos, vel, acc


def test_single_pendulum(backend):
    model = RigidBody()
    # One link
    #
    # image x_axis as going into the scree
    #
    link = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw.from_tuple(1, 0, 0, 0, 0, 0)),
        inertia=Inertia(
            centre_of_mass=Isometry3(translation=np.array([0, 0, -r_0])),
            mass=1,
            moments=np.array([1, 0, 0, 1, 0, 1]),
        ),
    )

    tip = model.add_effector(
        parent=link, at=Isometry3(translation=np.array([0, 0, -l_0]))
    )
    origin = np.zeros((3,), dtype=float)
    transform_symbolic = function(
        [VectorSpace("q", 1)],
        implementation=lambda q: model.forward_kinematics(q)[0].apply(origin),
        backend=backend,
    )

    zero_angle = np.array([0], dtype=float)
    rest_transforms = model.forward_kinematics(zero_angle)
    rest_transform = rest_transforms[tip]

    assert np.allclose(
        rest_transform.apply(origin), np.array([0, 0, -l_0], dtype=float)
    )
    assert np.allclose(
        transform_symbolic(zero_angle), np.array([0, 0, -l_0], dtype=float)
    )

    full_left = np.array([np.pi / 2])
    rest_transforms = model.forward_kinematics(full_left)
    rest_transform = rest_transforms[tip]

    assert np.allclose(
        rest_transform.apply(origin), np.array([0, l_0, 0], dtype=float)
    )


def test_single_slider():
    model = RigidBody()
    link = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw.from_tuple(0, 0, 0, 0, 0, 1)),
        inertia=Inertia(
            centre_of_mass=Isometry3(translation=np.array([0, 0, r_0])),
            mass=1,
            moments=np.array([1, 0, 0, 1, 0, 1]),
        ),
    )

    tip = model.add_effector(
        parent=link, at=Isometry3(translation=np.array([0, 0, l_0]))
    )
    zero_angle = np.array([0], dtype=float)
    rest_transforms = model.forward_kinematics(zero_angle)
    rest_transform = rest_transforms[tip]
    origin = np.zeros((3,), dtype=float)

    assert np.allclose(
        rest_transform.apply(origin), np.array([0, 0, l_0], dtype=float)
    )

    full_left = np.array([1])
    rest_transforms = model.forward_kinematics(full_left)
    rest_transform = rest_transforms[tip]

    assert np.allclose(
        rest_transform.apply(origin), np.array([0, 0, l_0 + 1], dtype=float)
    )


def build_three_link_model():
    # using Murray Et. Al's approach

    zeta_1 = Screw.from_tuple(0, 0, 1, 0, 0, 0)
    zeta_2 = Screw.from_tuple(-1, 0, 0, 0, -l_0, 0)
    zeta_3 = Screw.from_tuple(-1, 0, 0, 0, -l_0, l_1)
    g_sl1 = Isometry3(translation=np.array([0, 0, r_0]))
    g_sl2 = Isometry3(translation=np.array([0, r_1, l_0]))
    g_sl3 = Isometry3(translation=np.array([0, l_1 + r_2, l_0]))
    g_effector = Isometry3(translation=np.array([0, l_2 + l_1, l_0]))

    com_1 = Isometry3(translation=np.array([0, 0, r_0]))

    zeta_1_relative = Screw.from_tuple(0, 0, 1, 0, 0, 0)
    transform_12 = Isometry3(translation=np.array([0, 0, l_0]))

    zeta_2_relative = Screw.from_tuple(-1, 0, 0, 0, 0, 0)
    transform_23 = Isometry3(translation=np.array([0, l_1, 0]))
    com_2 = Isometry3(translation=np.array([0, r_1, 0]))

    zeta_3_relative = Screw.from_tuple(-1, 0, 0, 0, 0, 0)
    com_3 = Isometry3(translation=np.array([0, r_2, 0]))
    end_effector = Isometry3(translation=np.array([0, l_2, 0]))

    t_final = end_effector @ transform_23 @ transform_12

    assert t_final == g_effector
    Ad_2 = SE3Adjoint(transform_12)
    Ad_3 = SE3Adjoint(transform_23 @ transform_12)

    assert zeta_2 == Ad_2(zeta_2_relative)
    assert zeta_3 == Ad_3(zeta_3_relative)
    assert g_sl1 == com_1
    assert g_sl2 == com_2 @ transform_12
    assert g_sl3 == com_3 @ transform_23 @ transform_12

    model = RigidBody()

    base = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(zeta_1_relative),
        inertia=Inertia(
            com_1, m_0, moments=np.array([Ix_0, 0, 0, Iy_0, 0, Iz_0])
        ),
    )

    link_1 = model.add_link(
        parent=base,
        at=transform_12,
        joint=Revolute(zeta_2_relative),
        inertia=Inertia(
            com_2, m_1, moments=np.array([Ix_1, 0, 0, Iy_1, 0, Iz_1])
        ),
    )

    link_2 = model.add_link(
        parent=link_1,
        at=transform_23,
        joint=Revolute(zeta_3_relative),
        inertia=Inertia(
            com_3, m_2, moments=np.array([Ix_2, 0, 0, Iy_2, 0, Iz_2])
        ),
    )

    effector = model.add_effector(parent=link_2, at=end_effector)

    assert model.total_joints() == 3
    return model


def test_pe(backend):
    model = build_three_link_model()

    def murray_V(angles):
        g = 9.8
        h_1 = r_0
        h_2 = l_0 - r_1 * np.sin(angles[1])
        h_3 = (
            l_0 - l_1 * np.sin(angles[1]) - r_2 * np.sin(angles[1] + angles[2])
        )

        return g * (h_1 + h_2 + h_3)

    q = np.zeros((3,))

    v = model.potential_energy(
        angles=q, gravity_vector=np.array([0, 0, -9.8]).T
    )

    v_test = murray_V(q)

    assert v == v_test


def build_scara_model():
    w_z = Screw.from_tuple(0, 0, 1, 0, 0, 0)
    e_z = Screw.from_tuple(0, 0, 0, 0, 0, 1)

    t_1 = Isometry3(translation=np.array([0, 0, l_0]))
    com_1 = Isometry3(translation=np.array([0, r_1, 0]))
    moments_1 = np.array([Ix_0, 0, 0, Iy_0, 0, Iz_0])

    t_2 = Isometry3(translation=np.array([0, l_1, 0]))
    com_2 = Isometry3(translation=np.array([0, r_2, 0]))
    moments_2 = np.array([Ix_1, 0, 0, Iy_1, 0, Iz_1])

    t_3 = Isometry3(translation=np.array([0, l_2, 0]))
    com_3 = Isometry3.identity()
    moments_3 = np.array([Ix_2, 0, 0, Iy_2, 0, Iz_2])

    t_4 = Isometry3.identity()
    com_4 = Isometry3.identity()
    moments_4 = np.array([Ix_3, 0, 0, Iy_3, 0, Iz_3])

    end_effector = Isometry3(translation=np.array([0, 0, 0]))

    t_final = end_effector @ t_4 @ t_3 @ t_2 @ t_1
    model = RigidBody()

    link_1 = model.add_link(
        parent=model.WORLD,
        at=t_1,
        joint=Revolute(w_z),
        inertia=Inertia(com_1, m_0, moments=moments_1),
    )

    link_2 = model.add_link(
        parent=link_1,
        at=t_2,
        joint=Revolute(w_z),
        inertia=Inertia(com_2, m_1, moments=moments_2),
    )

    link_3 = model.add_link(
        parent=link_2,
        at=t_3,
        joint=Revolute(w_z),
        inertia=Inertia(com_3, m_2, moments=moments_3),
    )
    link_4 = model.add_link(
        parent=link_3,
        at=t_4,
        joint=Revolute(e_z),
        inertia=Inertia(com_4, m_3, moments=moments_4),
    )

    effector = model.add_effector(parent=link_4, at=Isometry3.identity())
    return model, t_final


def test_scara_jacobian():
    def murray_spatial_jacobian(q):
        a_1 = l_1 * np.cos(q[0])
        a_2 = l_1 * np.sin(q[0])
        b_1 = a_1 + l_2 * np.cos(q[0] + q[1])
        b_2 = a_2 + l_2 * np.sin(q[0] + q[1])
        return np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [0, a_1, b_1, 0],
                [0, a_2, b_2, 0],
                [0, 0, 0, 1],
            ]
        )

    def scara_end_effector(q):
        cs = np.cos(q[0] + q[1] + q[2])
        ss = np.sin(q[0] + q[1] + q[2])
        px = -l_1 * np.sin(q[0]) - l_2 * np.sin(q[0] + q[1])
        py = l_1 * np.cos(q[0]) + l_2 * np.cos(q[0] + q[1])
        pz = l_0 + q[3]
        g_st = np.array(
            [[cs, -ss, 0, px], [ss, cs, 0, py], [0, 0, 1, pz], [0, 0, 0, 1]]
        )
        return g_st

    scara_model, t_final = build_scara_model()
    test_angles = [
        np.zeros((4,)),
        np.array([np.pi, 0, 0, 0]),
        np.array([0, np.pi / 4, 0, 0]),
        np.array([0, 0, np.pi / 4, 0]),
        np.array([0, 0, 0, 0.25]),
        np.array([np.pi / 4, np.pi / 2, 0, 0]),
        np.array([np.pi / 4, np.pi / 2, np.pi / 2, 0]),
    ]
    o = np.zeros((3,), dtype=float)

    for angles in test_angles:
        (body_tool_transform,) = scara_model.forward_kinematics(angles)
        expected_matrix = scara_end_effector(angles)
        p_t = body_tool_transform.apply(o)
        p_e = expected_matrix[0:3, 3].reshape((3,))
        assert np.allclose(p_t, p_e), f"Tool points off at {angles}"

        m = body_tool_transform.as_matrix()

        assert np.allclose(expected_matrix, m), f"Failed on {angles}"

    for angles in test_angles:
        (jacobian,) = scara_model.spatial_manipulator_jacobian(angles)
        expected_jacobian = murray_spatial_jacobian(angles)
        assert np.allclose(
            jacobian, expected_jacobian
        ), f"Incorrect Jacobian on {angles}"

    for angles in test_angles:
        (body_tool_transform,) = scara_model.forward_kinematics(angles)
        ad = SE3Adjoint(body_tool_transform)
        (body_jacobian,) = scara_model.body_manipulator_jacobian(angles)
        jacobian = ad @ body_jacobian
        expected_jacobian = murray_spatial_jacobian(angles)
        assert np.allclose(
            jacobian, expected_jacobian
        ), f"Incorrect Jacobian on {angles}"


def test_three_link_inverse_dynamics():
    def get_jacobians(q):

        swap = np.block(
            [[np.zeros((3, 3)), np.eye(3)], [np.eye(3), np.zeros((3, 3))]]
        )
        j_1 = np.zeros((6, 3), dtype=float)
        j_1[5, 0] = 1

        j_2 = np.zeros((6, 3), dtype=float)
        j_2[0, 0] = -r_0 * np.cos(q[1])
        j_2[2, 1] = -r_0
        j_2[3, 1] = -1
        j_2[4, 0] = -np.sin(q[1])
        j_2[5, 0] = np.cos(q[1])

        j_3 = np.zeros((6, 3), dtype=float)
        j_3[
            0,
            0,
        ] = -l_1 * np.cos(
            q[1]
        ) - r_1 * np.cos(q[1] + q[2])
        j_3[1, 1] = l_1 * np.sin(q[2])
        j_3[2, 1] = -r_1 - l_0 * np.cos(q[2])
        j_3[2, 2] = -r_1
        j_3[3, 1] = -1
        j_3[3, 2] = -1
        j_3[4, 0] = -np.sin(q[1] + q[2])
        j_3[5, 0] = np.cos(q[1] + q[2])

        return [swap @ j_1, swap @ j_2, swap @ j_3]

    def three_link_inverse_dynamics_mass_matrix(q):
        s1 = np.sin(q[1])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        s12 = np.sin(q[1] + q[2])
        c12 = np.cos(q[1] + q[2])
        M = np.zeros((3, 3), dtype=float)
        M[0, 0] = (
            Iy_1 * s1**2
            + Iy_2 * s12**2
            + Iz_0
            + Iz_1 * c1**2
            + Iz_2 * c12**2
            + m_1 * r_0**2 * c1**2
            + m_2 * (l_0 * c1 + r_1 * c12) ** 2
        )
        M[1, 1] = (
            Ix_1
            + Ix_2
            + m_2 * l_0**2
            + m_1 * r_0**2
            + m_2 * r_1**2
            + 2 * m_2 * l_0 * r_1 * c2
        )
        M[1, 2] = Ix_2 + m_2 * r_1**2 + m_2 * l_0 * r_1 * c2
        M[2, 1] = M[1, 2]
        M[2, 2] = Ix_2 + m_2 * r_1**2

        return M

    model = build_three_link_model()
    q_tests = [
        np.zeros((3,)),
        np.array([1, 0, 0], dtype=float),
        np.array([0, 1, 0], dtype=float),
        np.array([0, 0, 1], dtype=float),
        np.array([1, 1, 1], dtype=float),
    ]

    screws = [
        Screw.from_tuple(0, 0, 1, 0, 0, 0),
        Screw.from_tuple(-1, 0, 0, 0, -l_0, 0),
        Screw.from_tuple(-1, 0, 0, 0, -l_0, l_1),
    ]
    model_screws = model._get_joint_global_bases()

    for i, (expected, test) in enumerate(zip(screws, model_screws)):
        assert np.allclose(
            expected.to_array(), test.to_array()
        ), f"Screw {i} is not correct.\n Expected {expected}\n got {test}"

    for q_test in q_tests:

        test_jacobians = model._get_link_com_jacobians(q_test)
        jacobians = get_jacobians(q_test)

        for i, (t_j, j) in enumerate(zip(test_jacobians, jacobians)):
            assert np.allclose(
                t_j, j
            ), f"Failed on jacobian {i} for test value {q_test}\n Expected jacobian: {j}\n Got {t_j}"

        m_exact = three_link_inverse_dynamics_mass_matrix(q_test)
        m_test = model.mass_matrix(q_test)

        assert np.allclose(
            m_exact, m_test
        ), f"Failed on mass matrix for test value {q_test}"


def test_single_pendulum_dynamics():
    # rest position is hanging down.
    g = 9.8
    g_vec = g * np.array([0, 0, -1])
    model = RigidBody()
    inertia = Inertia(
        centre_of_mass=Isometry3(translation=np.array([0, 0, -l_0 / 2])),
        moments=np.array([Ix_0, 0, 0, Iy_0, 0, Iz_0]),
        mass=m_0,
    )
    w_vec = np.array([0, 1, 0, 0, 0, 0])
    model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw.from_array(w_vec)),
        inertia=inertia,
    )

    def potential_energy(q):
        return -np.cos(q[0]) * g * m_0 * l_0 / 2

    def kinetic_energy(q, qdot):
        linear = 0.5 * inertia.mass * (l_0 / 2) ** 2 * qdot[0] ** 2
        angular = (
            0.5
            * qdot[0] ** 2
            * np.dot(w_vec, np.dot(inertia.as_matrix(), w_vec))
        )
        return linear + angular

    iz = np.dot(w_vec, np.dot(inertia.as_matrix(), w_vec))
    assert Iz_0 == iz

    def torque(q, qdot, qddot):
        m = inertia.mass * (l_0 / 2) ** 2 + iz
        c = 0
        n = np.sin(q[0]) * g * m_0 * l_0 / 2

        return m * qddot + c * qdot + n

    test_scenarios = [
        (np.zeros((1,)), np.zeros((1,)), np.zeros((1,))),
        (np.array([np.pi / 4]), np.zeros((1,)), np.zeros((1,))),
        (-np.array([np.pi / 4]), np.zeros((1,)), np.zeros((1,))),
    ] + list(Sampler(1).sample_position_velocity_and_accel(10))

    for q, dq, ddq in test_scenarios:
        pe_test = model.potential_energy(q, g_vec)
        pe = potential_energy(q)
        assert np.allclose(pe_test, pe)

        ke_test = 0.5 * dq[0] ** 2 * model.mass_matrix(q)
        ke = kinetic_energy(q, dq)
        assert np.allclose(ke_test, ke)

        tau = torque(q, dq, ddq)
        tau_test = model.inverse_dynamics(q, dq, ddq, g_vec)
        assert np.allclose(tau_test, tau)


def build_double_pendulum():
    model = RigidBody()
    #   x - > lr, z-> up down
    #
    #   x   <- Base
    #   |   <- Link 1, COM 1/2 way down
    #   .   <- Revolute Joint
    #   |   <- Link 2, COM at tip
    #  ( )

    link_1_length = 1
    link_2_length = 1
    moment_1 = 1
    moment_2 = 1
    com_1 = Isometry3(translation=np.array([0, 0, -link_1_length / 2]))
    mass_1 = 1
    moments_1 = np.array([1, 0, 0, moment_1, 0, 1])

    com_2 = Isometry3(translation=np.array([0, 0, -link_2_length / 2]))
    mass_2 = 1
    moments_2 = np.array([1, 0, 0, moment_2, 0, 1])
    g = 9.8

    def double_pendulum_dynamics(q, qdot):
        alpha = (
            moment_1
            + moment_2
            + mass_1 * (link_1_length / 2) ** 2
            + mass_2 * (link_1_length**2 + (link_2_length / 2) ** 2)
        )
        beta = mass_2 * link_1_length * link_2_length / 2
        delta = moment_2 + mass_2 * (link_2_length / 2) ** 2
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        m = np.array(
            [
                [alpha + 2 * beta * c2, delta + beta * c2],
                [delta + beta * c2, delta],
            ]
        )
        c = np.array(
            [
                [-beta * s2 * qdot[1], -beta * s2 * (qdot[0] + qdot[1])],
                [beta * s2 * qdot[0], 0],
            ]
        )

        # pe = -m_1g * z_1  -m_2 g * z_2
        # z_1 = - l_1/2 cos(q)
        # z_2 = l1 cos(q) - l_2/2 cos(q_1 + q_2)
        #
        # pe = -g ( m_1l_1/2 + m_2l1) cos(q_1) - g cos(q_1+ q_2) l_2/2
        #    = -gamma cos(q_1) - kappa cos(q_1+q_2)
        # d{pe} = + gamma sin(q_1) +kappa sin(q_1 + q_2), - kappa sin( q_1+ q_2)
        gamma = g * (mass_1 * link_1_length / 2 + mass_1 * link_1_length)
        kappa = g * (mass_2 * link_2_length / 2)
        n = np.array(
            [
                [
                    gamma * np.sin(q[0]) + kappa * np.sin(q[0] + q[1]),
                    kappa * np.sin(q[0] + q[1]),
                ]
            ]
        )

        pe = -gamma * np.cos(q[0]) - kappa * np.cos(q[0] + q[1])
        return m, c, n, pe

    link_1 = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw(rotation=np.array([0, -1, 0]))),
        inertia=Inertia(centre_of_mass=com_1, mass=mass_1, moments=moments_1),
    )

    link_2 = model.add_link(
        parent=link_1,
        at=Isometry3(translation=np.array([0, 0, -link_1_length])),
        joint=Revolute(Screw(rotation=np.array([0, -1, 0]))),
        inertia=Inertia(centre_of_mass=com_2, mass=mass_2, moments=moments_2),
    )
    effector = model.add_effector(
        link_2, Isometry3(translation=np.array([0, 0, -link_2_length]))
    )

    return model, double_pendulum_dynamics


def test_double_pendulum(backend):
    model, dynamics = build_double_pendulum()
    g = np.array([0, 0, -9.8])

    def lagrange_impl(q, dq):
        m = model.mass_matrix(q)
        v = model.potential_energy(q, g)

        return 0.5 * np.dot(dq, m @ dq) - v

    f = function(
        [VectorSpace("q", 2), VectorSpace("dq", 2)], lagrange_impl, backend
    )
    test_scenarios = (
        [
            (np.array([0, np.pi / 2]), np.zeros((2,)), np.zeros((2,))),
            (np.array([0, np.pi / 2]), np.zeros((2,)), np.zeros((2,))),
            (np.array([np.pi / 2, 0]), np.zeros((2,)), np.zeros((2,))),
            (np.array([np.pi / 6, np.pi / 6]), np.zeros((2,)), np.zeros((2,))),
        ]
        + list(Sampler(2).sample_position_and_velocity(5))
        + list(Sampler(2).sample_position_velocity_and_accel(5))
    )

    for q, dq, ddq in test_scenarios:
        M, C, N, pe = dynamics(q, dq)
        mass_test = model.mass_matrix(q)
        assert np.allclose(mass_test, M)

        pe_test = model.potential_energy(q, g)

        assert np.allclose(pe_test, pe)

        tau_test = model.inverse_dynamics(q, dq, ddq, np.array([0, 0, -9.8]))

        tau = M @ ddq + C @ dq + N

        assert np.allclose(tau_test, tau)


def test_scara_inverse_dynamics():
    def scara_inverse_dynamics_matricies(q, qdot):
        alpha = Iz_0 + m_0 * r_0**2 + (m_1 + m_2 + m_3) * l_0**2
        beta = Iz_1 + Iz_2 + Iz_3 + (m_2 + m_3) * l_1**2 + m_1 * r_1**2
        gamma = l_0 * l_1 * (m_2 + m_3) + l_0 * m_1 * r_1
        delta = Iz_2 + Iz_3
        c1 = np.cos(q[1])
        s1 = np.sin(q[1])
        M = np.array(
            [
                [alpha + beta + 2 * gamma * c1, beta + gamma * c1, delta, 0],
                [beta + gamma * c1, beta, delta, 0],
                [delta, delta, delta, 0],
                [0, 0, 0, m_3],
            ]
        )
        C = np.array(
            [
                [
                    -gamma * s1 * qdot[1],
                    -gamma * s1 * (qdot[0] + qdot[1]),
                    0,
                    0,
                ],
                [gamma * s1 * qdot[0], 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        N = np.array([[0, 0, 0, m_3 * 9.8]])

        return M, C, N

    def scara_inverse_dynamics(q, q_dot, q_ddot):
        M, C, N = scara_inverse_dynamics_matricies(q, q_dot)
        return M @ q_ddot + C @ q_dot + N

    model, _ = build_scara_model()

    def sampler(vel=True, acc=True):

        test_configurations = [
            np.zeros((4,)),
            np.array([1, 0, 0, 0]),
            np.array([0, 1, 0, 0]),
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 0, 1]),
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 0, 1]),
            np.array([0, 1, 0, 1]),
            np.array([0, 0, 1, 1]),
            np.array([1, 1, 1, 0]),
            np.array([1, 1, 0, 1]),
            np.array([1, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
        ]
        for t in test_configurations:
            yield t, np.zeros((4,)), np.zeros((4,))

            r_1 = np.random.default_rng().uniform(-1, 1, size=(4,))
            r_2 = np.random.default_rng().uniform(-1, 1, size=(4,))
            if vel:
                yield t, r_1, np.zeros((4,))
            if acc:
                yield t, np.zeros((4,)), r_2
            if vel and acc:
                yield t, r_1, r_2

    for q, q_dot, q_ddot in sampler():
        M, C, N = scara_inverse_dynamics_matricies(q, q_dot)

        m_test = model.mass_matrix(q)
        assert np.allclose(m_test, M)
        test_torque = model.inverse_dynamics(
            q, q_dot, q_ddot, np.array([0, 0, g])
        )

        m_qdot = test_torque - C @ q_dot - N
        m_qdot_expected = M @ q_ddot
        assert np.allclose(
            m_qdot, m_qdot_expected
        ), f"Failed on mass matrix for test value {q}, {q_dot}, {q_ddot}"

        exact_torque = scara_inverse_dynamics(q, q_dot, q_ddot)
        assert np.allclose(exact_torque, test_torque), (
            f"Failed on torque comparison for test value {q}, {q_dot, q_ddot}\n"
            f"Expected {exact_torque} but got {test_torque}"
        )


def build_elbow_model(base_rotation=0):
    body = RigidBody()
    shoulder_z = body.add_link(
        body.WORLD,
        Isometry3(
            translation=np.array([0, 0, l_0]),
            rotation=Rotation3(axis=np.array([0, 0, 1]), angle=base_rotation),
        ),
        Revolute(Screw.w_z()),
        inertia=Inertia(
            centre_of_mass=Isometry3(translation=np.array([0, 0, l_0 / 2])),
            mass=1,
            moments=np.array([1, 0, 0, 1, 0, 1]),
        ),
    )
    shoulder_x = body.add_link(
        shoulder_z,
        Isometry3.identity(),
        Revolute(-Screw.w_x()),
        inertia=Inertia.zero(),
    )
    elbow_x = body.add_link(
        shoulder_x,
        Isometry3(translation=np.array([0, l_1, 0])),
        Revolute(-Screw.w_x()),
        inertia=Inertia(
            centre_of_mass=Isometry3(translation=np.array([0, l_1 / 2, 0])),
            mass=1,
            moments=np.array([1, 0, 0, 1, 0]),
        ),
    )
    wrist_z = body.add_link(
        elbow_x,
        Isometry3(translation=np.array([0, l_2, 0])),
        Revolute(Screw.w_z()),
        inertia=Inertia(
            centre_of_mass=Isometry3(translation=np.array([0, l_2 / 2, 0])),
            mass=1,
            moments=np.array([1, 0, 0, 1, 0]),
        ),
    )
    wrist_x = body.add_link(
        wrist_z,
        Isometry3.identity(),
        Revolute(-Screw.w_x()),
        inertia=Inertia.zero(),
    )
    wrist_y = body.add_link(
        wrist_x,
        Isometry3.identity(),
        Revolute(Screw.w_y()),
        inertia=Inertia.zero(),
    )
    effector = body.add_effector(wrist_y, Isometry3.identity())

    return body


def test_elbow_manipulator():

    def tool_position(q, base_rotation=0):
        return np.array(
            [
                -np.sin(q[0] + base_rotation)
                * (l_1 * np.cos(q[1]) + l_2 * np.cos(q[1] + q[2])),
                np.cos(q[0] + base_rotation)
                * (l_1 * np.cos(q[1]) + l_2 * np.cos(q[1] + q[2])),
                l_0 - l_1 * np.sin(q[1]) - l_2 * np.sin(q[1] + q[2]),
            ]
        )

    expected_screws = [
        Screw.from_tuple(0, 0, 1, 0, 0, 0),
        Screw.from_tuple(-1, 0, 0, 0, -l_0, 0),
        Screw.from_tuple(-1, 0, 0, 0, -l_0, l_1),
        Screw.from_tuple(0, 0, 1, l_1 + l_2, 0, 0),
        Screw.from_tuple(-1, 0, 0, 0, -l_0, l_1 + l_2),
        Screw.from_tuple(0, 1, 0, -l_0, 0, 0),
    ]
    model = build_elbow_model()
    for i, (expected_screw, actual_screw) in enumerate(
        zip(expected_screws, model._get_joint_global_bases())
    ):
        rel, tes = expected_screw.to_array(), actual_screw.to_array()
        assert np.allclose(rel, tes), f"Zeta {i +1}: expected {rel}, got {tes}"

    test_configurations = [
        np.zeros((6,)),
        np.array([0.5, 0, 0, 0, 0, 0]) * np.pi,
        np.array([0, 0.5, 0, 0, 0, 0]) * np.pi,
    ]

    model = build_elbow_model(base_rotation=0)
    for i, q_i in enumerate(test_configurations):
        (T_i,) = model.forward_kinematics(q_i)
        p_test = T_i.apply(np.zeros((3,)))
        p = tool_position(q_i, base_rotation=0)
        assert np.allclose(
            p_test, p
        ), f"Failed on config {i} with args :{q_i}\n. Expected {p_test}, got {p}"


def test_hexapod_leg(backend):
    from numpy import sin, cos
    from coker.toolkits.spatial import Rotation3

    def build_hexapod_leg(hip_anchor_distance, hip_anchor_angle):
        model = RigidBody()
        coxa_length = 0.05
        femur_length = 0.08
        tibia_length = 0.16
        coxa_intertia = Inertia(
            centre_of_mass=Isometry3(translation=np.array([0.007, 0, 0])),
            mass=0.128,
            moments=np.array(
                [
                    1.096e-05,
                    -2.510e-09,
                    -3.152e-08,
                    1.915e-05,
                    -2.755e-09,
                    1.712e-05,
                ]
            ),
        )

        femur_inertia = Inertia(
            centre_of_mass=Isometry3(translation=np.array([0.04, 0.0, 0.0])),
            mass=0.251,
            moments=np.array(
                [2.067e-05, -3.782e-10, 2.992e-08, 0.0, 0.0, 0.0]
            ),
        )

        tibia_inertia = Inertia(
            centre_of_mass=Isometry3(translation=np.array([0.035, 0, -0.027])),
            mass=0.008,
            moments=np.array(
                [
                    1.679e-05,
                    4.209e-09,
                    5.611e-07,
                    1.527e-05,
                    -8.506e-09,
                    2.889e-06,
                ]
            ),
        )

        body_intertia = Inertia(
            centre_of_mass=Isometry3.identity(),
            mass=0.0001,
            moments=1e-9 * np.array([1, 0, 0, 1, 0, 1]),
        )

        foot_angle_deg = 12.5 - 90

        hip_anchor = Isometry3(
            rotation=Rotation3(axis=e_z, angle=hip_anchor_angle)
        ) @ Isometry3(translation=np.array([hip_anchor_distance, 0, 0]))

        coxa = model.add_link(
            parent=model.WORLD,
            at=hip_anchor,
            joint=Revolute(Screw.w_z()),
            inertia=coxa_intertia,
        )

        femur_coxa_joint = Isometry3(
            translation=coxa_length * np.array([1.0, 0.0, 0.0])
        )
        femur_tibia_joint = Isometry3(
            translation=femur_length * np.array([1.0, 0.0, 0.0])
        )
        v = tibia_length * np.array(
            [
                np.cos(np.deg2rad(foot_angle_deg)),
                0.0,
                np.sin(np.deg2rad(foot_angle_deg)),
            ]
        )
        foot_transform = Isometry3(translation=v)

        femur = model.add_link(
            parent=coxa,
            at=femur_coxa_joint,
            joint=Revolute(-Screw.w_y()),
            inertia=femur_inertia,
        )

        tibia = model.add_link(
            parent=femur,
            at=femur_tibia_joint,
            joint=Revolute(-Screw.w_y()),
            inertia=tibia_inertia,
        )
        model.add_effector(parent=tibia, at=foot_transform)

        return model

    def exact_fk(q, hip_distance, hip_angle):
        # derived from sympy model
        [theta_h, theta_k, theta_a] = q.flatten().tolist()
        a11 = hip_distance + (
            0.08 * cos(theta_k)
            + 0.16 * cos(theta_a + theta_k - 1.35263017029561)
            + 0.05
        ) * cos(theta_h)
        a12 = (
            0.08 * cos(theta_k)
            + 0.16 * cos(theta_a + theta_k - 1.35263017029561)
            + 0.05
        ) * sin(theta_h)
        a13 = 0.08 * sin(theta_k) + 0.16 * sin(
            theta_a + theta_k - 1.35263017029561
        )
        point = np.array([a11, a12, a13])
        r = Rotation3(axis=e_z, angle=hip_angle)

        return r.apply(point)

    def exact_xform(q, hip_distance, hip_angle):
        [theta_h, theta_k, theta_a] = q.flatten().tolist()
        coxa_length = 0.05
        femur_length = 0.08
        tibia_length = 0.16

        foot_angle_deg = 12.5 - 90
        v = tibia_length * np.array(
            [
                np.cos(np.deg2rad(foot_angle_deg)),
                0.0,
                np.sin(np.deg2rad(foot_angle_deg)),
            ]
        )
        i1 = Isometry3(rotation=Rotation3(axis=e_z, angle=hip_angle))
        i2 = i1 @ Isometry3(translation=np.array([hip_distance, 0, 0]))
        i3 = i2 @ Isometry3(rotation=Rotation3(e_z, angle=theta_h))
        i4 = i3 @ Isometry3(translation=np.array([coxa_length, 0, 0]))
        i5 = i4 @ Isometry3(rotation=Rotation3(-e_y, angle=theta_k))
        i6 = i5 @ Isometry3(translation=np.array([femur_length, 0, 0]))
        i7 = i6 @ Isometry3(rotation=Rotation3(-e_y, angle=theta_a))
        i8 = i7 @ Isometry3(translation=v)
        return i8

    test_configurations = [
        (0, 0),
        (0.11, 0),
        (0, np.pi / 2),
        (0.11, np.pi / 2),
    ]

    test_values = [
        np.array([0, 0, 0], dtype=float),
        np.array([np.pi / 4, 0, 0], dtype=float),
        np.array([-np.pi / 4, 0, 0], dtype=float),
        np.array([0, np.pi / 4, 0], dtype=float),
        np.array([0, -np.pi / 4, 0], dtype=float),
        np.array([0, 0, -np.pi / 4], dtype=float),
    ]

    for i, (hip_distance_i, hip_angle_i) in enumerate(test_configurations):
        leg_model = build_hexapod_leg(hip_distance_i, hip_angle_i)

        def impl(q):
            (tx,) = leg_model.forward_kinematics(q)
            return tx @ np.array([0, 0, 0])

        symbolic_fk = function(
            arguments=[VectorSpace("q", 3)],
            implementation=impl,
            backend=backend,
        )

        for test_value in test_values:
            (fk,) = leg_model.forward_kinematics(test_value)
            xform = exact_xform(test_value, hip_distance_i, hip_angle_i)
            assert is_close(
                xform, fk
            ), f"For config {i} angles {test_value};\n     Expected: {xform}\n     but got: {fk}\n"
            point = xform @ np.array([0, 0, 0])
            soln = exact_fk(test_value, hip_distance_i, hip_angle_i)

            assert np.allclose(
                point, soln
            ), f"For config {i} angles {test_value};\n     Expected: {soln}\n     but got: {point}\n"

            symbolic_result = symbolic_fk(test_value)

            assert np.allclose(symbolic_result, soln, atol=1e-6)
