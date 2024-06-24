from coker.algebra.kernel import kernel, VectorSpace
from coker.toolkits.kinematics import *
# Test based on 3-link open-chain manipulator
# from Murry Et. Al

from test.util import is_close

g = - 9.8
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

class Sampler:

    def __init__(self, n: int):
        self.dimensions = n
        self.rng = np.random.default_rng()

    def sample_positions(self, count):
        vel = np.zeros((self.dimensions, ))
        acc = np.zeros((self.dimensions, ))
        pos = np.zeros((self.dimensions, ))
        yield pos, vel, acc
        for i in range(count):
            pos = self.rng.uniform(-1,1,size=(self.dimensions,))
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

def test_single_pendulum():
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
            moments=np.array([1, 0, 0, 1, 0, 1])
        ),
    )

    tip = model.add_effector(parent=link, at=Isometry3(translation=np.array([0, 0, -l_0])))

    zero_angle = np.array([0], dtype=float)
    rest_transforms = model.forward_kinematics(zero_angle)
    rest_transform = rest_transforms[tip]
    origin = np.zeros((3,), dtype=float)

    assert np.allclose(rest_transform.apply(origin), np.array([0, 0, -l_0], dtype=float))

    full_left = np.array([np.pi / 2])
    rest_transforms = model.forward_kinematics(full_left)
    rest_transform = rest_transforms[tip]

    assert np.allclose(rest_transform.apply(origin), np.array([0, l_0, 0], dtype=float))


def test_single_slider():
    model = RigidBody()
    link = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw.from_tuple(0, 0, 0, 0, 0, 1)),
        inertia=Inertia(
            centre_of_mass=Isometry3(translation=np.array([0, 0, r_0])),
            mass=1,
            moments=np.array([1, 0, 0, 1, 0, 1])
        ),
    )

    tip = model.add_effector(parent=link, at=Isometry3(translation=np.array([0, 0, l_0])))
    zero_angle = np.array([0], dtype=float)
    rest_transforms = model.forward_kinematics(zero_angle)
    rest_transform = rest_transforms[tip]
    origin = np.zeros((3,), dtype=float)

    assert np.allclose(rest_transform.apply(origin), np.array([0, 0, l_0], dtype=float))

    full_left = np.array([1])
    rest_transforms = model.forward_kinematics(full_left)
    rest_transform = rest_transforms[tip]

    assert np.allclose(rest_transform.apply(origin), np.array([0, 0, l_0 + 1], dtype=float))


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
        inertia=Inertia(com_1, m_0, moments=np.array([Ix_0, 0, 0, Iy_0, 0, Iz_0])),
    )

    link_1 = model.add_link(
        parent=base,
        at=transform_12,
        joint=Revolute(zeta_2_relative),
        inertia=Inertia(com_2, m_1, moments=np.array([Ix_1, 0, 0, Iy_1, 0, Iz_1]))
    )

    link_2 = model.add_link(
        parent=link_1,
        at=transform_23,
        joint=Revolute(zeta_3_relative),
        inertia=Inertia(com_3, m_2, moments=np.array([Ix_2, 0, 0, Iy_2, 0, Iz_2]))
    )

    effector = model.add_effector(
        parent=link_2,
        at=end_effector
    )

    assert model.total_joints() == 3
    return model


def test_pe(backend):
    model = build_three_link_model()

    def murray_V(angles):
        g = 9.8
        h_1 = r_0
        h_2 = l_0 - r_1 * np.sin(angles[1])
        h_3 = l_0 - l_1 * np.sin(angles[1]) - r_2 * np.sin(angles[1] + angles[2])

        return g * (h_1 + h_2 + h_3)

    q = np.zeros((3,))

    v = model.potential_energy(
        angles=q,
        gravity_vector=np.array([0, 0, -9.8]).T
    )

    v_test = murray_V(q)

    assert v == v_test

    v_kernel = kernel(
        [VectorSpace('q', 3)],
        lambda a: model.potential_energy(angles=a, gravity_vector=np.array([0, 0, -9.8]).T),
        backend
    )

    assert v_kernel(q) == v


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
        inertia=Inertia(com_2, m_1, moments=moments_2)
    )

    link_3 = model.add_link(
        parent=link_2,
        at=t_3,
        joint=Revolute(w_z),
        inertia=Inertia(com_3, m_2, moments=moments_3)
    )
    link_4 = model.add_link(
        parent=link_3,
        at=t_4,
        joint=Revolute(e_z),
        inertia=Inertia(com_4, m_3, moments=moments_4)
    )

    effector = model.add_effector(
        parent=link_4,
        at=Isometry3.identity()
    )
    return model, t_final


def test_scara_jacobian():
    def murray_spatial_jacobian(q):
        a_1 = l_1 * np.cos(q[0])
        a_2 = l_1 * np.sin(q[0])
        b_1 = a_1 + l_2 * np.cos(q[0] + q[1])
        b_2 = a_2 + l_2 * np.sin(q[0] + q[1])
        return np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, a_1, b_1, 0],
            [0, a_2, b_2, 0],
            [0, 0, 0, 1]
        ])

    def scara_end_effector(q):
        cs = np.cos(q[0] + q[1] + q[2])
        ss = np.sin(q[0] + q[1] + q[2])
        px = -l_1 * np.sin(q[0]) - l_2 * np.sin(q[0] + q[1])
        py = l_1 * np.cos(q[0]) + l_2 * np.cos(q[0] + q[1])
        pz = l_0 + q[3]
        g_st = np.array([
            [cs, -ss, 0, px],
            [ss, cs, 0, py],
            [0, 0, 1, pz],
            [0, 0, 0, 1]
        ])
        return g_st

    scara_model, t_final = build_scara_model()
    test_angles = [
        np.zeros((4,)),
        np.array([np.pi, 0, 0, 0]),
        np.array([0, np.pi / 4, 0, 0]),
        np.array([0, 0, np.pi / 4, 0]),
        np.array([0, 0, 0, 0.25])
    ]
    o = np.zeros((3,), dtype=float)

    for angles in test_angles:
        body_tool_transform, = scara_model.forward_kinematics(angles)
        expected_matrix = scara_end_effector(angles)
        p_t = body_tool_transform.apply(o)
        p_e = expected_matrix[0:3, 3].reshape((3,))
        assert np.allclose(p_t, p_e), f"Tool points off at {angles}"

        m = body_tool_transform.as_matrix()

        assert np.allclose(expected_matrix, m), f"Failed on {angles}"

    for angles in test_angles:
        jacobian = scara_model.spatial_manipulator_jacobian(angles)
        expected_jacobian = murray_spatial_jacobian(angles)
        assert np.allclose(jacobian, expected_jacobian), f"Incorrect Jacobian on {angles}"

    for angles in test_angles:
        body_tool_transform, = scara_model.forward_kinematics(angles)
        ad = SE3Adjoint(body_tool_transform)
        body_jacobian = scara_model.body_manipulator_jacobian(angles)
        jacobian = ad @ body_jacobian
        expected_jacobian = murray_spatial_jacobian(angles)
        assert np.allclose(jacobian, expected_jacobian), f"Incorrect Jacobian on {angles}"


def test_three_link_inverse_dynamics():
    def get_jacobians(q):

        swap = np.block(
            [[np.zeros((3,3)), np.eye(3)],
             [np.eye(3), np.zeros((3,3))]]
        )
        j_1 = np.zeros((6,3), dtype=float)

        j_1[5, 0] = 1
        j_2 = np.zeros((6,3), dtype=float)
        j_2[0, 0] = -r_0 * np.cos(q[1])
        j_2[2, 1]= -r_0
        j_2[3,1] = -1
        j_2[4,0] = -np.sin(q[1])
        j_2[5,0] = np.cos(q[1])

        j_3 = np.zeros((6,3), dtype=float)
        j_3[0,0,] = -l_0*np.cos(q[1]) - r_1 *np.cos(q[1] + q[2])
        j_3[1,1] = l_0 * np.sin(q[2])
        j_3[2,1] = -r_1 -l_0*np.cos(q[2])
        j_3[2,2] = -r_1
        j_3[3,1] = -1
        j_3[3,2] = -1
        j_3[4,0] = -np.sin(q[1] + q[2])
        j_3[5, 0] = np.cos(q[1] + q[2])

        return [swap @ j_1, swap @ j_2, swap @ j_3]

    def three_link_inverse_dynamics_mass_matrix(q):
        s1 = np.sin(q[1])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        s12 = np.sin(q[1] + q[2])
        c12 = np.cos(q[1] + q[2])
        M = np.zeros((3, 3), dtype=float)
        M[0, 0] = Iy_1 * s1**2 + Iy_2 * s12**2 + Iz_0 + Iz_1 * c1**2 + Iz_2 *c12**2 + m_1 *r_0**2*c1**2 + m_2*(l_0*c1 + r_1*c12)**2
        M[1, 1] = Ix_1 + Ix_2 + m_2*l_0**2 + m_1*r_0**2 + m_2* r_1**2 + 2*  m_2*l_0*r_1 * c2
        M[1, 2] = Ix_2 + m_2 * r_1**2 + m_2*l_0*r_1*c2
        M[2, 1] = M[1, 2]
        M[2, 2] = Ix_2 + m_2 * r_1**2

        return M

    model = build_three_link_model()
    q_tests = [
        np.zeros((3,)),
        np.array([1,0,0],dtype=float),
        np.array([0, 1, 0], dtype=float),
        np.array([0, 0, 1], dtype=float),
    ]


    for q_test in q_tests:
        test_jacobians = model._get_link_com_jacobians(q_test)
        jacobians = get_jacobians(q_test)

        for i, (t_j, j) in enumerate(zip(test_jacobians, jacobians)):
            assert np.allclose(t_j, j), f"Failed on jacobian {i} for test value {q_test}"

        m_exact = three_link_inverse_dynamics_mass_matrix(q_test)
        m_test = model.mass_matrix(q_test)

        assert np.allclose(m_exact, m_test), f"Failed on mass matrix for test value {q_test}"


def test_single_pendulum_dynamics():
    # rest position is hanging down.
    g = 9.8
    g_vec = g * np.array([0,0,-1])
    model = RigidBody()
    inertia= Inertia(
            centre_of_mass=Isometry3(translation=np.array([0, 0, -l_0/2])),
            moments=np.array([Ix_0, 0,0, Iy_0,0, Iz_0]),
            mass=m_0
        )
    w_vec = np.array([0, 1, 0, 0, 0, 0])
    model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw.from_array(w_vec)),
        inertia=inertia
    )

    def potential_energy(q):
        return -np.cos(q[0]) * g * m_0*l_0 / 2

    def kinetic_energy(q, qdot):
        linear = 0.5 * inertia.mass * (l_0/2 ) ** 2 * qdot[0]**2
        angular =0.5 * qdot[0]**2 * np.dot(w_vec, np.dot(inertia.as_matrix(), w_vec))
        return linear + angular

    iz = np.dot(w_vec, np.dot(inertia.as_matrix(), w_vec))
    assert Iz_0 == iz
    def torque(q, qdot, qddot):
        m = inertia.mass * (l_0 / 2) ** 2 + iz
        c = 0
        n = np.sin(q[0]) * g * m_0*l_0 / 2

        return m * qddot + c*qdot + n

    test_scenarios = [
        (np.zeros((1,)), np.zeros((1,)),np.zeros((1,))),
        (np.array([np.pi/4]), np.zeros((1,)),np.zeros((1,))),
         (-np.array([np.pi / 4]), np.zeros((1,)),np.zeros((1,))),
    ] + list(Sampler(1).sample_position_velocity_and_accel(10))

    for q, dq, ddq in test_scenarios:
        pe_test = model.potential_energy(q, g_vec)
        pe = potential_energy(q)
        assert np.allclose(pe_test, pe)

        ke_test = 0.5 * dq[0]**2 * model.mass_matrix(q)
        ke = kinetic_energy(q, dq)
        assert np.allclose(ke_test, ke)

        tau = torque(q,dq,ddq)
        tau_test = model.inverse_dynamics(q,dq,ddq, g_vec)
        assert np.allclose(tau_test,tau)





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
    com_1 = Isometry3(translation=np.array([0, 0, -link_1_length/ 2]))
    mass_1 = 1
    moments_1 = np.array([1, 0, 0, moment_1, 0, 1])

    com_2 = Isometry3(translation=np.array([0, 0, -link_2_length/2]))
    mass_2 = 1
    moments_2 = np.array([1, 0, 0, moment_2, 0, 1])
    g = 9.8
    def double_pendulum_dynamics(q, qdot):
        alpha = moment_1 + moment_2 + mass_1 * (link_1_length/2) **2 + mass_2 * (link_1_length **2 + (link_2_length/2)**2)
        beta = mass_2 *link_1_length * link_2_length / 2
        delta = moment_2 + mass_2 * (link_2_length /2 )**2
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        m = np.array([
            [alpha + 2 * beta * c2, delta +beta *c2],
            [delta + beta*c2, delta]
        ])
        c = np.array([
            [-beta * s2 * qdot[1], -beta * s2 *(qdot[0] + qdot[1])],
            [beta * s2 * qdot[0], 0]
        ])

        # pe = -m_1g * z_1  -m_2 g * z_2
        # z_1 = - l_1/2 cos(q)
        # z_2 = l1 cos(q) - l_2/2 cos(q_1 + q_2)
        #
        # pe = -g ( m_1l_1/2 + m_2l1) cos(q_1) - g cos(q_1+ q_2) l_2/2
        #    = -gamma cos(q_1) - kappa cos(q_1+q_2)
        # d{pe} = + gamma sin(q_1) +kappa sin(q_1 + q_2), - kappa sin( q_1+ q_2)
        gamma = g * (mass_1 * link_1_length / 2 + mass_1 * link_1_length)
        kappa = g * (mass_2 * link_2_length / 2)
        n = np.array([[
            gamma * np.sin(q[0]) + kappa* np.sin(q[0] + q[1]),
            kappa * np.sin(q[0] + q[1])
        ]])

        pe = - gamma * np.cos(q[0]) - kappa * np.cos(q[0] + q[1])
        return m, c, n, pe

    link_1 = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw(rotation=np.array([0, -1, 0]))),
        inertia=Inertia(
            centre_of_mass=com_1,
            mass=mass_1,
            moments=moments_1
        )
    )

    link_2 = model.add_link(
        parent=link_1,
        at=Isometry3(translation=np.array([0, 0, -link_1_length])),
        joint=Revolute(Screw(rotation=np.array([0, -1, 0]))),
        inertia=Inertia(
            centre_of_mass=com_2,
            mass=mass_2,
            moments=moments_2
        )
    )
    effector = model.add_effector(link_2, Isometry3(translation=np.array([0, 0, -link_2_length])))

    return model, double_pendulum_dynamics


def test_double_pendulum(backend):
    model, dynamics = build_double_pendulum()
    g = np.array([0, 0, -9.8])

    def lagrange_impl(q, dq):
        m = model.mass_matrix(q)
        v = model.potential_energy(q, g)

        return 0.5 * np.dot(dq, m @ dq) - v

    f = kernel([VectorSpace('q', 2), VectorSpace('dq', 2)], lagrange_impl, backend)
    test_scenarios = [
        (np.array([0, np.pi / 2]), np.zeros((2,)), np.zeros((2,))),
        (np.array([0, np.pi/2]), np.zeros((2,)), np.zeros((2,))),
        (np.array([np.pi / 2, 0]), np.zeros((2,)), np.zeros((2,))),
        (np.array([np.pi /6, np.pi/6]), np.zeros((2,)), np.zeros((2,))),
    ] + list(Sampler(2).sample_position_and_velocity(5)) \
                     +list(Sampler(2).sample_position_velocity_and_accel(5))

    for q,dq,ddq in test_scenarios:
        M, C, N, pe = dynamics(q,dq)
        mass_test = model.mass_matrix(q)
        assert np.allclose(mass_test, M)

        pe_test = model.potential_energy(q,g)

        assert np.allclose(pe_test, pe)

        tau_test = model.inverse_dynamics(q, dq, ddq, np.array([0, 0, -9.8]))

        tau = M @ ddq + C @ dq + N

        assert np.allclose(tau_test, tau)



def test_scara_inverse_dynamics():
    def scara_inverse_dynamics_matricies(q, qdot):
        alpha = Iz_0 + m_0 * r_0 ** 2 + (m_1 + m_2 + m_3) * l_0 ** 2
        beta = Iz_1 + Iz_2 + Iz_3 + (m_2 + m_3) * l_1 ** 2 + m_1 * r_1 ** 2
        gamma = l_0 * l_1 * (m_2 + m_3) + l_0 * m_1 * r_1
        delta = Iz_2 + Iz_3
        c1 = np.cos(q[1])
        s1 = np.sin(q[1])
        M = np.array([
            [alpha + beta + 2 * gamma * c1, beta + gamma * c1, delta, 0],
            [beta + gamma * c1, beta, delta, 0],
            [delta, delta, delta, 0],
            [0, 0, 0, m_3]]
        )
        C = np.array([
            [-gamma * s1 * qdot[1], -gamma * s1 * (qdot[0] + qdot[1]), 0, 0],
            [gamma * s1 * qdot[0], 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        N = np.array([[0, 0, 0, m_3 * 9.8]])

        return M, C, N

    def scara_inverse_dynamics(q, q_dot, q_ddot):
        M, C, N = scara_inverse_dynamics_matricies(q, q_dot)
        return M @ q_ddot + C @ q_dot + N


    model, _ = build_scara_model()

    def sampler(vel=True, acc=True):

        test_configurations = [
            np.zeros((4,)),
            np.array([1,0,0,0]),
            np.array([0,1,0,0]),
            np.array([0,0,1,0]),
            np.array([0,0,0,1]),
            np.array([1,0,1,0]),
            np.array([1,1,0,0]),
            np.array([1,0,0,1]),
            np.array([0,1,0,1]),
            np.array([0,0,1,1]),
            np.array([1,1,1,0]),
            np.array([1,1,0,1]),
            np.array([1,0,1,1]),
            np.array([0,1,1,1]),
            np.array([1,1,1,1])
        ]
        for t in test_configurations:
            yield t, np.zeros((4,)), np.zeros((4,))

            r_1 = np.random.default_rng().uniform(-1,1,size=(4,))
            r_2 = np.random.default_rng().uniform(-1, 1, size=(4,))
            if vel:
                yield t, r_1, np.zeros((4,))
            if acc:
                yield t, np.zeros((4,)), r_2
            if vel and acc:
                yield t, r_1, r_2

    for (q,q_dot, q_ddot) in sampler():
        M, C, N = scara_inverse_dynamics_matricies(q,q_dot)

        m_test = model.mass_matrix(q)
        assert np.allclose(m_test, M)
        test_torque = model.inverse_dynamics(q, q_dot, q_ddot, np.array([0, 0, g]))

        m_qdot = test_torque - C @ q_dot - N
        m_qdot_expected = M @ q_ddot
        assert np.allclose(m_qdot, m_qdot_expected), f"Failed on mass matrix for test value {q}, {q_dot}, {q_ddot}"

        exact_torque = scara_inverse_dynamics(q, q_dot, q_ddot)
        assert np.allclose(exact_torque, test_torque), \
            f"Failed on torque comparison for test value {q}, {q_dot, q_ddot}\n"\
            f"Expected {exact_torque} but got {test_torque}"

