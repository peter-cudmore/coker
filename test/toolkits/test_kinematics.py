import numpy as np
from coker.algebra.kernel import kernel, VectorSpace
from coker.toolkits.spatial import Screw, Isometry3, SE3Adjoint
from coker.toolkits.kinematics import *
# Test based on 3-link open-chain manipulator
# from Murry Et. Al

from test.util import is_close

# Parameters from the book, r being center of mass, l being next joint
l_0 = 0.5
l_1 = 0.5
l_2 = 0.5
r_0 = l_0 / 2
r_1 = l_1 / 2
r_2 = l_2 / 2


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
    transform_1 = Isometry3(translation=np.array([0, 0, l_0]))

    zeta_2_relative = Screw.from_tuple(-1, 0, 0, 0, 0, 0)
    transform_2 = Isometry3(translation=np.array([0, 0, 0]))
    com_2 = Isometry3(translation=np.array([0, r_1, 0]))

    zeta_3_relative = Screw.from_tuple(-1, 0, 0, 0, 0, 0)
    transform_3 = Isometry3(translation=np.array([0, l_1, 0]))
    com_3 = Isometry3(translation=np.array([0, r_2, 0]))
    end_effector = Isometry3(translation=np.array([0, l_2, 0]))

    t_final = end_effector @ transform_3 @ transform_2 @ transform_1

    assert t_final == g_effector
    Ad_1 = SE3Adjoint(transform_1)
    Ad_2 = SE3Adjoint(transform_2 @ transform_1)
    Ad_3 = SE3Adjoint(transform_3 @ transform_2 @ transform_1)

    assert zeta_1 == Ad_1(zeta_1_relative)
    assert zeta_2 == Ad_2(zeta_2_relative)
    assert zeta_3 == Ad_3(zeta_3_relative)

    model = RigidBody()

    base = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Weld(),
        inertia=Inertia(com_1, 1, moments=np.array([1, 0, 0, 1, 0, 1])),
    )

    link_1 = model.add_link(
        parent=base,
        at=transform_1,
        joint=Planar(zeta_1_relative, zeta_2_relative),
        inertia=Inertia(com_2, 1, moments=np.array([1, 0, 0, 1, 0, 1]))
    )

    link_2 = model.add_link(
        parent=link_1,
        at=transform_2,
        joint=Revolute(zeta_3_relative),
        inertia=Inertia(com_3, 1, moments=np.array([1, 0, 0, 1, 0, 1]))
    )

    effector = model.add_effector(
        parent=link_2,
        at=transform_3
    )

    assert model.total_joints() == 3
    return model


def test_pe():
    model = build_three_link_model()

    def murray_V(angles):
        g = 9.8
        h_1 = r_0
        h_2 = l_0 - r_1 * np.sin(angles[1])
        h_3 = l_0 - l_1 * np.sin(angles[1]) - r_2 * np.sin(angles[1] + angles[2])

        return g * (h_1 + h_2 + h_3)

    q = np.zeros((3, ))

    v = model.potential_energy(
        angles=q,
        gravity_vector=np.array([0, 0, -9.8]).T
    )

    v_test = murray_V(q)

    assert v == v_test

    v_kernel = kernel(
        [VectorSpace('q', 3)],
        lambda a: model.potential_energy(angles=a, gravity_vector=np.array([0, 0, -9.8]).T)
    )

    assert v_kernel(q) == v

def build_scara_model():
    w_z = Screw.from_tuple(0, 0, 1, 0, 0, 0)
    e_z = Screw.from_tuple(0, 0, 0, 0, 0, 1)
    t_1 = Isometry3(translation=np.array([0, 0, l_0]))
    t_2 = Isometry3(translation=np.array([0, l_1, 0]))
    t_3 = Isometry3(translation=np.array([0, l_2, 0]))
    t_4 = Isometry3.identity()
    com_1 = Isometry3(translation=np.array([0, 0, r_0]))
    com_2 = Isometry3(translation=np.array([0, r_1, 0]))
    com_3 = Isometry3(translation=np.array([0, r_2, 0]))
    com_4 = Isometry3.identity()
    end_effector = Isometry3(translation=np.array([0, 0, 0]))

    t_final = end_effector @ t_4 @ t_3 @ t_2 @ t_1
    model = RigidBody()

    base = model.add_link(
        parent=model.WORLD,
        at=Isometry3.identity(),
        joint=Weld(),
        inertia=Inertia(com_1, 1, moments=np.array([1, 0, 0, 1, 0, 1])),
    )

    link_1 = model.add_link(
        parent=base,
        at=t_1,
        joint=Revolute(w_z),
        inertia=Inertia(com_2, 1, moments=np.array([1, 0, 0, 1, 0, 1]))
    )

    link_2 = model.add_link(
        parent=link_1,
        at=t_2,
        joint=Revolute(w_z),
        inertia=Inertia(com_3, 1, moments=np.array([1, 0, 0, 1, 0, 1]))
    )
    link_3 = model.add_link(
        parent=link_2,
        at=t_3,
        joint=Planar(w_z, e_z),
        inertia=Inertia(com_4, 1, moments=np.array([1, 0, 0, 1, 0, 1]))
    )

    effector = model.add_effector(
        parent=link_3,
        at=t_4
    )
    return model, t_final


def test_scara_jacobian():

    def murray_spatial_jacobian(q):
        a_1 = l_1 * np.cos(q[0])
        a_2 = l_1 * np.sin(q[0])
        b_1 = a_1 + l_2 * np.cos(q[0] + q[1])
        b_2 = a_2 + l_2 * np.sin(q[0] + q[1])
        return np.array([
            [0,   0,   0, 0],
            [0,   0,   0, 0],
            [1,   1,   1, 0],
            [0, a_1, b_1, 0],
            [0, a_2, b_2, 0],
            [0,   0,   0, 1]
        ])

    def scara_end_effector(q):
        cs = np.cos(q[0] + q[1] + q[2])
        ss = np.sin(q[0] + q[1] + q[2])
        px = -l_1 * np.sin(q[0]) - l_2 * np.sin(q[0] + q[1])
        py = l_1 * np.cos(q[0]) + l_2 * np.cos(q[0] + q[1])
        pz = l_0 + q[3]
        g_st = np.array([
            [cs, -ss, 0, px],
            [ss, cs,  0, py],
            [0,   0,  1, pz],
            [0,   0,  0,  1]
        ])
        return g_st

    scara_model, t_final = build_scara_model()
    test_angles = [
        np.zeros((4, )),
        np.array([np.pi, 0, 0, 0]),
        np.array([0, np.pi / 4, 0, 0]),
        np.array([0,0, np.pi / 4, 0]),
        np.array([0, 0,  0, 0.25])
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
        jacobian = scara_model.manipulator_jacobian(angles)
        expected_jacobian = murray_spatial_jacobian(angles)
        assert np.allclose(jacobian, expected_jacobian), f"Incorrect Jacobian on {angles}"

