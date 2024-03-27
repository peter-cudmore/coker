import numpy as np
from coker.algebra.kernel import kernel, VectorSpace
from coker.toolkits.spatial import Screw, Isometry3, SE3Adjoint
from coker.toolkits.kinematics import *
# Test based on 3-link open-chain manipulator
# from Murry Et. Al

# Parameters from the book, r being center of mass, l being next joint
l_0 = 0.5
l_1 = 0.5
l_2 = 0.5
r_0 = l_0 / 2
r_1 = l_1 / 2
r_2 = l_2 / 2


def build_model():
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
    model = build_model()

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

