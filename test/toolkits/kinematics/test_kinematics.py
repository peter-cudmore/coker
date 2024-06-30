import numpy as np
from coker.toolkits.kinematics import RigidBody, Revolute, Free, Inertia, Weld
from coker.toolkits.spatial import Isometry3, Screw, Rotation3, SE3Adjoint


block_1m = Inertia(
    centre_of_mass=Isometry3(translation=np.array([0.5, 0, 0])),
    mass=1,
    moments=np.array([0.1, 0, 0, 0.1, 0, 0.1]),
)
e_x = np.array([1,0,0])
e_y = np.array([0,1,0])
e_z = np.array([0,0,1])

def test_free_joint():
    model = RigidBody()
    block = model.add_link(
        parent=model.WORLD,
        joint=Free(),
        at=Isometry3.identity(),
        inertia=block_1m
    )

    tip = model.add_effector(parent=block, at=Isometry3(translation=np.array([1, 0, 0])))

    # 0 ----- x

    # at rest, tip is at (1, 0, 0)
    q = np.zeros((6,))

    free_joint, = model.joint_transforms(q, tip)
    assert np.allclose(free_joint.translation, np.array([0, 0, 0]))


def mod_pi(x):
    while x > np.pi:
        x -= 2 * np.pi
    while x < - np.pi:
        x += 2 * np.pi
    return x


def is_close(lhs, rhs):
    if isinstance(lhs, Isometry3) and isinstance(rhs, Isometry3):
        return np.allclose(lhs.translation, rhs.translation) and np.allclose(
            lhs.rotation.axis, rhs.rotation.axis
        ) and np.allclose(mod_pi(lhs.rotation.angle), mod_pi(rhs.rotation.angle))
    if isinstance(lhs, Screw) and isinstance(rhs, Screw):
        return np.allclose(lhs.rotation, rhs.rotation) and np.allclose(lhs.translation, rhs.translation)

    raise NotImplementedError


def test_rotated_base_frame():

    rotation = Rotation3(axis=np.array([0, 0, 1]), angle=np.pi)

    base_model = RigidBody()
    arm = base_model.add_link(
        parent=base_model.WORLD,
        at=Isometry3(rotation=rotation),
        joint=Revolute(Screw.w_z()),
        inertia=block_1m
    )

    end_arm = base_model.add_link(
        parent=arm,
        at=Isometry3(translation=np.array([1, 0, 0])),
        joint=Revolute(Screw.w_y()),
        inertia=block_1m
    )
    effector = base_model.add_effector(end_arm, at=Isometry3(translation=np.array([1, 0, 0])))

    q0 = np.zeros((2,))
    pivot, elbow = base_model.joint_transforms(q0)
    tool, = base_model.forward_kinematics(q0)

    pivot_expected = Isometry3(rotation=rotation)
    assert is_close(pivot, pivot_expected)

    elbow_expected = Isometry3(translation=np.array([-1, 0, 0]), rotation=rotation)
    assert is_close(elbow, elbow_expected)

    tool_expected = Isometry3(translation=np.array([-2, 0, 0]), rotation=rotation)

    assert is_close(tool, tool_expected)

    q1 = np.array([-1, 0]) * np.pi

    tool, = base_model.forward_kinematics(q1)
    tool_expected = Isometry3(translation=np.array([2, 0, 0]))
    assert is_close(tool, tool_expected)

    q2 = np.array([-np.pi, np.pi / 2])
    tool, = base_model.forward_kinematics(q2)
    tool_expected = Isometry3(translation=np.array([1, 0, -1]), rotation=Rotation3(axis=np.array([0, 1, 0]), angle=np.pi/2))
    assert is_close(tool, tool_expected)

    q3 = np.array([0, np.pi / 2])
    tool, = base_model.forward_kinematics(q3)
    rot = Rotation3(axis=np.array([0, 0, 1]), angle=np.pi) * Rotation3(axis=np.array([0, 1, 0]), angle=np.pi/2)
    tool_expected = Isometry3(translation=np.array([-1, 0, 1]), rotation=rot)
    assert is_close(tool, tool_expected)


def test_add_body():

    # Goal.
    # Set up double pendulum with joint 1 = e_z, joint_2 = e_x
    #
    #
    # +------0 - - T        z
    # |                     y x
    # S
    #
    # then set up a second problem where the anchor is rotated by pi/2
    # then, we should always have q_1 = q_2 - [pi/2, 0]
    #

    base_model = RigidBody()
    arm = base_model.add_link(
        parent=base_model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw.w_z()),
        inertia=block_1m
    )

    end_arm = base_model.add_link(
        parent=arm,
        at=Isometry3(translation=np.array([1, 0, 0])),
        joint=Revolute(Screw.w_y()),
        inertia=block_1m
    )

    effector = base_model.add_effector(end_arm, at=Isometry3(translation=np.array([1, 0, 0])))

    rot_offset = np.pi / 2

    variant = RigidBody()
    rotation = Rotation3(axis=np.array([0, 0, 1]), angle=rot_offset)
    variant.add_body(
        body=base_model,
        parent=variant.WORLD,
        at=Isometry3(rotation=rotation)
    )
    explicit = RigidBody()
    upper_arm = explicit.add_link(
        parent=explicit.WORLD,
        at=Isometry3(rotation=rotation),
        inertia=block_1m,
        joint=Revolute(Screw.w_z())
    )
    forearm = explicit.add_link(
        parent=arm,
        at=Isometry3(translation=np.array([1, 0, 0])),
        inertia=block_1m,
        joint=Revolute(Screw.w_y())
    )
    tip = explicit.add_effector(forearm, at=Isometry3(translation=np.array([1, 0, 0])))

    for t_v, t_e in zip(variant.transforms, explicit.transforms):
        assert is_close(t_v, t_e)

    for r_v, r_e in zip(variant._rest_transforms, explicit._rest_transforms):
        assert is_close(r_v, r_e)

    for i in range(len(variant.joint_bases)):
        for b_v_i, b_e_i in zip(variant.joint_global_basis(i), explicit.joint_global_basis(i)):
            assert is_close(b_v_i, b_e_i), f"Basis {i} not aligned"

    test_set = [
        (np.array([-np.pi/2, 0]), Isometry3(translation=np.array([2, 0, 0]))),
        (np.array([0, 0]), Isometry3(translation=np.array([0, 2, 0]), rotation=Rotation3(e_z, np.pi/2))),
        (np.array([-np.pi/2, -np.pi/2]), Isometry3(translation=np.array([1, 0, 1]), rotation=Rotation3(-e_y, np.pi / 2))),
    ]

    for (q, toolhead) in test_set:
        f_variant, = variant.forward_kinematics(q)
        f_explicit, = explicit.forward_kinematics(q)

        assert is_close(f_explicit, f_variant)
        assert is_close(f_explicit, toolhead)
        assert is_close(f_variant, toolhead)


def test_add_multiple():


    #          Rotator
    #          V
    #  --o--|--+--|--o--
    #          |
    #             ^
    #             Arm


    # two link model
    base_model = RigidBody()
    arm = base_model.add_link(
        parent=base_model.WORLD,
        at=Isometry3.identity(),
        joint=Revolute(Screw.w_z()),
        inertia=block_1m
    )

    end_arm = base_model.add_link(
        parent=arm,
        at=Isometry3(translation=np.array([1, 0, 0])),
        joint=Revolute(Screw.w_y()),
        inertia=block_1m
    )

    effector = base_model.add_effector(end_arm, at=Isometry3(translation=np.array([1, 0, 0])))

    rotator_model = RigidBody()
    base_intertia = Inertia(centre_of_mass=Isometry3.identity(), mass=1, moments=np.array([1, 0, 0, 1, 0, 1]))
    base = rotator_model.add_link(parent=rotator_model.WORLD, at=Isometry3.identity(), joint=Revolute(Screw.w_z()), inertia=base_intertia)

    rotator_model.add_body(base_model, parent=base, at=Isometry3(translation=np.array([0.5, 0, 0])))
    rotator_model.add_body(base_model, parent=base, at=Isometry3(translation=np.array([-0.5, 0, 0]), rotation=Rotation3(e_z, np.pi)))
    rest_transforms = [
        Isometry3.identity(),
        Isometry3(translation=np.array([0.5, 0, 0])),
        Isometry3(translation=np.array([1.5, 0, 0])),
        Isometry3(translation=np.array([-0.5, 0, 0]), rotation=Rotation3(e_z, np.pi)),
        Isometry3(translation=np.array([-1.5, 0, 0]),  rotation=Rotation3(e_z, np.pi))
    ]

    for t, rt in zip(rest_transforms, rotator_model._rest_transforms):
        assert is_close(t, rt)

    # translation = -w x p
    # rotation    =  w

    absolute_screw_bases = [
        Screw.w_z(),
        Screw(translation=-np.cross(e_z, 0.5 * e_x), rotation=e_z),
        Screw(translation=-np.cross(e_y, 1.5 * e_x), rotation=e_y),
        Screw(translation=-np.cross(e_z, -0.5 * e_x), rotation=e_z),
        Screw(translation=-np.cross(-e_y, -1.5 * e_x), rotation=-e_y),
    ]

    for i, basis in enumerate(absolute_screw_bases):
        test_basis, = rotator_model.joint_global_basis(i)
        local_basis, = rotator_model.joint_bases[i]

        if i in {0, 1, 3}:
            assert is_close(local_basis, Screw.w_z())
        elif i in {2,4}:
            assert is_close(local_basis, Screw.w_y())

        assert is_close(test_basis, basis), f"Basis {i} is incorrect"

    basis_0 = np.array([1, 0, 0, 0, 0])
    basis_1 = np.array([0, 1, 0, 1, 0])
    basis_2 = np.array([0, 0, 1, 0, 1])

    t1, t2 = rotator_model.forward_kinematics(np.zeros(5,))

    t1_expected = Isometry3(translation=np.array([2.5, 0, 0]))
    t2_expected = Isometry3(translation=np.array([-2.5, 0, 0]), rotation=Rotation3(e_z, np.pi))
    assert is_close(t1, t1_expected)
    assert is_close(t2, t2_expected)

    t1, t2 = rotator_model.forward_kinematics(basis_0 * np.pi)

    t1_expected = Isometry3(translation=np.array([-2.5, 0, 0]), rotation=Rotation3(e_z, np.pi))
    t2_expected = Isometry3(translation=np.array([2.5, 0, 0]))

    assert is_close(t1, t1_expected)
    assert is_close(t2, t2_expected)

    t1, t2 = rotator_model.forward_kinematics(basis_2 * np.pi / 2)
    t1_expected = Isometry3(translation=np.array([1.5, 0, -1]), rotation=Rotation3(e_y, np.pi / 2))
    t2_expected = Isometry3(translation=np.array([-1.5, 0, 1]), rotation=Rotation3(e_z, np.pi) * Rotation3(e_y, np.pi / 2))

    _, r1, r2, l1, l2 = rotator_model.joint_transforms(basis_2 * np.pi/2)

    r1_expected = Isometry3(translation=np.array([0.5, 0, 0]))
    r2_expected = Isometry3(translation=np.array([1.5, 0, 0]), rotation=Rotation3(e_y, np.pi / 2))
    l1_expected = Isometry3(translation=np.array([-0.5, 0, 0]), rotation=Rotation3(e_z, np.pi))
    l2_expected = Isometry3(translation=np.array([-1.5, 0, 0]), rotation=Rotation3(e_z, np.pi) * Rotation3(e_y, np.pi / 2))

    assert is_close(r1, r1_expected)
    assert is_close(r2, r2_expected)
    assert is_close(l1, l1_expected)
    assert is_close(l2, l2_expected)

    assert is_close(t1, t1_expected), f"Twist up failed - right"
    assert is_close(t2, t2_expected), f"Twist up failed - left"

