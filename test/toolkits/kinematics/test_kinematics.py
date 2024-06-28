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

    free_joint, = model.joint_locations(q, tip)
    assert np.allclose(free_joint.translation, np.array([0, 0, 0]))


def is_close(lhs, rhs):

    if isinstance(lhs, Isometry3) and isinstance(rhs, Isometry3):

        return np.allclose(lhs.translation, rhs.translation) and np.allclose(
            lhs.rotation.as_vector(), rhs.rotation.as_vector()
        )
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
    pivot, elbow = base_model.joint_locations(q0)
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
    tool_expected = Isometry3(translation=np.array([1, 0, 1]), rotation=Rotation3(axis=np.array([0, 1, 0]), angle=np.pi/2))
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
        joint=Revolute(Screw.w_x())
    )
    tip = explicit.add_effector(forearm, at=Isometry3(translation=np.array([1, 0, 0])))

    for t_v, t_e in zip(variant.transforms, explicit.transforms):
        assert is_close(t_v, t_e)

    for r_v, r_e in zip(variant._rest_transforms, explicit._rest_transforms):
        assert is_close(r_v, r_e)

    for i, (b_v, b_e) in enumerate(zip(variant.joint_bases, explicit.joint_bases)):
        for b_v_i, b_e_i in zip(b_v, b_e):
            assert is_close(b_v_i, b_e_i), f"Basis {i} not aligned"



    test_set = [
        (np.array([-np.pi/2, 0]), Isometry3(translation=np.array([2, 0, 0]))),
        (np.array([0, 0]), Isometry3(translation=np.array([0, 2, 0]), rotation=Rotation3(e_z, np.pi/2))),
        (np.array([-np.pi/2, -np.pi/2]), Isometry3(translation=np.array([1, 0, 1]), rotation=Rotation3(e_x, np.pi / 2))),
    ]

    for (q, toolhead) in test_set:
        f_variant, = variant.forward_kinematics(q)
        f_explicit, = explicit.forward_kinematics(q)

        assert is_close(f_explicit, toolhead)
        assert is_close(f_variant, toolhead)


#         pivot_base, elbow_base = base_model.joint_locations(q)
        pivot_variant, elbow_variant = variant.joint_locations(q)

