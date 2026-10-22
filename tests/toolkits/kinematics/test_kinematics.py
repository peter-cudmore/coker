import numpy as np
import pytest

from coker import VectorSpace, function
from coker.algebra.ops import OP
from coker.toolkits.kinematics import Free, Inertia, Revolute, RigidBody, Weld
from coker.toolkits.spatial import Isometry3, Rotation3, Screw

E_X = np.array([1.0, 0.0, 0.0])
E_Y = np.array([0.0, 1.0, 0.0])
E_Z = np.array([0.0, 0.0, 1.0])
UNIT_INERTIA = Inertia(
    centre_of_mass=Isometry3.identity(),
    mass=1.0,
    moments=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
)


def _build_two_link_model(base=Isometry3.identity()):
    model = RigidBody()
    shoulder = model.add_link(model.WORLD, base, Revolute(Screw.w_z()), UNIT_INERTIA)
    elbow = model.add_link(
        shoulder,
        Isometry3(translation=E_X),
        Revolute(Screw.w_y()),
        UNIT_INERTIA,
    )
    model.add_effector(elbow, Isometry3(translation=E_X))
    return model


@pytest.mark.parametrize(
    ("joint", "coordinates", "angles", "expected"),
    [
        (Weld(), 0, np.array([]), E_X),
        (Revolute(Screw.w_z()), 1, np.array([np.pi / 2]), E_Y),
        (Revolute(Screw(translation=E_Z)), 1, np.array([2.0]), E_X + 2 * E_Z),
    ],
)
def test_joint_coordinate_contract(joint, coordinates, angles, expected):
    model = RigidBody()
    link = model.add_link(model.WORLD, Isometry3.identity(), joint, UNIT_INERTIA)
    model.add_effector(link, Isometry3(translation=E_X))

    assert model.total_joints() == coordinates
    assert np.allclose(model.forward_kinematics(angles)[0].translation, expected)


def test_free_joint_exposes_six_motion_coordinates():
    model = RigidBody()
    link = model.add_link(model.WORLD, Isometry3.identity(), Free(), UNIT_INERTIA)
    model.add_effector(link, Isometry3(translation=E_X))
    angles = np.array([1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2])

    assert model.total_joints() == 6
    assert np.allclose(
        model.forward_kinematics(angles)[0].translation,
        np.array([1.0, 3.0, 3.0]),
        atol=1e-9,
    )


def test_rotated_base_matches_direct_and_symbolic_fk():
    model = _build_two_link_model(Isometry3(rotation=Rotation3(E_Z, np.pi)))
    q = np.array([-np.pi, -np.pi / 2])
    expected = np.array([1.0, 0.0, 1.0])

    assert np.allclose(model.forward_kinematics(q)[0].translation, expected)
    compiled = function(
        [VectorSpace("q", 2)],
        lambda angles: model.forward_kinematics(angles)[0].translation,
        backend="numpy",
    )
    assert np.allclose(compiled(q), expected)


def test_add_body_matches_explicit_tree():
    source = _build_two_link_model()
    imported = RigidBody()
    imported.add_body(
        source,
        Isometry3(rotation=Rotation3(E_Z, np.pi / 2)),
        imported.WORLD,
    )

    explicit = _build_two_link_model(Isometry3(rotation=Rotation3(E_Z, np.pi / 2)))
    for angles in (np.zeros(2), np.array([np.pi / 3, -np.pi / 4])):
        assert np.allclose(
            imported.forward_kinematics(angles)[0].as_matrix(),
            explicit.forward_kinematics(angles)[0].as_matrix(),
            atol=1e-9,
        )


@pytest.mark.parametrize("branched", [True, False])
def test_to_function_matches_direct_fk_and_jacobian(branched):
    model = RigidBody()
    parents = []
    parent = model.WORLD
    for leg in range(2):
        if branched:
            parent = model.WORLD
        for _ in range(2):
            parent = model.add_link(
                parent,
                Isometry3(translation=np.array([0.4, 0.1 * leg, 0.0])),
                Revolute(Screw.w_z()),
                Inertia.zero(),
            )
        parents.append(parent)
    for parent in parents:
        model.add_effector(parent, Isometry3(translation=np.array([0.2, 0.0, 0.0])))

    angles = np.linspace(-0.4, 0.4, model.total_joints())
    direct = model.forward_kinematics(angles)
    spatial = model.spatial_manipulator_jacobian(angles)
    expected_positions = np.concatenate([transform.translation for transform in direct])
    expected_jacobians = np.concatenate(
        [
            jacobian[3:, :] + np.cross(jacobian[:3, :].T, transform.translation).T
            for transform, jacobian in zip(direct, spatial)
        ]
    )
    compiled = function(
        [VectorSpace("q", model.total_joints())],
        model.to_function(),
        backend="numpy",
    )
    positions, jacobians = compiled(angles)

    assert np.allclose(positions, expected_positions)
    assert np.allclose(jacobians, expected_jacobians)


def test_unit_revolute_jacobian_has_known_spatial_and_cartesian_columns():
    model = RigidBody()
    link = model.add_link(
        model.WORLD,
        Isometry3.identity(),
        Revolute(Screw.w_z()),
        Inertia.zero(),
    )
    model.add_effector(link, Isometry3(translation=E_X))
    angles = np.array([0.0])

    (spatial,) = model.spatial_manipulator_jacobian(angles)
    _, cartesian = model.to_function()(angles)

    assert np.allclose(spatial[:, 0], np.array([0, 0, 1, 0, 0, 0]))
    assert np.allclose(cartesian[:, 0], np.array([0, 1, 0]))


def test_single_pendulum_energy_and_inverse_dynamics():
    length = 0.5
    inertia = Inertia(
        centre_of_mass=Isometry3(translation=np.array([0.0, 0.0, -length / 2])),
        mass=1.0,
        moments=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
    )
    model = RigidBody()
    model.add_link(
        model.WORLD,
        Isometry3.identity(),
        Revolute(Screw(rotation=E_Y)),
        inertia,
    )
    gravity = np.array([0.0, 0.0, -9.8])
    q = np.array([np.pi / 4])
    dq = np.array([0.3])
    ddq = np.array([-0.2])
    rotational_inertia = 1.0
    effective_inertia = rotational_inertia + (length / 2) ** 2
    expected_energy = -9.8 * length / 2 * np.cos(q[0])
    expected_torque = effective_inertia * ddq[0] + 9.8 * length / 2 * np.sin(q[0])

    assert np.allclose(model.potential_energy(q, gravity), expected_energy)
    assert np.allclose(model.mass_matrix(q), np.array([[effective_inertia]]))
    assert np.allclose(
        model.inverse_dynamics(q, dq, ddq, gravity),
        np.array([expected_torque]),
    )


def _build_hexapod_leg():
    model = RigidBody()
    parent = model.WORLD
    for axis, offset in ((E_Z, E_X), (-E_Y, E_X), (-E_Y, E_X)):
        parent = model.add_link(
            parent,
            Isometry3(translation=offset),
            Revolute(Screw(rotation=axis)),
            Inertia.zero(),
        )
    model.add_effector(parent, Isometry3(translation=np.array([0.16, 0.0, -0.03])))
    return model


def test_hexapod_fk_trace_is_compact_and_exact(backend):
    model = _build_hexapod_leg()

    def implementation(angles):
        return model.forward_kinematics(angles)[0].translation

    compiled = function([VectorSpace("q", 3)], implementation, backend=backend)
    operations = [node[0] for node in compiled.tape.nodes._nodes]
    nonlinear_count = sum(
        operation.is_nonlinear()
        for operation in operations
        if isinstance(operation, OP)
    )
    angles = np.array([0.3, -0.5, 0.8])

    assert np.allclose(compiled(angles), implementation(angles), atol=1e-9)
    assert operations.count(OP.ARCTAN2) == 0
    assert len(compiled.tape) < 611
    assert nonlinear_count < 159


@pytest.mark.parametrize("parent", [-2, 0, 1])
def test_public_contract_add_link_rejects_unavailable_parent(parent):
    model = RigidBody()

    with pytest.raises(ValueError):
        model.add_link(parent, Isometry3.identity(), Weld(), Inertia.zero())

    assert model.parents == []
    assert model.joints == []
    assert model.transforms == []
    assert model.inertia == []
    assert model.end_effectors == []


@pytest.mark.parametrize("parent", [-1.0, 0.0, "world", None])
def test_public_contract_add_link_rejects_noninteger_parent(parent):
    model = RigidBody()
    pose = Isometry3.identity()
    joint = Weld()
    inertia = Inertia.zero()
    model.add_link(model.WORLD, pose, joint, inertia)

    with pytest.raises(TypeError):
        model.add_link(parent, Isometry3.identity(), Weld(), Inertia.zero())

    assert model.parents == [model.WORLD]
    assert model.joints == [joint]
    assert model.transforms == [pose]
    assert model.inertia == [inertia]
    assert model.end_effectors == []


@pytest.mark.parametrize("parent", [-2, RigidBody.WORLD, 1])
def test_public_contract_add_effector_rejects_invalid_parent(parent):
    model = RigidBody()
    model.add_link(model.WORLD, Isometry3.identity(), Weld(), Inertia.zero())

    with pytest.raises(ValueError):
        model.add_effector(parent, Isometry3.identity())

    assert model.end_effectors == []


@pytest.mark.parametrize("parent", [0.0, "0", None])
def test_public_contract_add_effector_rejects_noninteger_parent(parent):
    model = RigidBody()
    model.add_link(model.WORLD, Isometry3.identity(), Weld(), Inertia.zero())

    with pytest.raises(TypeError):
        model.add_effector(parent, Isometry3.identity())

    assert model.end_effectors == []


@pytest.mark.parametrize(
    ("world", "parent", "effector_parent"),
    [
        (np.int32(-1), np.int32(0), np.int32(1)),
        (np.int64(-1), np.int64(0), np.int64(1)),
        (RigidBody.WORLD, False, True),
    ],
    ids=["numpy-int32", "numpy-int64", "bool"],
)
def test_public_contract_parent_indices_support_integer_protocol(
    world, parent, effector_parent
):
    model = RigidBody()
    root = model.add_link(world, Isometry3.identity(), Weld(), Inertia.zero())
    child = model.add_link(parent, Isometry3(translation=E_X), Weld(), Inertia.zero())
    effector = model.add_effector(effector_parent, Isometry3(translation=E_Y))

    assert (root, child, effector) == (0, 1, 0)
    assert model.parents == [RigidBody.WORLD, 0]
    np.testing.assert_allclose(
        model.forward_kinematics(np.zeros(0))[0].translation, E_X + E_Y
    )


@pytest.mark.parametrize("angles", [np.zeros(1), np.zeros(3), np.zeros((3, 1))])
def test_public_contract_potential_energy_rejects_coordinate_shape(angles):
    model = _build_two_link_model()

    with pytest.raises(ValueError):
        model.potential_energy(angles, np.array([0.0, 0.0, -9.8]))


def test_public_contract_potential_energy_accepts_column_coordinates():
    model = _build_two_link_model()
    angles = np.array([np.pi / 3, -np.pi / 4])
    gravity = np.array([0.0, -9.8, 0.0])

    np.testing.assert_allclose(
        model.potential_energy(angles.reshape(2, 1), gravity),
        model.potential_energy(angles, gravity),
    )


def test_public_contract_inertia_origin_rejects_rotated_mass_frame():
    inertia = Inertia(
        centre_of_mass=Isometry3(rotation=Rotation3(E_Z, np.pi / 2)),
        mass=1.0,
        moments=UNIT_INERTIA.moments.copy(),
    )

    with pytest.raises(ValueError):
        inertia.as_matrix_origin()

    np.testing.assert_allclose(inertia.as_matrix(), UNIT_INERTIA.as_matrix())
