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
    shoulder = model.add_link(
        model.WORLD, base, Revolute(Screw.w_z()), UNIT_INERTIA
    )
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

    explicit = _build_two_link_model(
        Isometry3(rotation=Rotation3(E_Z, np.pi / 2))
    )
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
        [VectorSpace("q", model.total_joints())], model.to_function(), backend="numpy"
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
    expected_torque = (
        effective_inertia * ddq[0] + 9.8 * length / 2 * np.sin(q[0])
    )

    assert np.allclose(model.potential_energy(q, gravity), expected_energy)
    assert np.allclose(model.mass_matrix(q), np.array([[effective_inertia]]))
    assert np.allclose(
        model.inverse_dynamics(q, dq, ddq, gravity), np.array([expected_torque])
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
