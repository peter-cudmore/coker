import numpy as np

from coker import Scalar, VectorSpace, function
from coker.toolkits.spatial import (
    Isometry3,
    Rotation3,
    Screw,
    SE3Adjoint,
    SE3CoAdjoint,
)
from coker.toolkits.spatial.algebra import hat
from coker.toolkits.spatial.unit_quaternion import UnitQuaternion

E_X = np.array([1.0, 0.0, 0.0])
E_Y = np.array([0.0, 1.0, 0.0])
E_Z = np.array([0.0, 0.0, 1.0])


def test_rotation_group_action_and_inverse():
    quarter_turn = UnitQuaternion.from_axis_angle(E_Z, np.pi / 2)
    rotation = Rotation3(E_Z, np.pi / 2)

    assert np.allclose(quarter_turn.conjugate(E_X), E_Y)
    assert np.allclose(rotation.apply(E_X), E_Y)
    assert np.allclose(
        (rotation * rotation.inverse()).as_matrix(), np.eye(3), atol=1e-9
    )


def test_isometry_composition_inverse_and_homogeneous_action():
    transform = Isometry3(
        rotation=Rotation3(E_Z, np.pi / 2), translation=E_X
    )
    point = np.array([1.0, 0.0, 0.0])
    homogeneous_point = np.array([[1.0], [0.0], [0.0], [1.0]])

    assert np.allclose(transform @ point, np.array([1.0, 1.0, 0.0]))
    assert np.allclose(
        transform.inverse() @ (transform @ point), point, atol=1e-9
    )
    assert np.allclose(
        (transform @ homogeneous_point)[:3, 0], np.array([1.0, 1.0, 0.0])
    )


def test_screw_exponential_covers_revolute_and_prismatic_motion():
    revolute = Screw(rotation=E_Z, magnitude=np.pi / 2)
    prismatic = Screw(translation=E_Z)

    assert np.allclose(revolute.exp().apply(E_X), E_Y, atol=1e-9)
    assert np.allclose(prismatic.exp(5).translation, 5 * E_Z)
    assert np.allclose(prismatic.exp(0).as_matrix(), np.eye(4))


def test_adjoint_and_coadjoint_match_matrix_actions():
    screw = Screw.from_tuple(1, 0, 0, 1, 2, 3)
    transforms = [
        Isometry3(translation=E_X),
        Isometry3(rotation=Rotation3(E_Z, np.pi / 2)),
        Isometry3(rotation=Rotation3(E_Z, np.pi / 2), translation=E_X),
    ]

    for transform in transforms:
        adjoint = SE3Adjoint(transform)
        coadjoint = SE3CoAdjoint(transform)
        assert np.allclose(
            adjoint.apply(screw).to_array(),
            adjoint.as_matrix() @ screw.to_array(),
        )
        assert np.allclose(
            coadjoint.apply(screw).to_array(),
            coadjoint.as_matrix() @ screw.to_array(),
        )


def test_noncommuting_isometry_chain_preserves_group_result():
    transform = (
        Isometry3(translation=E_X)
        @ Isometry3(rotation=Rotation3(E_Z, np.pi / 2))
        @ Isometry3(translation=E_X)
        @ Screw.w_z().exp(-np.pi / 2)
        @ Isometry3(translation=E_X)
    )

    assert np.allclose(
        transform.apply(np.zeros(3)), np.array([2.0, 1.0, 0.0]), atol=1e-9
    )


def test_symbolic_spatial_contract(backend):
    def implementation(axis, angle, point, translation, displacement):
        rotation = Rotation3(axis, angle)
        transform = Isometry3(rotation=rotation, translation=translation)
        screw = Screw(translation=E_Z)
        return (
            hat(axis),
            transform.apply(point),
            (rotation * rotation.inverse()).as_matrix(),
            screw.exp(displacement).translation,
        )

    compiled = function(
        [
            VectorSpace("axis", 3),
            Scalar("angle"),
            VectorSpace("point", 3),
            VectorSpace("translation", 3),
            Scalar("displacement"),
        ],
        implementation,
        backend=backend,
    )
    args = (E_Z, np.pi / 2, E_X, E_X, 2.0)
    for actual, expected in zip(compiled(*args), implementation(*args)):
        assert np.allclose(actual, expected, atol=1e-9)
