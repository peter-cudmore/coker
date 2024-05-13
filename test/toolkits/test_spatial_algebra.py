import numpy as np
import pytest

from test.util import is_close, validate_symbolic_call

u_x = np.array([1, 0, 0], dtype=float)
u_y = np.array([0, 1, 0], dtype=float)
u_z = np.array([0, 0, 1], dtype=float)

origin = np.array([0, 0, 0, 1], dtype=float).reshape((4, 1))


def test_hat():
    from coker.toolkits.spatial.algebra import hat

    expected = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    result = hat(u_x)
    assert is_close(result, expected)


def test_quaternions():
    from coker.toolkits.spatial.unit_quaternion import UnitQuaternion

    q_z = UnitQuaternion.from_axis_angle(u_z, np.pi / 2)

    result = q_z.conjugate(u_x)
    assert is_close(result, u_y, tolerance=1e-4)

    q_x = UnitQuaternion.from_axis_angle(u_x, np.pi / 2)

    q_xz = q_x * q_z  # rotate around z, then rotate around x
    result = q_xz.conjugate(u_x)
    assert is_close(result, u_z, tolerance=1e-4)

    with pytest.raises(TypeError):
        q = q_x + q_z

    identity = q_z * q_z.inverse()
    identity_2 = q_z.inverse() * q_z
    result = identity.conjugate(u_x)
    assert is_close(result, u_x, tolerance=1e-4)

    assert identity == identity_2


def test_isometry():
    from coker.toolkits.spatial.algebra import Isometry3, Rotation3

    eye = Isometry3.identity()

    assert is_close(u_x, eye @ u_x, tolerance=1e-9)
    assert is_close(u_x, eye.transpose() @ u_x, tolerance=1e-9)

    x_shift = Isometry3(translation=u_x)
    result = x_shift @ origin
    assert is_close(u_x, result[0:3].reshape(u_x.shape), tolerance=1e-9)

    assert is_close(origin, x_shift.transpose() @ result, tolerance=1e-3)

    rot_and_shift = Isometry3(rotation=Rotation3(axis=u_z, angle=np.pi / 2), translation=u_x)
    test_point = np.array([1, 0, 0, 1], dtype=float).reshape((4, 1))
    image = rot_and_shift @ test_point

    assert is_close(image, np.array([1, 1, 0, 1], dtype=float).reshape((4, 1)), tolerance=1e-4)

    # potential bug in screw code
    ax = Rotation3(axis=u_z, angle=0)
    iso = Isometry3(rotation=ax)
    eye = Isometry3.identity()

    assert is_close(iso.as_matrix(), eye.as_matrix(), tolerance=1e-9)
    xform = iso @ eye
    assert is_close(xform.as_matrix(), eye.as_matrix(), 1e-9)


def test_screws():
    from coker.toolkits.spatial.algebra import Screw, Isometry3, Rotation3

    s = Screw(rotation=u_z, translation=np.array([0, 0, 0], dtype=float), magnitude=np.pi / 2)
    r = s.exp(1)
    r_expected = Isometry3(rotation=Rotation3(u_z, np.pi / 2))

    assert is_close(r.translation, r_expected.translation)
    assert r.rotation == r_expected.rotation

    r = s.exp(0)
    r_expected = Isometry3.identity()
    assert is_close(r.translation, r_expected.translation)
    assert r.rotation == r_expected.rotation

    array = s.to_array()
    expected_array = np.array([0, 0, np.pi/2, 0, 0, 0])
    assert np.allclose(array, expected_array)


def test_prismatic_screw():
    from coker.toolkits.spatial.algebra import Screw, Isometry3, Rotation3

    screw = Screw.from_tuple(0, 0, 0, 0, 0, 1)
    identity = screw.exp(0).as_matrix()
    assert is_close(identity, np.eye(4),  1e-9)

    transform = screw.exp(5)
    expected_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 5],
        [0, 0, 0, 1]
    ], dtype=float)
    actual = transform.as_matrix()
    assert is_close(actual, expected_matrix,  1e-6)


def test_symbolic_isometries():
    from coker.algebra.kernel import Scalar
    from coker.toolkits.spatial import Isometry3, Rotation3

    def translation(x):
        v = np.array([x, 0, 0])
        iso = Isometry3(translation=v)
        return iso.as_matrix()

    test_set = [
        [1.0],
        [0.0],
        [-2.0]
    ]

    validate_symbolic_call(
        'test_translation',
        translation,
        [Scalar('x')],
        test_set)

    def rotation(theta):
        r = Rotation3(axis=np.array([0, 0, 1]), angle=theta)
        return r.as_matrix()

    test_set = [
        [0.0],
        [np.pi / 2],
        [-np.pi / 2]
    ]

    validate_symbolic_call(
        'test_rotation',
        rotation,
        [Scalar('x')],
        test_set)

    def combined(x, theta):
        p = np.array([x, 0, 0])
        r = Rotation3(axis=np.array([0, 0, 1]), angle=theta)
        iso = Isometry3(translation=p, rotation=r)
        return iso.as_matrix()

    test_set = [
        [0.0, 0.0],
        [1.0, np.pi/2],
        [-1.0, -np.pi / 3],
    ]

    validate_symbolic_call('test_isometry',
                           combined,
                           [Scalar('x'), Scalar('theta')],
                           test_set)

