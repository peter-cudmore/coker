import numpy as np
import pytest

import coker
from coker.toolkits.spatial import Isometry3, Rotation3, SE3Adjoint, Screw
from mpl_toolkits.mplot3d.proj3d import rot_x

from test.conftest import backends
from test.util import is_close, validate_symbolic_call
from coker import VectorSpace, kernel, Scalar

u_x = np.array([1, 0, 0], dtype=float)
u_y = np.array([0, 1, 0], dtype=float)
u_z = np.array([0, 0, 1], dtype=float)

origin = np.array([0, 0, 0, 1], dtype=float).reshape((4, 1))


def test_hat(backend):
    from coker.toolkits.spatial.algebra import hat

    expected = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    result = hat(u_x)
    assert is_close(result, expected)

    hat_symbolic = kernel([VectorSpace('u',3)], hat, backend=backend)
    result = hat_symbolic(u_x)
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

def test_quaternions_symbolic(backend):
    from coker.toolkits.spatial.unit_quaternion import UnitQuaternion
    def conj_impl(axis, angle):
        q_z = UnitQuaternion.from_axis_angle(axis, angle)
        result = q_z.conjugate(u_x)
        return result

    conj = kernel([VectorSpace('u',3), coker.Scalar('angle')], conj_impl, backend=backend)
    result = conj(u_z, np.pi/2)
    assert is_close(result, u_y, tolerance=1e-4)

    def inverse_impl(axis, angle, u):
        q_z = UnitQuaternion.from_axis_angle(axis, angle)
        return (q_z.inverse() * q_z).conjugate(u)
    inverse = kernel([VectorSpace('u',3), coker.Scalar('angle'), VectorSpace('v', 3)], inverse_impl, backend=backend)
    result = inverse(u_z, np.pi/2, u_y)
    assert is_close(result, u_y, tolerance=1e-4)

def test_quaternion_product(backend):

    from coker.toolkits.spatial.unit_quaternion import UnitQuaternion
    def q_product(a_1, b_1, a_2, b_2):
        q_1 = UnitQuaternion.from_axis_angle(a_1, b_1)
        q_2 = UnitQuaternion.from_axis_angle(a_2, b_2)
        q = q_1 * q_2
        q_out = np.array(q.to_elements())
        assert q_out.shape == (4,)
        return q_out

    args = [
        VectorSpace('a_1',3),
        Scalar('b_1'),
        VectorSpace('a_2',3),
        Scalar('b_2')
    ]
    cases = [
        [u_x, 0, u_x, 0],
        [u_x, np.pi, u_y, 0],
        [u_x, np.pi, u_x, -np.pi],
    ]
    validate_symbolic_call('q_product', q_product, args, cases, backend=backend)


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

def test_isometry_rotations():
    ry = Isometry3(rotation=Rotation3(axis=u_y, angle=np.pi / 2))
    rx = Isometry3(rotation=Rotation3(axis=u_x, angle=-np.pi / 2))
    rz = Isometry3(rotation=Rotation3(axis=u_z, angle=np.pi / 2))
    ryx = ry @ rx
    rzy = rz @ ry

    assert is_close(rzy, ryx)


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

    validate_symbolic_call(
        'exp', lambda t: s.exp(t).as_matrix(), [Scalar('theta')],
        [[0], [np.pi/3], [np.pi], [1]]  ,backend='numpy'                                          )


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


def test_symbolic_isometries(backend):
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
        test_set,
        backend)

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
        test_set, backend)

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
                           test_set, backend)

def test_symbolic_isometry_product_translation(backend):
    from coker.algebra.kernel import Scalar
    from coker.toolkits.spatial.algebra import Screw, UnitQuaternion


    def f_impl(x, y):
        g_0 = Isometry3(translation=x)
        g_1 = Isometry3(translation=y)
        return (g_0 @ g_1).as_vector()

    args = [VectorSpace('x',3), VectorSpace('y',3 )]
    test_values = [
        [u_x, u_y],
        [u_y, u_x]
    ]
    validate_symbolic_call('test_symbolic_isometry_product', f_impl, args, test_values,backend)

def test_symbolic_isometry_product_rotation(backend):
    from coker.algebra.kernel import Scalar
    from coker.toolkits.spatial.algebra import Screw, UnitQuaternion


    def f_impl(x, y):
        rotation_1 = Rotation3(axis=x, angle=np.pi / 2)
        rotation_2 = Rotation3(axis=y, angle=np.pi / 2)
        rot = rotation_1 * rotation_2
        return rot.as_matrix()

    args = [VectorSpace('x',3), VectorSpace('y',3 )]
    test_values = [
        [u_x, u_y],
        [u_y, u_x]
    ]
    validate_symbolic_call('test_symbolic_isometry_product', f_impl, args, test_values, backend)


def test_symbolic_isometry_product_screw():
    from coker.algebra.kernel import Scalar
    from coker.toolkits.spatial.algebra import Screw

    s_1 = Screw(rotation=u_z, translation=np.array([0, 0, 0], dtype=float), magnitude=np.pi / 2)
    def f_impl(t, theta_1):
        g = Isometry3(translation=t)
        s = s_1.exp(theta_1)
        g = g @ s
        return g.as_vector()

    args = [VectorSpace('u',3), Scalar('theta_1')]
    test_values = [
        [np.array([0, 0, 0]), 0],
        [np.array([0, 0, 0]), np.pi/2],
    ]
    validate_symbolic_call('test_symbolic_isometry_product', f_impl, args, test_values, 'numpy')



def test_as_matrix():
    from coker.toolkits.spatial import SE3Adjoint, SE3CoAdjoint, Screw

    transforms =[
        Isometry3(rotation=Rotation3(axis=np.array([0, 0, 1]), angle=np.pi / 2)),
        Isometry3.identity(),
        Isometry3(translation=np.array([1,0,1]))
    ]
    for transform in transforms:

        screw_1 = Screw.from_tuple(1,0,0, 1, 2, 3)
        adj = SE3Adjoint(transform=transform)

        image_screw = adj.apply(screw_1).to_array()
        image_screw_2 = adj.as_matrix() @ screw_1.to_array()

        assert np.allclose(image_screw, image_screw_2)

        coadj = SE3CoAdjoint(transform)
        image_screw = coadj.apply(screw_1).to_array()
        image_screw_2 = coadj.as_matrix() @ screw_1.to_array()

        assert np.allclose(image_screw, image_screw_2)


def test_bug_1():

    r = Isometry3(rotation=Rotation3(np.array([0., 0., 0.]), 0.0), translation=np.array([1., 0., 0.]))
    r_2 = Isometry3(rotation=Rotation3(np.array([0, 0, 1]), 0), translation=np.array([1, 0, 0]))

    p = r @ r_2

    assert p.rotation.angle == 0
    assert np.allclose(p.translation, np.array([2., 0., 0.]))

def test_adjoint():

    from coker.toolkits.spatial import Screw, SE3Adjoint, Isometry3

    s = Screw.w_y()
    shift_x = Isometry3(translation=np.array([1., 0., 0.]))
    s_shift_x = SE3Adjoint(shift_x).apply(s)

    assert np.allclose(s_shift_x.to_array(), np.array([0,1,0, 0,0,1]))

    rot_z = Isometry3(rotation=Rotation3(np.array([0., 0., 1.]), np.pi / 2))
    s_rot_z = SE3Adjoint(rot_z).apply(s)
    assert np.allclose(s_rot_z.to_array(), np.array([-1,0,0, 0,0,0]))


def test_isometry_chain():

    e_x = np.array([1,0,0])
    e_y = np.array([0,1,0])
    e_z = np.array([0,0,1])

    #   -
    # _| |_   == ---
    #
    part_1 = (
        Isometry3(translation=e_x)
        @ Isometry3(translation=e_y)
        @ Isometry3(translation=e_x)
        @ Isometry3(translation=-e_y)
        @ Isometry3(translation=e_x)
    )
    part_1_equivament = Isometry3(translation=3 * e_x)

    assert is_close(part_1, part_1_equivament)

    part_1_rotations = (
        Isometry3(translation=e_x)
        @ Isometry3(rotation=Rotation3(axis=e_z, angle=np.pi / 2))
        @ Isometry3(translation=e_x)
        @ Isometry3(rotation=Rotation3(axis=-e_z, angle=np.pi / 2))
        @ Isometry3(translation=e_x)
        @ Isometry3(rotation=Rotation3(axis=-e_z, angle=np.pi / 2))
        @ Isometry3(translation=e_x)
        @ Isometry3(rotation=Rotation3(axis=e_z, angle=np.pi / 2))
        @ Isometry3(translation=e_x)
    )
    assert is_close(part_1_rotations, part_1_equivament)
    p = part_1_rotations.apply(np.zeros((3,)))
    p_expected = np.array([3,0,0])
    assert np.allclose(p, p_expected)

    part_1_screws = (
            Isometry3(translation=e_x)
            @ Isometry3(rotation=Rotation3(axis=e_z, angle=np.pi / 2))
            @ Isometry3(translation=e_x)
            @ Screw.w_z().exp(-np.pi/2)
            @ Isometry3(translation=e_x)
            @ Isometry3(rotation=Rotation3(axis=-e_z, angle=np.pi / 2))
            @ Isometry3(translation=e_x)
            @ Screw.w_z().exp(np.pi / 2)
            @ Isometry3(translation=e_x)
    )
    assert is_close(part_1_screws, part_1_equivament)

    part_2 = (
        Isometry3(translation=e_x)          # (1, 0, 0)
        @ Isometry3(translation=e_y)        # (1, 1, 0)
        @ Isometry3(translation=e_z)        # (1, 1, 1)
    )
    p20 = Isometry3(translation=e_x)
    p21 = p20 @ Isometry3(rotation=Rotation3(axis=e_z, angle=np.pi / 2))
    t21 =  Isometry3(translation=e_x).as_matrix() @ Isometry3(rotation=Rotation3(axis=e_z, angle=np.pi / 2)) .as_matrix()

    assert np.allclose(p21.as_matrix(), t21)

    p22 = p21 @ Isometry3(translation=e_x)
    t22 = t21 @ Isometry3(translation=e_x).as_matrix()
    p22m = p22.as_matrix()
    assert np.allclose(p22m, t22)

    p23 = p22 @ Isometry3(rotation=Rotation3(axis=e_y, angle=-np.pi / 2))
    t23 = t22 @ Isometry3(rotation=Rotation3(axis=-e_y, angle=np.pi / 2)).as_matrix()
    p23m = p23.as_matrix()
    assert np.allclose(p23m, t23)

    p24 = p23 @ Isometry3(translation=e_x)

    p = part_2 @ origin
    p_expected = p24 @ origin

    assert  np.allclose(p, p_expected)