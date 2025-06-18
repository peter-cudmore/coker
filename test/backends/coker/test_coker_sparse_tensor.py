from coker.backends.coker.sparse_tensor import dok_ndarray
from coker.backends.coker.tensor_contants import hat
import numpy as np


def test_convert():
    eye = np.eye(3, dtype=float)
    sparse_eye = dok_ndarray.fromarray(eye)
    assert sparse_eye.shape == (3, 3)
    assert sparse_eye.keys == {(0, 0): 1.0, (1, 1): 1.0, (2, 2): 1.0}
    assert np.allclose(sparse_eye.toarray(), eye)

    test = np.array(
        [[0, 1, 2],
         [0, 0, 3],
         [4, 0, 0]
         ]
    )
    test_keys = {
        (0, 1): 1,
        (0, 2): 2,
        (1, 2): 3,
        (2, 0): 4
    }
    test_sparse = dok_ndarray((3, 3), test_keys)
    assert np.allclose(test_sparse.toarray(), test)


def test_matmul_array():
    t = dok_ndarray(shape=(3, 3), data={(2, 2): 1})
    t_test = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    assert np.allclose(t.toarray(), t_test)

    b = np.array([1, 2, 3])


    y_expected = t_test @ b

    y = t @ b

    assert np.allclose(y.toarray(), y_expected)

    c = np.eye(3)

    id_1 = t @ c
    id_2 = c @ t

    id_1_array = id_1.toarray()
    assert np.allclose(id_1_array, t_test)
    assert np.allclose(id_2, t_test)
    assert np.allclose(id_1_array, id_2)

    bt = b.T
    z = bt @ t
    z_test = bt @ t_test
    assert np.allclose(z, z_test)

def test_matmul_tensor():
    ex = np.array([[1, 0, 0]])
    f = dok_ndarray(shape=(3, 3, 3), data={
        (0, 0, 0): 1,
        (0, 1, 1): 1,
        (0, 2, 2): 1,
        (1, 1, 1): 1,
        (2, 2, 2): 1,
    })
    eye_tensor = ex @ f
    eye_array = eye_tensor
    eye = np.eye(3, dtype=float)
    assert np.allclose(eye_array, eye)


def test_cross():
    a = np.array([1, 2, 3])
    a_hat = np.array([
        [0, -3, 2],
        [3, 0, -1],
        [-2, 1, 0]
    ])
    tensor = hat(a).toarray()
    assert np.allclose(tensor, a_hat)

    b = np.array([0, 1, 0])
    axb = np.cross(a, b)

    axb_tensor = hat(a) @ b

    assert np.allclose(axb_tensor.toarray(), axb)

    axa = np.cross(a, a)
    axa_test = hat(a) @ a
    assert np.allclose(axa_test.toarray(), axa)



def test_outer_product():
    a = np.array([[1, 2, 3]])
    b = dok_ndarray(shape=(3, 1), data={(0, 0): 1, (2, 0): 3})
    b_array = b.toarray()
    outer = b @ a
    assert outer.shape == (3, 3)

    expected = np.outer(b_array, a)
    actual = (b @ a).toarray()

    assert np.allclose(expected, actual)

    a_dok = dok_ndarray.fromarray(a)
    ba = b @ a_dok
    ba_array = ba.toarray()
    assert np.allclose(ba_array, actual)

