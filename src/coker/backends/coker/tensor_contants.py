from coker.backends.coker.sparse_tensor import (
    dok_ndarray,
    tensor_vector_product,
)


def hat(a):
    return tensor_vector_product(levi_civita_3_tensor, a, axis=1)


levi_civita_3_tensor = dok_ndarray(
    (3, 3, 3),
    {
        (0, 1, 2): 1,
        (2, 0, 1): 1,
        (1, 2, 0): 1,
        (0, 2, 1): -1,
        (1, 0, 2): -1,
        (2, 1, 0): -1,
    },
)


def dot_tensor(n):
    return dok_ndarray((1, n, n), data={(0, i, i): 1 for i in range(n)})
