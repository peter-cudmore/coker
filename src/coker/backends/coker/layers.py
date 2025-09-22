from coker import Dimension, OP, scalar_types
from coker.backends import get_backend_by_name
from coker.backends.coker.sparse_tensor import dok_ndarray
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.weights import BilinearWeights

from typing import List, Set, Tuple, Dict


import numpy as np


def vec(item):
    if isinstance(item, np.ndarray):
        return item.flatten(order="F")
    if isinstance(item, scalar_types):
        return np.array([item])
    raise NotImplementedError(type(item))


class InputLayer:
    def __init__(self):
        """maps such that arg_i = map_i(x)

        - For scalars; map_i is a linear operator
        - For vector; map_i is a matrix


        """
        self.vec_to_arg_maps: List[Set[Tuple[int, ...]]] = []
        self.out_shape = []
        self.dimension = 0

    def add_input(self, dim: Dimension) -> int:

        idx = len(self.vec_to_arg_maps)
        if dim.is_scalar():
            self.vec_to_arg_maps.append({(0, self.dimension)})
            self.dimension += 1
            self.out_shape.append((1,))
        else:
            basis = set()
            for i, idx in enumerate(dim.index_iterator(row_major=False)):
                # for example, a vector would have an index tuple
                # of idx = (row, ) so the basis becomes (row, d).
                # We interpret this as a matrix
                # For a matrix, we have bases (row, col, d)

                basis.add((*idx, self.dimension))
                self.dimension += 1
            self.vec_to_arg_maps.append(basis)
            self.out_shape.append((*dim,))

        return idx

    def get_projection(self, arg: int):
        # return a matrix that maps
        # M: vec(x) -> arg_i

        nzeros = self.vec_to_arg_maps[arg]
        shape = (*self.out_shape[arg], self.dimension)
        m = dok_ndarray(shape)
        for idx in nzeros:
            m[idx] = 1
        return m

    def __call__(self, *args):
        assert len(args) == len(self.vec_to_arg_maps)

        x = np.concatenate([vec(a) for a in args])
        return x

    def forwards(self, *tangent_space, y=None):
        r"""Reverse mode autodiff

        Here, :math:`y = \sum_j P_j x_j$`

        so  dy = \sum P_j dx_j

        """
        n_args = len(self.vec_to_arg_maps)
        assert len(tangent_space) == 2 * n_args
        dx = tangent_space[n_args : 2 * n_args]

        return np.concatenate([vec(dx_i) for dx_i in dx])


def to_float(x):
    if isinstance(x, (float, int)):
        return float(x)

    assert isinstance(x, np.ndarray)
    x_out = x.flatten()
    assert x_out.shape == (1,)
    return x_out[0]


class OutputLayer:

    def __init__(self):
        self.outputs = {}

    def inputs(self):
        return list(self.outputs.keys())

    def add_output(self, memory: MemorySpec, shape: Dimension):
        assert memory not in self.outputs
        if shape.dim is None:
            self.outputs[memory] = to_float
            return

        def shape_fn(array):
            return np.reshape(array, shape.dim)

        self.outputs[memory] = shape_fn

    def call(self, context: Dict[MemorySpec, np.ndarray]):
        result = [f(context[k]) for k, f in self.outputs.items()]
        if len(result) == 1:
            return result[0]
        return result


class GenericLayerOP:
    def __init__(self, output, op, *weights):
        self.op = op
        self.weights = weights
        self.output = output

        assert all(
            isinstance(w, BilinearWeights) for w in weights
        ), f"Unkown type in {weights}"

    def inputs(self) -> List[MemorySpec]:
        return [w.memory for w in self.weights]

    def outputs(self) -> List[MemorySpec]:
        return [self.output]

    def __call__(self, *x):
        backend = get_backend_by_name("numpy", set_current=False)
        x = [w_i(x_i) for w_i, x_i in zip(self.weights, x)]
        return backend.call(self.op, *x)

    def push_forward(self, *tangent_space):
        n = len(self.weights)
        x, dx = tangent_space[0:n], tangent_space[n:]
        y = self(*x)

        x, dx = zip(
            *[
                w_i.push_forwards(x_i, dx_i)
                for w_i, x_i, dx_i in zip(self.weights, x, dx)
            ]
        )
        df = differentials[self.op](*x)
        if all(isinstance(df_i, (dok_ndarray, np.ndarray)) for df_i in df):
            dy = sum(df_i @ dx_i for df_i, dx_i in zip(df, dx))
        else:
            dy = sum(df_i * dx_i for df_i, dx_i in zip(df, dx))
        return y, dy


class IdentityLayer:
    def __init__(self, memory: MemorySpec, weights: BilinearWeights):
        self.memory = memory
        self.weights = weights

    def __call__(self, x):
        y = self.weights(x)
        return y

    def push_forward(self, x, dx):
        return self.weights.push_forwards(x, dx)

    def inputs(self) -> List[MemorySpec]:
        return [self.weights.memory]

    def outputs(self) -> List[MemorySpec]:
        return [self.memory]


def d_dot(x, y) -> Tuple:
    # d = x.T @ y  = sum(x_iy_i)
    # dd_i = y.T @ dx_i
    # dd_i = x.T @ dy_i
    return y.T, x.T


differentials = {
    OP.SIN: np.cos,
    OP.COS: lambda x: -np.sin(x),
    OP.MUL: lambda a, b: [b, a],
    OP.ADD: lambda a, b: [1, 1],
    OP.SUB: lambda a, b: [1, -1],
    OP.DOT: d_dot,
}
