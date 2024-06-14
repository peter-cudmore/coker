import dataclasses

from collections import defaultdict
from typing import Tuple, Type, Optional, Iterator, List,Set, Union, Dict
import numpy as np

import coker
from coker import Tensor, Tracer, OP
from coker.backends.backend import Backend, ArrayLike


def flatten_dim(dim: coker.Dimension):
    d = 1
    if not dim.is_scalar():
        for d_i in dim:
            d *= d_i
    return d


@dataclasses.dataclass
class FlatProjection:
    arg_idx: int
    coord: Optional[Tuple[int, ...]]


def to_vec(item, dim):
    if isinstance(item, np.ndarray):
        assert dim == item.shape
        try:
            r, = item.shape
            return item
        except ValueError:
            pass
        return item.flatten(order='F')
    if isinstance(item, (float, complex, int)):
        return np.array([item])
    raise NotImplementedError(type(item))


def to_array(item, dim):
    if isinstance(item, np.ndarray):
        assert dim == item.shape
        try:
            r, = item.shape
            np.reshape(item, newshape=(r, 1))
        except ValueError:
            pass
        return item
    if isinstance(item, (float, complex, int)):
        return np.array([[item]])
    raise NotImplementedError(type(item))



class Projection:
    def __init__(self, domain: int, rnge: int):
        self.domain = domain
        self.range = rnge
        self.dok = set()

    def extend_domain(self, domain: int):
        assert domain > self.domain
        self.domain = domain

    def extend_range(self, r: int):
        assert self.range < r
        self.range = r

    def from_slice(self, range_start: int,  slc:slice):
        assert range_start >= 0
        assert range_start + slc.stop < self.range
        step = slc.step if slc.step else 1
        for i, j in enumerate(range(slc.start, slc.stop, step)):
            self.dok.add((i, j))

    def to_numpy(self):
        m = np.zeros((self.range, self.domain))
        for i, j in self.dok:
            m[i, j] = 1
        return m


def partition_graph(graph: coker.Tape, outputs: List[Tracer]):
    constants = set()
    inputs = set()
    linear_terms = set()
    n = len(graph.nodes)
    layers = [0] * n
    max_layer = 0
    dependents = defaultdict(set)

    for i in range(n):
        if i in graph.input_indicies:
            # argument
            layers[i] = 0
            inputs.add(i)
            continue

        op, *args = graph.nodes[i]
        if op is OP.VALUE or all(arg.index in constants for arg in args):
            constants.add(i)
            layers[i] = 0
            continue

        for arg in args:
            dependents[arg.index].add(i)

        # always linear
        if op in {OP.ADD, OP.SUB, OP.NEG, OP.TRANSPOSE}:
            layers[i] = max(layers[arg.index] for arg in args)
            linear_terms.add(i)

        # maybe linear, maybe quadratic
        elif op in {OP.DOT, OP.MUL, OP.DIV, OP.MATMUL}:
            lhs, rhs = args
            if lhs.index in constants:
                layers[i] = layers[rhs.index]
                linear_terms.add(i)
            elif rhs.index in constants:
                layers[i] = layers[lhs.index]
                linear_terms.add(i)
            else:
                layer = max(layers[lhs.index], layers[rhs.index]) + 1
                if layer > max_layer:
                    max_layer = layer
                layers[i] = layer

        else:
            # OP is nonlinear
            layer = max(layers[arg.index] for arg in args) + 1
            if layer > max_layer:
                max_layer = layer
            layers[i] = layer

    if isinstance(outputs, Tracer):
        layers[outputs.index] = max_layer
    else:
        for i in {o.index for o in outputs}:
            layers[i] = max_layer

    return layers, dependents

class CokerBackend(Backend):
    def __init__(self):
        pass

    def from_native(self, array: ArrayLike) -> Tensor:
        pass

    def to_native(self, array: Tensor) -> ArrayLike:
        pass

    def native_types(self) -> Tuple[Type]:
        pass

    def call(self, op, *args) -> ArrayLike:
        pass

    def evaluate(self, graph, inputs: ArrayLike, outputs: ArrayLike):
        pass

    def reshape(self, array: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
        pass


