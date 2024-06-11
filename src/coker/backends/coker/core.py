import dataclasses

from collections import defaultdict
from typing import Tuple, Type, Optional, Iterator, List,Set, Union, Dict
import numpy as np

import coker
from coker import Tensor, Tracer, OP, Kernel
from coker.backends.backend import Backend, ArrayLike, get_backend_by_name


class dok_ndarray:
    def __init__(self, shape, keys=None):
        self.shape = shape
        self.keys = keys if keys is not None else {}

    def __setitem__(self, key, value):
        self.keys[key] = value

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key, )
        return self.keys[key]

    def clone(self) -> 'dok_ndarray':
        return dok_ndarray(shape=self.shape, keys=self.keys.copy())


    def toarray(self):
        m = np.zeros(shape=self.shape)
        for k, v in self.keys.items():
            m[k] = v

        return m

    @staticmethod
    def fromarray(other: np.ndarray):
        shape = other.shape
        keys = {
            item.index: item[0]
            for item in np.nditer(other, flags=['multi_index'])
            if float(item[0]) != 0.0
        }
        return dok_ndarray(shape, keys)

    def __neg__(self):
        keys = {k: -v for k,v in self.keys.items()}
        return dok_ndarray(shape=self.shape, keys=keys)

    def __mul__(self, other):
        assert isinstance(other, (float, int))

        if other == 0:
            return dok_ndarray(shape=self.shape, keys={})

        keys = {
            k: v * other for k, v in self.keys.items()
        }
        return dok_ndarray(shape=self.shape, keys=keys)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, dok_ndarray):
            keys = self.keys.copy()
            for k in other.keys.items():
                if k in keys:
                    keys[k] += other[k]
                else:
                    keys[k] = other[k]
        else:
            raise NotImplementedError()
        return dok_ndarray(shape=self.shape, keys=keys)

    def __sub__(self, other):
        if isinstance(other, dok_ndarray):
            keys = self.keys.copy()
            for k in other.keys.items():
                if k in keys:
                    keys[k] -= other[k]
                else:
                    keys[k] = -other[k]
        else:
            raise NotImplementedError()
        return dok_ndarray(shape=self.shape, keys=keys)

    def __matmul__(self, other):
        assert self.shape[-1] == other.shape[0]
        keys = {}
        for k, v in self.keys.items():
            *k_prime, i = k
            k_prime = tuple(k_prime)
            if k_prime in keys:
                keys[k_prime] += v * other[i]
            else:
                keys[k_prime] = v * other[i]

        shape = (*self.shape[:-1], *other.shape[1:])

        return dok_ndarray(shape=shape, keys=keys)



# Key Assumptions
# - Inputs are a tuple of nd arrays of known size
# - Outputs are ndarrays of known size

# Each Layer i is represented by

# Input Layer:
# Y_0 = P_jX_j                              as a flat vector

# Z_i = T_i(Pz_i Y_{i-1} + R_i)             as a flat vector
# Y_i = sum_j P^j ([Y, Z]^j)                as a flat vector

# Ouptut Layers:
# g_j = B_j^k Y_k                           as an ND array

# Pz_i projects Y_{i-1} into Dom(T_i)
# P^j is a sequence of tensors of increasing order

# B_j is the jth output, which is a mapping from Y_0 x Y_1 x Y_2 ...
# to


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


def index_tuple(dim: coker.Dimension, row_major=True) -> Iterator[Tuple[int, ...]]:
    mods = []

    count = 1
    iterator = reversed(dim) if row_major else iter(dim)

    for d in iterator:
        count = d * count
        mods.insert(0, count)

    for i in range(count):
        yield tuple(i % m for m in mods)


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



@dataclasses.dataclass
class Row:
    """
    Z = [IX | sum(Iota_i f_i (Pi_i X)) ]
    Y = AZ + B(Z, Z) + C
    """
    layer: int
    # Quadratic Part
    linear:    Optional[dok_ndarray] = None # Matrix    dim_z -> dim_y
    quadratic: Optional[dok_ndarray] = None  # Symmetric Tensor dim_z * dim_z -> dim_y
    constant: Optional[dok_ndarray] = None  # Vector dim_y

    @property
    def dimension(self):
        d = [
            m.shape[0]
            for m in {self.constant, self.linear, self.quadratic}
            if m is not None
        ]
        assert len(d) > 0

        if len(d) != 1:
            assert all(d_i == d[0] for d_i in d[1:])
        return d[0]

    def __add__(self, other):

        if isinstance(other, scalar):
            assert self.dimension == 1
            v = dok_ndarray(shape=(1, ), keys={(0,): other})
            constant = self.constant + v if self.constant is not None else v

            return Row(
                self.layer, constant=constant, linear=self.linear, quadratic=self.quadratic
            )

        assert isinstance(other, Row), f"Can't add type {type(other)} to a row"
        assert other.layer == self.layer
        # Z_3 = [X | Z_1 | Z_2]
        # dim (Z_3) = dim(Z_1) + dim(Z_2) - 2 dim(X)

        # A_3 = [A_x1 + A_x2 | Az_1 | Az_2]
        # B^i_3 = [B^i_x1 + B^i_x2  | B^i_z1 | B^i_z2]
        #
        if self.constant is None:
            constant = other.constant.clone()
        elif other.constant is None:
            constant = self.constant.clone()
        else:
            constant = self.constant + other.constant

        if self.linear is None:
            linear = other.linear.clone()
        elif other.linear is None:
            linear = self.linear.clone()
        else:
            linear = self.linear + other.linear

        if self.quadratic is None:
            quadratic = other.quadratic.clone()
        elif other.quadratic is None:
            quadratic = self.quadratic.clone()
        else:
            quadratic = self.quadratic + other.quadratic

        return Row(layer=self.layer, linear=linear, quadratic=quadratic, constant=constant)

    def __rmul__(self, other: Union['scalar']):
        assert isinstance(other, scalar)

        constant = other * self.constant if self.constant is not None else None
        linear = other * self.linear if self.linear is not None else None
        quadratic = other * self.quadratic if self.quadratic is not None else None

        return Row(
            self.layer, constant=constant, linear=linear, quadratic=quadratic
        )

    def __rmatmul__(self, other):
        constant = other @ self.constant if self.constant is not None else None
        linear = other @ self.linear if self.linear is not None else None
        quadratic = other @ self.quadratic if self.quadratic is not None else None

        return Row(
            self.layer, constant=constant, linear=linear, quadratic=quadratic
        )

    def __sub__(self, other):
        assert isinstance(other, Row)
        assert other.layer == self.layer
        # Z_3 = [X | Z_1 | Z_2]
        # dim (Z_3) = dim(Z_1) + dim(Z_2) - 2 dim(X)

        # A_3 = [A_x1 + A_x2 | Az_1 | Az_2]
        # B^i_3 = [B^i_x1 + B^i_x2  | B^i_z1 | B^i_z2]
        #
        if self.constant is None:
            constant = -other.constant.clone()
        elif other.constant is None:
            constant = self.constant.clone()
        else:
            constant = self.constant - other.constant

        if self.linear is None:
            linear = -other.linear.clone()
        elif other.linear is None:
            linear = self.linear.clone()
        else:
            linear = self.linear - other.linear

        if self.quadratic is None:
            quadratic = -other.quadratic.clone()
        elif other.quadratic is None:
            quadratic = self.quadratic.clone()
        else:
            quadratic = self.quadratic - other.quadratic

        return Row(layer=self.layer, linear=linear, quadratic=quadratic, constant=constant)

    @property
    def is_constant(self):
        return self.linear is None and self.quadratic is None


class Layer:
    #
    # mapping Y = F(X)
    __slots__ = (
        'input_dimension',          # int
        'output_dimension',         # int
        'constant',                 # Dense Vector
        'linear',                   # Sparse Matrix
        'quadratic',                # Sparse Tensor
        'nonlinear_projections',    # List of Sparse Tensors
        'nonlinear_functions',      # List of nonlinear functions
        'nonlinear_inclusions'      # List of Sparse Tensors
    )

    def __init__(self, input_dimension: int):
        self.input_dimension = input_dimension
        self.output_dimension = 0
        self.constant = None
        self.quadratic = None
        self.nonlinear_functions = None
        self.nonlinear_inclusions = None
        self.nonlinear_projections = None


class InputLayer:
    def __init__(self):

        """ maps such that arg_i = map_i(x)

        - For scalars; map_i is a linear operator
        - For vector; map_i is a matrix


        """
        self.vec_to_arg_maps: List[Set[Tuple[int,...]]] = []
        self.out_shape = []
        self.dimension = 0

    def add_input(self, dim: coker.Dimension) -> int:
        idx = len(self.vec_to_arg_maps)
        if dim.is_scalar():
            self.vec_to_arg_maps.append({(0, self.dimension)})
            self.dimension += 1
            self.out_shape.append((1, ))
        else:
            basis = set()
            for i, idx in enumerate(index_tuple(dim, row_major=False)):
                # for example, a vector would have an index tuple
                # of idx = (row, ) so the basis becomes (row, d).
                # We interpret this as a matrix
                # For a matrix, we have bases (row, col, d)

                basis.add((*idx, self.dimension))
                self.dimension += 1
            self.vec_to_arg_maps.append(basis)
            self.out_shape.append((*dim, ))

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

        def vec(item):
            if isinstance(item, np.ndarray):
                return item.flatten(order='F')
            if isinstance(item, (int, float)):
                return np.array([item])
            raise NotImplementedError()
        x = np.concatenate([vec(a) for a in args])
        return x


def add(lhs: Union[Row, dok_ndarray], rhs: Union[Row, dok_ndarray]):
    if isinstance(lhs, Row) and isinstance(rhs, dok_ndarray):
        r = Row(layer=lhs.layer, constant=rhs)
        return lhs + r

    if isinstance(rhs, Row) and isinstance(lhs, dok_ndarray):
        r = Row(layer=rhs.layer, constant=lhs)
        return rhs + r

    return lhs + rhs

def sub(lhs: Union[Row, dok_ndarray], rhs: Union[Row, dok_ndarray]):
    if isinstance(lhs, Row) and isinstance(rhs, dok_ndarray):
        r = Row(layer=lhs.layer, constant=rhs)
        return lhs - r

    if isinstance(rhs, Row) and isinstance(lhs, dok_ndarray):
        r = Row(layer=rhs.layer, constant=lhs)
        return r - rhs

    return lhs - rhs

scalar = (float, int)


def mul(lhs: Union[Row, dok_ndarray], rhs: Union[Row, dok_ndarray]):

    if isinstance(lhs, scalar):
        return lhs * rhs

    if isinstance(rhs, scalar):
        return rhs * lhs

    raise NotImplementedError()


def matmul(lhs: Union[Row, dok_ndarray], rhs: Union[Row, dok_ndarray]):
    if isinstance(lhs, dok_ndarray):
        return lhs @ rhs

    raise NotImplementedError()


ops = {
    OP.ADD: add,
    OP.SUB: sub,
    OP.NEG: lambda arg: -arg,
    OP.MUL: mul,
    OP.MATMUL: matmul
}


def reverse_graph(graph: coker.Tape, outputs: List[Tracer]):

    dependent = defaultdict(set)

    for idx, node in enumerate(graph.nodes):
        if isinstance(node, Tracer):
            continue

        op, *args = node
        if op == OP.VALUE:
            continue
        for a in args:
            dependent[a.index].add(idx)

    return dependent


def trace_deps(graph: coker.Tape, outputs: List[Tracer]):

    deps = []
    quadratic_terms = []
    nonlinear_terms = []
    for idx, node in enumerate(graph.nodes):
        if idx in graph.input_indicies:
            deps.append({"x"})
            continue

        op, *args = node
        if op == OP.VALUE:
            deps.append(set())
            continue

        sets = [deps[arg.index] for arg in args if deps[arg.index]]

        if op in {OP.ADD, OP.SUB, OP.NEG, OP.TRANSPOSE}:
            this_deps = set()
            for s in sets:
                this_deps |= s
            deps.append(this_deps)

        elif op in (OP.MATMUL, op.MUL, op.DOT, OP.DIV, OP.CROSS):
            if len(sets) <= 1:
                this_deps = set()
                for s in sets:
                    this_deps |= s
                deps.append(this_deps)
            else:
                deps.append({f"q_{len(quadratic_terms)}"})
                quadratic_terms.append({a.index for a in args})

        else:
            deps.append(f"n_{len(nonlinear_terms)}")
            nonlinear_terms.append({a.index for a in args})

    return deps, quadratic_terms, nonlinear_terms


def sort_graph(kernel: Kernel):

    work_stack = [o.index for o in kernel.output]
    dependencies = {o.index: {o.index : 1} for o in kernel.output}
    graph = kernel.tape
    seen = set(work_stack)

    # backwards pass -> which outputs does each node influence
    while work_stack:
        i = work_stack.pop(0)
        if isinstance(graph.nodes[i], Tracer):
            # got an input
            continue

        op, *args = graph.nodes[i]
        if op == OP.VALUE:
            # leaf
            continue

        for a in args:
            idx = a.index
            if idx in dependencies:
                if i in dependencies[idx]:
                    dependencies[idx][i] += 1
                else:
                    dependencies[idx][i] = 1
            else:
                dependencies[idx] = {i: 1}

            if idx not in seen:
                work_stack.append(a.index)

    


class MatSet:
    def __init__(self):
        pass

    @staticmethod
    def from_linear(sparse_array):
        self = MatSet()
        self.linear = sparse_array
        self.dimension = sparse_array.shape[0]

        return self



def extract_layers(kernel: Kernel):
    graph = kernel.tape
    workspace = {}
    deps = {}
    input_layer = InputLayer()
    arg_layer_indicies = [
        (input_layer.add_input(graph.dim[idx]), idx)
        for idx in graph.input_indicies
    ]

    layer_inputs = set()
    for (layer_index, workspace_index) in arg_layer_indicies:
        matset = MatSet.from_linear(input_layer.get_projection(layer_index))
        workspace[workspace_index] = matset
        layer_inputs.add(workspace_index)

    for idx, node in enumerate(graph.nodes):
        if idx in graph.input_indicies:
            continue

        op, *args = node

        if op == OP.VALUE:
            arg, = args
            workspace[idx] = arg
            continue

        items = [workspace[a.index] for a in args]
        if op == OP.ADD:
            lhs, rhs = items
            workspace[idx] = lhs + rhs

        if op == OP.SUB:
            lhs, rhs = items
            workspace[idx] = lhs - rhs

        if op == OP.NEG:
            item, = items
            workspace[idx] = -item

        if op == OP.MUL:
            lhs, rhs = items
            # cases:
            # - one is an input, one is a constant
            #   -> linear matset
            # - both are inputs
            # -



def _extract_layers(graph: coker.Tape, outputs: List[Tracer]) -> List[Layer]:

    workspace = {}

    input_map = []

    work_list = [i for i in range(len(graph.nodes)) if i not in workspace]

    # inputs x become x = x_i * basis(i)  where x_i is a scalar symbol
    # in the downward pass we
    # Incrementally stack up:
    #   y^0_i = k_i + <r_i, x> + x.T @ Q_i @ x + <n, f(x)>
    #
    #   adding rows while we can
    #   we stop a layer when we have not more vialble rows
    #   a row is viable if
    #   - it is a nonlinear function of the (projected) input
    #   - it is a linear function of the input or a linear row

    # once all viable inputs have been complete, the output y_i
    # is treated as the input of the next layer

    # Once we've hit all outputs, we do a backwards pass to
    # eliminate rows that are not used in either the output, or the input of
    # the next layer
    input_layer = InputLayer()
    arg_layer_indicies = [
        (input_layer.add_input(graph.dim[idx]), idx)
        for idx in graph.input_indicies
    ]

    layer_inputs = set()
    for (layer_index, workspace_index) in arg_layer_indicies:
        workspace[workspace_index] = Row(
            layer=0, linear=input_layer.get_projection(layer_index)
        )
        layer_inputs.add(workspace_index)

    layers = [

    ]

    for idx in work_list:
        if idx in graph.input_indicies:
            continue
        op, *args = graph.nodes[idx]

        if op == OP.VALUE:
            arg, = args
            if isinstance(arg, np.ndarray):
                workspace[idx] = ('const', dok_ndarray.fromarray(arg))
            elif isinstance(arg, scalar):
                workspace[idx] = ('const', arg)
            else:
                raise NotImplementedError(type(arg))
            continue

        assert isinstance(op, OP), \
            f"Unhandled type {type(op)} in graph at node {idx}"
        assert op != OP.VALUE

        # All that is left are algebraic operations...

        rows = [workspace[a.index] for a in args]
        next_row = ops[op](*rows)

        workspace[idx] = next_row

    output_rows = [
        workspace[o.index]  for o in outputs
    ]


    return [
        input_layer
    ]


def evaluate_inner(graph, args, outputs, workspace: dict):
    for index, arg in zip(graph.input_indicies, args):
        workspace[index] = arg

    work_list = [i for i in range(len(graph.nodes)) if i not in workspace]

    def get_node(node):

        if isinstance(node, Tracer):
            return workspace[node.index]
        return backend.from_native(node)

    for w in work_list:
        op, *nodes = graph.nodes[w]

        try:
            args = [get_node(n) for n in nodes]
            if op == OP.VALUE:
                value, = args
            else:
                value = backend.call(op, *[get_node(n) for n in nodes])

        except KeyError as ex:
            raise NotImplementedError(f"Op {op} not implemented in python")
        workspace[w] = backend.reshape(value, graph.dim[w])

    return backend.to_native(workspace[outputs.index])


def generate_labels(kernel: Kernel):

    tape = kernel.tape
    constants = set()
    tape_indegree = []
    tape_outdegree = []

    labeled_nodes = set()             # output of these nodes are considered 'new variables'

    for i, node in enumerate(tape.nodes):
        tape_outdegree.append(0)

        if isinstance(node, Tracer):
            tape_indegree.append(0)
            labeled_nodes.add(i)
            continue

        op, *args = node
        if op == OP.VALUE:
            constants.add(i)
            tape_indegree.append(0)
            continue

        indices = [a.index for a in args]
        in_nodes = [idx for idx in indices if idx not in constants]

        if not in_nodes:
            constants.add(i)
            tape_indegree.append(0)
            continue

        # non-constant op
        #
        for j in in_nodes:
            tape_outdegree[j] += 1
        tape_indegree.append(len(in_nodes))

        # Strictly Linear nodes
        if op in {op.ADD, OP.SUB, OP.NEG}:
            continue

        # Multi-linear terms that mayne nonlinear
        if op in {OP.MUL, OP.CROSS, OP.MATMUL, OP.DOT} and len(in_nodes) == 1:
            continue

        labeled_nodes.add(i)

    for i, degree in enumerate(tape_outdegree):
        if degree >= 2:
            labeled_nodes.add(i)

    return tape_indegree, tape_outdegree, labeled_nodes, constants


@dataclasses.dataclass
class Input:
    index: int


@dataclasses.dataclass
class Unknown:
    index: int


class EdgeBuilder:
    def __init__(self, output, dimension):
        self.output = output
        self.input = set()
        self.unknowns = set()
        self.constants = {}
        self.dimension = dimension
        self.ops = {}

    def finalise(self):
        assert self.input
        assert not self.unknowns
        return self

    def can_finalise(self) -> bool:
        return self.input and not self.unknowns

    def push_op(self, index, op: OP, *args: Union[Input, Unknown, np.ndarray, *scalar]):
        assert op != OP.VALUE
        self.ops[index] = (op, *args)
        if index in self.unknowns:
            self.unknowns.remove(index)
        for a in args:
            if isinstance(a, Input):
                self.input |= {a.index}
            if isinstance(a, Unknown):
                self.unknowns.add(a.index)


def try_rewrite_mul(nodes: list, atoms, constants, i):

    op, lhs, rhs = nodes[i]
    assert op == OP.MUL


    # case 1
    #  *(x,b )         ->  *(b, x)
    if lhs.index in atoms and rhs.index in constants:
        nodes[i] = (op, rhs, lhs)
        return

    # Case 2
    #  *(x,x)          ->  *(x, x)
    if (lhs.index in atoms and rhs.index in atoms):
        return

    # Case 2b
    if (lhs.index in constants and rhs.index in constants):
        constants.add(i)
        return

    # Case 3
    #  *((a, x), x)    -> *(a,  *(x, x))
    if lhs.index not in atoms and nodes[lhs.index][0] == op.MUL:
        _, parent_lhs, parent_rhs = nodes[lhs.index]
        #  n_lhs = *(a, x) -> n_lhs = a
        #
        if parent_lhs.index in constants and parent_rhs.index in atoms:
            nodes[lhs.index] = (op.MUL, parent_rhs, rhs)
            nodes[i] = (op, parent_lhs, lhs)

    # Case 4
    #  *(x, (b, x))    -> *(b,  *(x, x))
    elif rhs.index not in atoms and nodes[rhs.index][0] == op.MUL:
        _, parent_lhs, parent_rhs = nodes[rhs.index]
        if parent_lhs.index in constants and parent_rhs.index in atoms:
            nodes[rhs.index] = (op.MUL, lhs, parent_rhs)
            nodes[i] = (op, parent_lhs, rhs)

    # Case 5
    #  *((a, x), *(b,x)) -> *(ab, *(x,x))

    elif (rhs.index not in atoms and lhs.index in atoms
            and nodes[rhs.index][0] == op.MUL and nodes[lhs.index][0] == op.MUL):
        _, pll, plr = nodes[lhs.index]
        _, prl, prr = nodes[rhs.index]
        if pll.index in constants and prl.index in constants:
            nodes[lhs.index] = (OP.VALUE, constants[pll.index] * constants[prl.index])
            nodes[rhs.index] = (OP.MUL, plr, prr)

    #  *(?, x))     -> *(x, ?)      (polynomails)
    #  *(?, ?)      -> *(?,?)       (more polynomials)


def rewrite_graph(kernel: Kernel):
    constants = {}

    _, _, labels, _ = generate_labels(kernel)

    outputs = {o.index for o in kernel.output}
    inputs = set(kernel.tape.input_indicies)
    atoms = inputs.copy()
    work_set = [
        i for i in range(len(kernel.tape.nodes)) if i not in inputs
    ]
    nodes = kernel.tape.nodes
    for i in work_set:
        op, *args = nodes[i]
        if op == OP.VALUE:
            arg, = args
            constants[i] = arg
            continue

        if all([a.index in constants for a in args]):
            constants[i] = get_backend_by_name('numpy').call(op, *[constants[a.index] for a in args])
            continue

        if op == OP.MUL:
            try_rewrite_mul(nodes, atoms, constants, i)

    kernel.tape.nodes = nodes
    return kernel


def build_edge_graph(kernel: Kernel):

    tape_indegree, tape_outdegree, labels, constants_idx = generate_labels(kernel)
    tape = kernel.tape

    workset = [o.index for o in kernel.output]
    seen = set(workset)

    constants = {
        idx: tape.nodes[idx][1] for idx in constants_idx if tape.nodes[idx][0] == OP.VALUE
    }
    seen |= set(constants.keys())

    # nodes are Mutli-linear or Nonlinear ops
    # key is output index,
    nodes: Dict[int, Tuple[int, ...]] = {}

    # edges are mappings y = f(x)

    edges: List[EdgeBuilder] = []

    builder: Union[EdgeBuilder, None] = None

    def map_args(a: Tracer):
        idx = a.index
        if idx in constants:
            return constants[idx]

        if idx in labels or idx in tape.input_indicies:
            return Input(idx)

        return Unknown(idx)

    while workset:
        # pull from the top of the stack
        this_idx = workset.pop()

        if this_idx in tape.input_indicies:
            # we have reached an input.
            if builder:
                edges.append(builder.finalise())
                builder = None
            continue

        op, *args = tape.nodes[this_idx]
        if this_idx in labels:
            nodes[this_idx] = (op, *[map_args(a) for a in args])
            for a in args:
                if a.index not in seen:
                    workset.insert(0, a.index)
            continue

        elif all([a.index in constants for a in args]):
            constants[this_idx] = get_backend_by_name('numpy').call(op, *[map_args(a.index) for a in args])
            continue

        if not builder:
            dim: coker.Dimension = tape.dim[this_idx]
            builder = EdgeBuilder(output=this_idx, dimension=dim.flat())

        assert op != OP.VALUE

        builder.push_op(this_idx, op, *[map_args(a) for a in args])

        for arg in args:
            if arg.index in seen:
                continue
            a_idx = arg.index
            if a_idx in labels:
                # we add it to the back of the workset, so that the edge
                # gets completed before we move on
                workset.insert(0, a_idx)
            else:
                workset.append(a_idx)
            seen.add(arg.index)

        if builder.can_finalise():
            edges.append(builder.finalise())
            builder = None

    return nodes, edges


def assign_layers_to_edge_graph(kernel: Kernel):

    # the 'layer' is the max distance from the ouptut

    nodes, edges = build_edge_graph(kernel)

    work_set = list(nodes.keys()) + [o.index for o in kernel.output]
    distance = {k: 0 for k in (set(work_set) | set(kernel.tape.input_indicies))}

    adjacency = {edge.output: edge.input for edge in edges}

    while work_set:
        this_idx = work_set.pop()
        parent = adjacency[this_idx]
        distance[parent] = max(distance[this_idx] + 1, distance[parent])
        if parent not in kernel.tape.input_indicies:
            work_set.insert(0, parent)

    return nodes, edges, distance


def parse_tape_into_opgraph(kernel: Kernel):

    work_set = [o.index for o in kernel.output]


    def complete_edge(last_node):
        pass

    while work_set:
        i = work_set.pop(0)

        if i in tape.input_indicies:
            labels[i] = f"x_{i}"
            complete_edge(i)
            continue








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


