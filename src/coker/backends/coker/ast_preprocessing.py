from typing import Set, Dict
from coker import OP, Kernel, Tracer


def generate_output_labels(kernel: Kernel) -> Dict[int, Set[int]]:
    """
    Summary:

        Forward pass through the graph, determining which nodes are
        to be considered "outputs"

        Criteria is either
            a) nonlinear
            b) used as inputs to multiple different nonlinear terms

    """
    tape = kernel.tape
    constants = set()
    tape_outdegree = []

    output_nodes = dict()             # output of these nodes are considered 'new variables'

    for i, node in enumerate(tape.nodes):
        tape_outdegree.append(0)

        if isinstance(node, Tracer):
            output_nodes[i] = set()
            continue

        op, *args = node
        if op == OP.VALUE:
            constants.add(i)
            continue

        indices = [a.index for a in args]
        in_nodes = [idx for idx in indices if idx not in constants]

        if not in_nodes:
            constants.add(i)
            continue

        # non-constant op
        #
        for j in in_nodes:
            tape_outdegree[j] += 1

        # Strictly Linear nodes
        if op in {op.ADD, OP.SUB, OP.NEG}:
            continue

        # Multi-linear terms that mayne nonlinear
        if op in {OP.MUL, OP.CROSS, OP.MATMUL, OP.DOT} and len(in_nodes) == 1:
            continue

        output_nodes[i] = set()

    for i, degree in enumerate(tape_outdegree):
        if degree >= 2:
            output_nodes[i] = set()

    def recurse_node(i, n):
        if n in tape.input_indicies:
            output_nodes[i].add(n)
            return
        op, *args = tape.nodes[n]

        if op == OP.VALUE:
            return {}

        for a in args:
            if a.index in output_nodes:
                output_nodes[i].add(a.index)
            else:
                recurse_node(i, a.index)

    for o in output_nodes.keys():
        recurse_node(o, o)

    return output_nodes, constants



