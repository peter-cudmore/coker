from coker import OP, Function
from coker.backends import get_backend_by_name


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
    if lhs.index in atoms and rhs.index in atoms:
        return

    # Case 2b
    if lhs.index in constants and rhs.index in constants:
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

    elif (
        rhs.index not in atoms
        and lhs.index in atoms
        and nodes[rhs.index][0] == op.MUL
        and nodes[lhs.index][0] == op.MUL
    ):
        _, pll, plr = nodes[lhs.index]
        _, prl, prr = nodes[rhs.index]
        if pll.index in constants and prl.index in constants:
            nodes[lhs.index] = (
                OP.VALUE,
                constants[pll.index] * constants[prl.index],
            )
            nodes[rhs.index] = (OP.MUL, plr, prr)

    #  *(?, x))     -> *(x, ?)      (polynomails)
    #  *(?, ?)      -> *(?,?)       (more polynomials)


def rewrite_graph(function: Function):
    constants = {}

    outputs = {o.index for o in function.output}
    inputs = set(function.tape.input_indicies)
    atoms = inputs.copy()
    work_set = [i for i in range(len(function.tape.nodes)) if i not in inputs]
    nodes = function.tape.nodes
    for i in work_set:
        op, *args = nodes[i]
        if op == OP.VALUE:
            (arg,) = args
            constants[i] = arg
            continue

        if all([a.index in constants for a in args]):
            constants[i] = get_backend_by_name(
                "numpy", set_current=False
            ).call(op, *[constants[a.index] for a in args])
            continue

        if op == OP.MUL:
            try_rewrite_mul(nodes, atoms, constants, i)

    function.tape.nodes = nodes
    return function
