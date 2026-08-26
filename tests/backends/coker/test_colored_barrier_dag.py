import numpy as np

from coker import VectorSpace, function
from coker.backends.coker.unfused_lowering import analyze_colored_barrier_dag


def _analysis(implementation):
    fn = function(
        [VectorSpace("x", 2)], implementation=implementation, backend="coker"
    )
    return analyze_colored_barrier_dag(fn)


def test_algebraic_chain_collapses_to_one_bilinear_region():
    dag = _analysis(lambda x: (x * x + 2.0 * x) - x)
    bilinear = [node for node in dag.nodes if node.color == "bilinear"]
    assert len(bilinear) == 1
    assert len(bilinear[0].members) >= 3
    assert dag.critical_path == 1


def test_independent_generic_branches_are_one_ready_antichain():
    dag = _analysis(lambda x: np.sin(x) + np.cos(x))
    generic = [node.node_id for node in dag.nodes if node.color == "generic"]
    assert len(generic) == 2
    assert generic == list(dag.schedule[0])
    assert dag.color_switches == 1



def test_generic_mediated_algebraic_component_is_split_and_acyclic():
    def implementation(x):
        algebraic = x * x
        return np.sin(algebraic) * algebraic
    dag = _analysis(implementation)
    positions = {
        node_id: ordinal
        for ordinal, batch in enumerate(dag.schedule)
        for node_id in batch
    }
    assert set(positions) == {node.node_id for node in dag.nodes}
    assert all(
        positions[dependency] < positions[node.node_id]
        for node in dag.nodes
        for dependency in node.dependencies
    )
    assert len([node for node in dag.nodes if node.color == "bilinear"]) >= 2

def test_colored_schedule_is_stable_across_repeated_analysis():
    implementation = lambda x: np.sin(x * x) + np.cos(x * x)
    first = _analysis(implementation)
    second = _analysis(implementation)
    assert first.nodes == second.nodes
    assert first.schedule == second.schedule
    assert first.color_switches == second.color_switches
