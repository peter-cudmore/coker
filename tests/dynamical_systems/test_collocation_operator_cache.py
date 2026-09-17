import numpy as np

from coker.dynamics.transcription import collocation


def test_reference_operator_caches_are_scoped(monkeypatch):
    lgr_points = collocation.lgr_points
    calls = 0

    def count_lgr_points(degree):
        nonlocal calls
        calls += 1
        return lgr_points(degree)

    monkeypatch.setattr(collocation, "lgr_points", count_lgr_points)
    first_cache = collocation._reference_operator_cache()
    second_cache = collocation._reference_operator_cache()

    first = first_cache(4)
    assert first_cache(4) is first
    assert second_cache(4) is not first
    assert calls == 2


def test_returned_operator_arrays_are_independent():
    first = collocation.generate_discritisation_operators((0.0, 2.0), 4)
    expected = collocation.generate_discritisation_operators((3.0, 5.0), 4)

    first_nodes, _, first_bases, first_derivative, first_weights = first
    first_nodes.fill(np.nan)
    first_bases[0].fill(np.nan)
    first_derivative[0].fill(np.nan)
    first_weights.fill(np.nan)
    first_bases.clear()
    first_derivative.clear()

    actual_nodes, _, actual_bases, actual_derivative, actual_weights = (
        collocation.generate_discritisation_operators((3.0, 5.0), 4)
    )
    (
        expected_nodes,
        _,
        expected_bases,
        expected_derivative,
        expected_weights,
    ) = expected

    np.testing.assert_allclose(actual_nodes, expected_nodes)
    np.testing.assert_allclose(actual_bases, expected_bases)
    np.testing.assert_allclose(actual_derivative, expected_derivative)
    np.testing.assert_allclose(actual_weights, expected_weights)


def test_reference_operators_apply_affine_interval_scaling():
    reference = collocation.generate_discritisation_operators((-1.0, 1.0), 5)
    scaled = collocation.generate_discritisation_operators((2.0, 8.0), 5)

    (
        reference_nodes,
        reference_time,
        reference_bases,
        reference_derivative,
        reference_weights,
    ) = reference
    (
        scaled_nodes,
        scaled_time,
        scaled_bases,
        scaled_derivative,
        scaled_weights,
    ) = scaled

    np.testing.assert_allclose(scaled_nodes, reference_nodes)
    np.testing.assert_allclose(scaled_bases, reference_bases)
    np.testing.assert_allclose(
        np.asarray(scaled_derivative) * 3.0, np.asarray(reference_derivative)
    )
    np.testing.assert_allclose(scaled_weights, reference_weights * 3.0)
    np.testing.assert_allclose(
        reference_time(reference_nodes), reference_nodes
    )
    np.testing.assert_allclose(scaled_time(np.array([-1.0, 1.0])), [2.0, 8.0])

    values = scaled_time(scaled_nodes) ** 2
    np.testing.assert_allclose(
        np.asarray(scaled_derivative) @ values, 2.0 * scaled_time(scaled_nodes)
    )
    np.testing.assert_allclose(scaled_weights @ values, [168.0])
