import numpy as np

from coker.dynamics.transcription.collocation import (
    InterpolatingPoly,
    InterpolatingPolyCollection,
)
from coker.dynamics.variational.solution import VariationalSolution


def test_mapped_solution_output_uses_numeric_parameter_vector():
    path = InterpolatingPolyCollection(
        [InterpolatingPoly(1, (0.0, 1.0), 1, np.array([2.0, 2.0]))]
    )

    def output(_time, state, _algebraic, _control, parameters, _quadrature):
        return state + parameters

    solution = VariationalSolution.from_solver(
        cost=0.0,
        path=path,
        projectors=(np.eye(1), None, None),
        control_solutions=[],
        parameters={"offset": 3.0},
        output=output,
        parameter_vector=np.array([3.0]),
    )

    np.testing.assert_allclose(solution.to_poly()(0.5), np.array([5.0]))
