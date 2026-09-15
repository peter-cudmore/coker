import numpy as np
import torch

from coker import VectorSpace
from coker.algebra.ops import Noop
from coker.backends import get_backend_by_name
from coker.dynamics import BoundVector, DynamicsSpec, VariationalProblemBuilder
from coker.dynamics.system import create_dynamics_from_spec
from coker.toolkits.codesign import Minimise


def test_pytorch_fits_bound_vector_parameter(monkeypatch):
    backend = get_backend_by_name("pytorch", set_current=False)
    monkeypatch.setattr(backend, "device", torch.device("cpu"))
    monkeypatch.setattr(backend, "dtype", torch.float64)
    system = create_dynamics_from_spec(
        DynamicsSpec(
            inputs=Noop(),
            parameters=(VectorSpace("gain", 1),),
            algebraic=None,
            initial_conditions=lambda _z, _u, _p: (0.0, None),
            dynamics=lambda _t, _x, _z, _u, p: p[0],
            constraints=Noop(),
            outputs=lambda _t, x, _z, _u, _p, _q: x,
            quadratures=Noop(),
        ),
        backend="pytorch",
    )
    with VariationalProblemBuilder(
        system,
        t_final=1.0,
        parameters=[
            BoundVector(
                "gain",
                lower_bound=[-1.0],
                upper_bound=[1.0],
                guess=[0.0],
            )
        ],
        backend="pytorch",
    ) as builder:
        problem = builder.build(
            Minimise((builder.output(builder.t_final)[0] - 0.5) ** 2)
        )

    solution = problem()
    np.testing.assert_allclose(
        solution.parameter_blocks["gain"], [0.5], atol=1e-2
    )
