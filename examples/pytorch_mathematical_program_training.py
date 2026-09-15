"""Fit explicit Coker network parameters with a PyTorch program."""

import numpy as np
import torch

from coker.backends import get_backend_by_name
from coker.backends.pytorch import PytorchBackend, PytorchNLPSolverOptions
from coker.toolkits.codesign import Minimise, ProblemBuilder


FEATURES = 2
HIDDEN = 4


def network_expression(features, *parameters):
    weights_1, bias_1, weights_2, bias_2 = parameters
    return (weights_2 @ np.sin(weights_1 @ features + bias_1) + bias_2)[0]


def main() -> None:
    torch.manual_seed(0)
    backend = get_backend_by_name("pytorch")
    assert isinstance(backend, PytorchBackend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float32 if device.type == "cuda" else torch.float64
    numpy_dtype = np.float32 if device.type == "cuda" else np.float64
    backend.device = device
    backend.dtype = torch_dtype
    options = PytorchNLPSolverOptions(
        optimiser_method="Adam",
        optimiser_options={"lr": 0.05},
        inner_iterations=1_000,
    )
    features = np.array(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
        dtype=numpy_dtype,
    )
    targets = np.array([-1.0, 1.0, 0.0, 3.0], dtype=numpy_dtype)

    with ProblemBuilder(solver_options=options) as builder:
        weights_1 = builder.new_variable(
            "weights_1",
            shape=(HIDDEN, FEATURES),
            initial_value=np.random.randn(HIDDEN, FEATURES).astype(
                numpy_dtype
            ),
        )
        bias_1 = builder.new_variable(
            "bias_1",
            shape=HIDDEN,
            initial_value=np.zeros(HIDDEN, dtype=numpy_dtype),
        )
        weights_2 = builder.new_variable(
            "weights_2",
            shape=(1, HIDDEN),
            initial_value=np.random.randn(1, HIDDEN).astype(numpy_dtype),
        )
        bias_2 = builder.new_variable(
            "bias_2",
            shape=1,
            initial_value=np.zeros(1, dtype=numpy_dtype),
        )
        parameters = (weights_1, bias_1, weights_2, bias_2)
        predictions = [
            network_expression(feature, *parameters) for feature in features
        ]
        builder.objective = Minimise(
            sum(
                (prediction - target) ** 2
                for prediction, target in zip(predictions, targets)
            )
            / len(targets)
        )
        builder.outputs = list(parameters)
        program = builder.build("pytorch")

    cost, *_ = program()
    assert program.solve_info is not None
    print(f"mean squared error: {cost:.6f}")
    print(f"optimiser: {program.solve_info.solver}")


if __name__ == "__main__":
    main()
