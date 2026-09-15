"""Train an explicitly parameterized Coker network through PyTorch."""

import numpy as np
import torch

from coker import VectorSpace, function


FEATURES = 2
HIDDEN = 4


def network_expression(features, *parameters):
    weights_1, bias_1, weights_2, bias_2 = parameters
    return (weights_2 @ np.sin(weights_1 @ features + bias_1) + bias_2)[0]


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64

    network = function(
        [
            VectorSpace("features", FEATURES),
            VectorSpace("weights_1", (HIDDEN, FEATURES)),
            VectorSpace("bias_1", HIDDEN),
            VectorSpace("weights_2", (1, HIDDEN)),
            VectorSpace("bias_2", 1),
        ],
        network_expression,
        backend="pytorch",
    )

    features = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
        dtype=dtype,
    )
    targets = torch.tensor([-1.0, 1.0, 0.0, 3.0], dtype=dtype)
    weights_1 = torch.nn.Parameter(torch.randn(HIDDEN, FEATURES, dtype=dtype))
    bias_1 = torch.nn.Parameter(torch.zeros(HIDDEN, dtype=dtype))
    weights_2 = torch.nn.Parameter(torch.randn(1, HIDDEN, dtype=dtype))
    bias_2 = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
    parameters = [weights_1, bias_1, weights_2, bias_2]
    optimizer = torch.optim.Adam(parameters, lr=0.05)

    for _ in range(500):
        optimizer.zero_grad()
        predictions = torch.stack(
            [network(row, *parameters) for row in features]
        )
        loss = torch.mean((predictions - targets) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = torch.stack(
            [network(row, *parameters) for row in features]
        )
        loss = torch.mean((predictions - targets) ** 2)
    print(f"mean squared error: {loss.item():.6f}")


if __name__ == "__main__":
    main()
