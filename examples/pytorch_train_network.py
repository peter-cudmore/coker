"""Train a PyTorch network imported into a Coker function graph."""

import torch

from coker import VectorSpace, function
from coker.backends import get_backend_by_name
from coker.backends.pytorch import PytorchBackend


class Regressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    backend = get_backend_by_name("pytorch")
    assert isinstance(backend, PytorchBackend)

    network = Regressor().to(dtype=dtype)
    signature_source = function(
        [VectorSpace("features", 2)],
        lambda features, *_: features,
        backend="pytorch",
    )
    imported_network = backend.import_module(network, signature_source.signature)
    predict = function(
        [VectorSpace("features", 2)],
        lambda features, *_: imported_network(features),
        backend="pytorch",
    )

    features = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
        dtype=dtype,
    )
    targets = torch.tensor(
        [[-1.0, -2.0], [1.0, -1.0], [0.0, 3.0], [3.0, 3.0]],
        dtype=dtype,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.05)

    for _ in range(200):
        optimizer.zero_grad()
        predictions = torch.stack([predict(row) for row in features])
        loss = torch.mean((predictions - targets) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = torch.stack([predict(row) for row in features])
        loss = torch.mean((predictions - targets) ** 2)
    print(f"mean squared error: {loss.item():.6f}")


if __name__ == "__main__":
    main()
