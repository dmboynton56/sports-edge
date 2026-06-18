"""PyTorch modules for MLB batter home-run probability experiments."""

from __future__ import annotations

import torch
from torch import nn


class MlbHrWideDeepNN(nn.Module):
    """Wide/deep binary classifier for batter-pitcher HR matchups.

    Continuous, leakage-safe rate features provide the wide signal. Entity
    embeddings let the model learn reusable batter, pitcher, and team matchup
    structure when enough history exists.
    """

    def __init__(
        self,
        categorical_cardinalities: list[int],
        num_continuous: int,
        *,
        embedding_dim: int = 16,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.18,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, min(embedding_dim, max(4, int(cardinality**0.25 * 4))))
                for cardinality in categorical_cardinalities
            ]
        )
        embedding_width = sum(embedding.embedding_dim for embedding in self.embeddings)
        self.continuous_bn = nn.BatchNorm1d(num_continuous)

        layers: list[nn.Module] = []
        input_dim = embedding_width + num_continuous
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, categorical_x: torch.Tensor, continuous_x: torch.Tensor) -> torch.Tensor:
        embeddings = [
            embedding(categorical_x[:, idx])
            for idx, embedding in enumerate(self.embeddings)
        ]
        continuous = self.continuous_bn(continuous_x)
        x = torch.cat([*embeddings, continuous], dim=1)
        return self.output(self.mlp(x)).squeeze(1)
