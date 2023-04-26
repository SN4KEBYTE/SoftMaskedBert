import torch
from torch import nn


class BiGRU(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.rnn = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(
            hidden_size * 2,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.rnn(x)

        return self.sigmoid(self.linear(out))
