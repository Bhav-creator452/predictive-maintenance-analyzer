
import torch
import torch.nn as nn


class RUL_LSTM(nn.Module):
    """LSTM model that predicts RUL from sensor sequences."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, sequence length, sensor count)
        outputs, _ = self.lstm(x)

        # Representation at the final time step
        last_output = outputs[:, -1, :]

        # One predicted RUL value per sequence
        return self.regressor(last_output).squeeze(-1)
