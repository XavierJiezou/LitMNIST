from torch import Tensor, nn


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        width: int = 28,
        height: int = 28,
        hidden_size: int = 256,
        hidden_layers: int = 1,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.input_size = channels * width * height
        self.output_size = num_classes

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            *[
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            ]
            * hidden_layers,
            nn.Linear(hidden_size, self.output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
