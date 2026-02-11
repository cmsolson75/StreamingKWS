import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),
            nn.AvgPool2d(2),
            ConvBlock(32, 64),
            nn.AvgPool2d(2),
            ConvBlock(64, 128),
            nn.AvgPool2d(2),
            ConvBlock(128, 256),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
