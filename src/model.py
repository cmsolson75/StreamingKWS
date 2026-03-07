import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal, Callable


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
    def __init__(self, num_classes: int, p: float = 0.5):
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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class TCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=9,
            stride=stride,
            padding=4,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.skip_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )

        self.skip_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_bn(self.skip_conv(x))

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + skip)


class TCResNet8(nn.Module):
    def __init__(self, num_classes: int, n_mels: int = 64, k: int = 1, p: float = 0.5):
        super().__init__()

        def scale_width(n):
            return int(n * k)

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=n_mels,
                out_channels=scale_width(16),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # idk if bias should be false
            nn.BatchNorm1d(scale_width(16)),
            nn.ReLU(),
            TCBlock(scale_width(16), scale_width(24)),
            TCBlock(scale_width(24), scale_width(32)),
            TCBlock(scale_width(32), scale_width(48)),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(scale_width(48), num_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1) if x.dim() == 4 else x
        return self.head(self.encoder(x))


ModelName = Literal["cnn", "tc_resnet8"]
ModelCtor = Callable[[int, float], nn.Module]


def tc_resnet8_ctor(num_classes: int, p: float = 0.5):
    return TCResNet8(num_classes, k=2, p=p)


def cnn_ctor(num_classes: int, p: float = 0.5):
    return AudioClassifier(num_classes, p=p)


_MODEL_CTORS: dict[ModelName, ModelCtor] = {
    "cnn": cnn_ctor,
    "tc_resnet8": tc_resnet8_ctor,
}


def model_factory(name: str, num_classes: int, p: float) -> nn.Module:
    ctor = _MODEL_CTORS.get(name)
    if ctor is None:
        raise ValueError(
            f"model name does not exist, got {name}. want {sorted(_MODEL_CTORS)}"
        )
    return ctor(num_classes, p)


if __name__ == "__main__":
    model = TCResNet8(in_channels=80, num_classes=12)
    x = torch.randn(1, 100, 80)  # (batch, time, n_mels)
    print(model(x).shape)  # (1, 12)
