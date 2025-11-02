"""Model architectures for traffic sign recognition."""

from typing import Tuple
import torch
import torch.nn as nn

from .config import DEVICE, NUM_CLASSES


class CNNModel(nn.Module):
    """Baseline convolutional network for 30x30 RGB signs."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(num_classes: int = NUM_CLASSES, device: torch.device = DEVICE) -> CNNModel:
    model = CNNModel(num_classes=num_classes)
    return model.to(device)
