"""CNN models for spectrogram classification."""

import torch
from torch import nn


class SimpleCNNClassifier(nn.Module):
    """A compact CNN baseline for spectrogram classification."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the CNN classifier.

        Args:
            in_channels: Number of input image channels.
            num_classes: Number of output classes.
            dropout: Dropout probability for the classifier head.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            inputs: Input tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Classification logits.
        """
        features = self.features(inputs)
        return self.classifier(features)
