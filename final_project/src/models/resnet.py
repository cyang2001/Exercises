"""ResNet model wrappers for spectrogram classification."""

from typing import Optional

import torch
from torch import nn
from torchvision.models import resnet18, resnet34


class ResNetBuilder:
    """Build ResNet backbones with a task-specific classifier head."""

    def build(
        self,
        model_name: str,
        num_classes: int,
        in_channels: int = 3,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> nn.Module:
        """Build a ResNet model instance.

        Args:
            model_name: ResNet variant name, such as ``resnet18`` or ``resnet34``.
            num_classes: Number of output classes.
            in_channels: Number of input channels for the first convolution.
            pretrained: Whether to request torchvision pretrained weights.
            dropout: Dropout probability before the final classification layer.

        Returns:
            A configured ResNet model.

        Raises:
            ValueError: If the requested model name is unsupported.
        """
        weights = self._get_weights(model_name=model_name, pretrained=pretrained)

        if model_name == "resnet18":
            model = resnet18(weights=weights)
        elif model_name == "resnet34":
            model = resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        self._adapt_input_layer(
            model=model,
            in_channels=in_channels,
            pretrained=pretrained,
        )
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    def _adapt_input_layer(
        self,
        model: nn.Module,
        in_channels: int,
        pretrained: bool,
    ) -> None:
        """Adapt the first convolution when the configured input channels differ.

        Args:
            model: ResNet backbone to update in-place.
            in_channels: Requested number of input channels.
            pretrained: Whether pretrained weights were loaded.
        """
        original_conv = model.conv1
        if in_channels == original_conv.in_channels:
            return

        updated_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            if pretrained and original_conv.weight.shape[1] == 3:
                mean_weight = original_conv.weight.mean(dim=1, keepdim=True)
                updated_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
            else:
                nn.init.kaiming_normal_(
                    updated_conv.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            if original_conv.bias is not None and updated_conv.bias is not None:
                updated_conv.bias.copy_(original_conv.bias)

        model.conv1 = updated_conv

    def _get_weights(self, model_name: str, pretrained: bool) -> Optional[object]:
        """Get the torchvision weights object when pretraining is enabled.

        Args:
            model_name: Requested ResNet variant.
            pretrained: Whether pretrained weights should be used.

        Returns:
            A torchvision weights object or ``None``.
        """
        if not pretrained:
            return None

        from torchvision.models import ResNet18_Weights, ResNet34_Weights

        if model_name == "resnet18":
            return ResNet18_Weights.DEFAULT
        if model_name == "resnet34":
            return ResNet34_Weights.DEFAULT
        return None
