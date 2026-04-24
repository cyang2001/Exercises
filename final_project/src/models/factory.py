"""Model factory for spectrogram classifiers."""

import logging
from typing import Any

from torch import nn

from src.models.cnn import SimpleCNNClassifier
from src.models.resnet import ResNetBuilder
from src.utils.logger import get_logger


class ModelFactory:
    """Create models from configuration values."""

    def __init__(self, cfg: Any, logger: logging.Logger = None) -> None:
        """Initialize the model factory.

        Args:
            cfg: Project configuration object.
            logger: Optional logger instance.
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)

    def build_model(self) -> nn.Module:
        """Build the configured model instance.

        Returns:
            A PyTorch model ready for training or inference.

        Raises:
            ValueError: If the model name is unsupported.
        """
        model_name = self.cfg.model.name
        in_channels = self._resolve_in_channels()
        self.logger.info("Building model: %s", model_name)

        if model_name == "cnn":
            return SimpleCNNClassifier(
                in_channels=in_channels,
                num_classes=self.cfg.model.num_classes,
                dropout=self.cfg.model.dropout,
            )

        if model_name in {"resnet18", "resnet34"}:
            return ResNetBuilder().build(
                model_name=model_name,
                num_classes=self.cfg.model.num_classes,
                in_channels=in_channels,
                pretrained=self.cfg.model.pretrained,
                dropout=self.cfg.model.dropout,
            )

        raise ValueError(f"Unsupported model name: {model_name}")

    def _resolve_in_channels(self) -> int:
        """Resolve the model input channel count from the merged configuration.

        Returns:
            The number of input channels expected by the model.
        """
        if self.cfg.get("data") is not None and self.cfg.data.get("num_channels") is not None:
            return int(self.cfg.data.num_channels)
        return int(self.cfg.model.in_channels)
