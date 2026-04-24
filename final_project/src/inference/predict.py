"""Inference helpers for trained classifiers."""

import logging
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from src.utils.logger import get_logger


class Predictor:
    """Run batched prediction with a trained model."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            model: Trained model used for inference.
            device: Torch device for inference.
            logger: Optional logger instance.
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logger or get_logger(__name__)

    def predict(self, data_loader: DataLoader) -> List[int]:
        """Predict class indices for a data loader.

        Args:
            data_loader: DataLoader with unlabeled or labeled samples.

        Returns:
            Predicted class indices in loader order.
        """
        self.model.eval()
        predictions: List[int] = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device)
                logits = self.model(images)
                batch_predictions = torch.argmax(logits, dim=1)
                predictions.extend(batch_predictions.detach().cpu().tolist())

        self.logger.info("Generated %s predictions.", len(predictions))
        return predictions
