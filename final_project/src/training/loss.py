"""Loss builders for model training."""

from typing import Optional

import torch
from torch import nn


class LossFactory:
    """Build training losses for the classification task."""

    def build_cross_entropy(
        self,
        class_weights: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Build a cross-entropy loss function.

        Args:
            class_weights: Optional class weights tensor on the target device.

        Returns:
            A configured cross-entropy loss instance.
        """
        return nn.CrossEntropyLoss(weight=class_weights)
