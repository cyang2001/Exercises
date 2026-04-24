"""Scheduler helpers for model optimization."""

from typing import Any, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SchedulerFactory:
    """Create learning-rate schedulers for training."""

    def build(self, optimizer: Optimizer, cfg: Any) -> Optional[ReduceLROnPlateau]:
        """Build the scheduler used by the baseline training loop.

        Args:
            optimizer: Optimizer instance attached to the model.
            cfg: Project configuration object.

        Returns:
            A scheduler instance or ``None`` when no scheduler is configured.
        """
        _ = cfg
        return ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
