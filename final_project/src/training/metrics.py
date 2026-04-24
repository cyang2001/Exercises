"""Metric helpers for model evaluation."""

from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score


class ClassificationMetrics:
    """Compute classification metrics from predictions and labels."""

    def compute(self, targets: List[int], predictions: List[int]) -> Dict[str, float]:
        """Compute accuracy and macro F1 metrics.

        Args:
            targets: Ground-truth class indices.
            predictions: Predicted class indices.

        Returns:
            A dictionary containing evaluation metric values.
        """
        return {
            "accuracy": accuracy_score(targets, predictions),
            "macro_f1": f1_score(targets, predictions, average="macro"),
        }
