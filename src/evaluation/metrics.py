"""
Evaluation metrics for classification and text generation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sklearn_confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Calculate classification metrics."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize classification metrics.

        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update with new predictions and targets.

        Args:
            predictions: Predicted labels or logits (B,) or (B, num_classes)
            targets: Ground truth labels (B,)
        """
        # Convert logits to predictions if needed
        if predictions.dim() == 2:
            predictions = torch.argmax(predictions, dim=1)

        # Move to CPU and convert to numpy
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        self.predictions.extend(preds_np.tolist())
        self.targets.extend(targets_np.tolist())

    def compute(self, average: str = 'macro') -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            average: Averaging method ('micro', 'macro', 'weighted')

        Returns:
            Dictionary of metrics
        """
        if len(self.predictions) == 0:
            logger.warning("No predictions to compute metrics")
            return {}

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = {
            'accuracy': accuracy_score(targets, preds),
            f'precision_{average}': precision_score(targets, preds, average=average, zero_division=0),
            f'recall_{average}': recall_score(targets, preds, average=average, zero_division=0),
            f'f1_{average}': f1_score(targets, preds, average=average, zero_division=0)
        }

        return metrics

    def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics.

        Returns:
            Dictionary of per-class metrics
        """
        if len(self.predictions) == 0:
            return {}

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        # Per-class precision, recall, F1
        precision = precision_score(targets, preds, average=None, zero_division=0)
        recall = recall_score(targets, preds, average=None, zero_division=0)
        f1 = f1_score(targets, preds, average=None, zero_division=0)

        per_class_metrics = {}

        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1': float(f1[i]) if i < len(f1) else 0.0
            }

        return per_class_metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.

        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        if len(self.predictions) == 0:
            return np.zeros((self.num_classes, self.num_classes))

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        cm = sklearn_confusion_matrix(targets, preds, labels=range(self.num_classes))

        return cm

    def get_classification_report(self) -> str:
        """
        Get detailed classification report.

        Returns:
            Classification report string
        """
        if len(self.predictions) == 0:
            return "No predictions available"

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        report = classification_report(
            targets,
            preds,
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )

        return report


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Convenience function to compute classification metrics.

    Args:
        predictions: Predicted labels or logits
        targets: Ground truth labels
        num_classes: Number of classes
        class_names: Class names (optional)

    Returns:
        Dictionary of metrics
    """
    metrics_calculator = ClassificationMetrics(num_classes, class_names)
    metrics_calculator.update(predictions, targets)
    return metrics_calculator.compute()


if __name__ == "__main__":
    # Test classification metrics
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing ClassificationMetrics...")
    print("=" * 80)

    # Create test data
    num_samples = 100
    num_classes = 4
    class_names = ['Healthy', 'Alternaria', 'Stemphylium', 'Marssonina']

    # Simulate predictions and targets
    torch.manual_seed(42)
    predictions_logits = torch.randn(num_samples, num_classes)
    targets = torch.randint(0, num_classes, (num_samples,))

    # Test metrics calculator
    print("\nTest 1: Basic metrics")
    print("-" * 80)

    metrics_calc = ClassificationMetrics(num_classes, class_names)
    metrics_calc.update(predictions_logits, targets)

    metrics = metrics_calc.compute(average='macro')

    print("Metrics (macro average):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test per-class metrics
    print("\n\nTest 2: Per-class metrics")
    print("-" * 80)

    per_class_metrics = metrics_calc.compute_per_class_metrics()

    for class_name, class_metrics in per_class_metrics.items():
        print(f"\n{class_name}:")
        for metric_name, value in class_metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Test confusion matrix
    print("\n\nTest 3: Confusion matrix")
    print("-" * 80)

    cm = metrics_calc.get_confusion_matrix()
    print("Confusion Matrix:")
    print(cm)

    # Test classification report
    print("\n\nTest 4: Classification report")
    print("-" * 80)

    report = metrics_calc.get_classification_report()
    print(report)

    # Test convenience function
    print("\n\nTest 5: Convenience function")
    print("-" * 80)

    metrics = compute_classification_metrics(
        predictions_logits,
        targets,
        num_classes,
        class_names
    )

    print("Computed metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 80)
    print("ClassificationMetrics tests completed!")
