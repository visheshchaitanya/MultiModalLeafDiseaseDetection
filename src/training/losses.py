"""
Loss functions for multi-task learning.
Combines classification loss and text generation loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and text generation.
    total_loss = alpha * text_loss + beta * class_loss
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        ignore_index: int = 0,
        use_uncertainty_weighting: bool = False
    ):
        """
        Initialize multi-task loss.

        Args:
            alpha: Weight for text generation loss
            beta: Weight for classification loss
            class_weights: Class weights for classification loss
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in text loss (padding)
            use_uncertainty_weighting: Use learnable uncertainty weighting
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Classification loss
        self.class_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

        # Text generation loss
        self.text_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

        # Learnable uncertainty weights (if enabled)
        if use_uncertainty_weighting:
            self.log_var_class = nn.Parameter(torch.zeros(1))
            self.log_var_text = nn.Parameter(torch.zeros(1))
            logger.info("Using learnable uncertainty weighting for multi-task loss")
        else:
            logger.info(f"Using fixed weights: alpha={alpha}, beta={beta}")

    def forward(
        self,
        class_logits: torch.Tensor,
        class_targets: torch.Tensor,
        text_logits: Optional[torch.Tensor] = None,
        text_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            class_logits: Classification logits (B, num_classes)
            class_targets: Classification targets (B,)
            text_logits: Text generation logits (B, seq_len, vocab_size) - optional
            text_targets: Text generation targets (B, seq_len) - optional

        Returns:
            Dictionary with total loss and component losses
        """
        # Classification loss
        class_loss = self.class_criterion(class_logits, class_targets)

        # Initialize output
        losses = {
            'class_loss': class_loss,
            'total_loss': class_loss
        }

        # Text generation loss (if provided)
        if text_logits is not None and text_targets is not None:
            # Reshape for cross entropy
            batch_size, seq_len, vocab_size = text_logits.shape
            text_logits_flat = text_logits.reshape(-1, vocab_size)
            text_targets_flat = text_targets.reshape(-1)

            text_loss = self.text_criterion(text_logits_flat, text_targets_flat)
            losses['text_loss'] = text_loss

            # Combine losses
            if self.use_uncertainty_weighting:
                # Uncertainty weighting: loss = (1 / (2 * sigma^2)) * loss + log(sigma)
                # where sigma^2 = exp(log_var)
                precision_class = torch.exp(-self.log_var_class)
                precision_text = torch.exp(-self.log_var_text)

                total_loss = (
                    precision_class * class_loss + self.log_var_class +
                    precision_text * text_loss + self.log_var_text
                )

                losses['class_weight'] = precision_class.item()
                losses['text_weight'] = precision_text.item()
            else:
                total_loss = self.beta * class_loss + self.alpha * text_loss

            losses['total_loss'] = total_loss

        return losses


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Class weights (num_classes,)
            gamma: Focusing parameter
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Logits (B, num_classes)
            targets: Targets (B,)

        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_loss_function(
    loss_config: Dict,
    class_weights: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Factory function to create loss function from config.

    Args:
        loss_config: Loss configuration dictionary
        class_weights: Optional class weights

    Returns:
        Loss function module
    """
    loss_type = loss_config.get('type', 'multi_task')

    if loss_type == 'multi_task':
        return MultiTaskLoss(
            alpha=loss_config.get('alpha', 0.7),
            beta=loss_config.get('beta', 0.3),
            class_weights=class_weights,
            label_smoothing=loss_config.get('label_smoothing', 0.0),
            ignore_index=loss_config.get('ignore_index', 0),
            use_uncertainty_weighting=loss_config.get('use_uncertainty_weighting', False)
        )
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=class_weights,
            gamma=loss_config.get('gamma', 2.0)
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Loss Functions...")
    print("=" * 80)

    batch_size = 8
    num_classes = 4
    vocab_size = 1000
    seq_len = 50

    # Test MultiTaskLoss
    print("\nTest 1: MultiTaskLoss with fixed weights")
    print("-" * 80)

    loss_fn = MultiTaskLoss(alpha=0.7, beta=0.3)

    class_logits = torch.randn(batch_size, num_classes)
    class_targets = torch.randint(0, num_classes, (batch_size,))
    text_logits = torch.randn(batch_size, seq_len, vocab_size)
    text_targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    losses = loss_fn(class_logits, class_targets, text_logits, text_targets)

    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Class loss: {losses['class_loss'].item():.4f}")
    print(f"Text loss: {losses['text_loss'].item():.4f}")

    # Test with uncertainty weighting
    print("\n\nTest 2: MultiTaskLoss with uncertainty weighting")
    print("-" * 80)

    loss_fn_uncertain = MultiTaskLoss(use_uncertainty_weighting=True)

    losses = loss_fn_uncertain(class_logits, class_targets, text_logits, text_targets)

    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Class loss: {losses['class_loss'].item():.4f}")
    print(f"Text loss: {losses['text_loss'].item():.4f}")
    print(f"Class weight: {losses['class_weight']:.4f}")
    print(f"Text weight: {losses['text_weight']:.4f}")

    # Test FocalLoss
    print("\n\nTest 3: FocalLoss")
    print("-" * 80)

    focal_loss = FocalLoss(gamma=2.0)

    class_logits = torch.randn(batch_size, num_classes)
    class_targets = torch.randint(0, num_classes, (batch_size,))

    loss = focal_loss(class_logits, class_targets)

    print(f"Focal loss: {loss.item():.4f}")

    # Test with class weights
    class_weights = torch.tensor([1.0, 2.0, 1.5, 1.2])
    focal_loss_weighted = FocalLoss(alpha=class_weights, gamma=2.0)

    loss_weighted = focal_loss_weighted(class_logits, class_targets)

    print(f"Weighted focal loss: {loss_weighted.item():.4f}")

    print("\n" + "=" * 80)
    print("Loss function tests completed!")
