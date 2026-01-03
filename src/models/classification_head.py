"""
Classification head for disease prediction.
Predicts disease class from fused multimodal features.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    MLP-based classification head for disease prediction.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [256],
        num_classes: int = 4,
        activation: str = 'relu',
        use_batch_norm: bool = False,
        dropout: float = 0.3
    ):
        """
        Initialize classification head.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            activation: Activation function ('relu', 'gelu')
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Build classifier layers
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        logger.info(f"ClassificationHead initialized: input_dim={input_dim}, "
                   f"hidden_dims={hidden_dims}, num_classes={num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Class logits (B, num_classes)
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Predicted class indices (B,)
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Class probabilities (B, num_classes)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities


class AttentionClassificationHead(nn.Module):
    """
    Classification head with self-attention mechanism.
    More sophisticated alternative to standard MLP.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize attention-based classification head.

        Args:
            input_dim: Input feature dimension
            num_heads: Number of attention heads
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )

        logger.info(f"AttentionClassificationHead initialized: input_dim={input_dim}, "
                   f"num_heads={num_heads}, num_classes={num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Class logits (B, num_classes)
        """
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (B, 1, input_dim)

        # Self-attention
        attended, _ = self.self_attention(x, x, x)  # (B, 1, input_dim)

        # Residual connection and layer norm
        x = self.layer_norm(x + attended)

        # Remove sequence dimension
        x = x.squeeze(1)  # (B, input_dim)

        # Classify
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    # Test classification head
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing ClassificationHead...")
    print("=" * 80)

    batch_size = 16
    input_features = torch.randn(batch_size, 512)

    # Test standard classification head
    print("\nTest 1: Standard Classification Head")
    print("-" * 80)

    classifier = ClassificationHead(
        input_dim=512,
        hidden_dims=[256],
        num_classes=4,
        dropout=0.3
    )

    logits = classifier(input_features)
    predictions = classifier.predict(input_features)
    probabilities = classifier.predict_proba(input_features)

    print(f"Input shape: {input_features.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample probabilities:\n{probabilities[:3]}")
    print(f"Parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # Test with different architectures
    print("\n\nTest 2: Different architectures")
    print("-" * 80)

    configs = [
        ([256], "Single hidden layer"),
        ([256, 128], "Two hidden layers"),
        ([512, 256, 128], "Three hidden layers"),
    ]

    for hidden_dims, desc in configs:
        clf = ClassificationHead(
            input_dim=512,
            hidden_dims=hidden_dims,
            num_classes=4
        )

        output = clf(input_features)
        params = sum(p.numel() for p in clf.parameters())

        print(f"\n{desc}: {hidden_dims}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {params:,}")

    # Test attention-based classifier
    print("\n\nTest 3: Attention Classification Head")
    print("-" * 80)

    attention_classifier = AttentionClassificationHead(
        input_dim=512,
        num_heads=8,
        num_classes=4
    )

    logits = attention_classifier(input_features)
    print(f"Logits shape: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in attention_classifier.parameters()):,}")

    print("\n" + "=" * 80)
    print("ClassificationHead tests completed!")
