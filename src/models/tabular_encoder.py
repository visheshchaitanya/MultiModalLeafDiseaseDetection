"""
Tabular encoder for processing environmental sensor data.
Uses MLP with batch normalization and dropout.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TabularEncoder(nn.Module):
    """
    MLP-based encoder for tabular sensor data.
    Processes temperature, humidity, and soil moisture values.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] = [64],
        output_dim: int = 128,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout: float = 0.2
    ):
        """
        Initialize tabular encoder.

        Args:
            input_dim: Input feature dimension (default: 3 for temp, humidity, moisture)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
            activation: Activation function ('relu', 'leaky_relu', 'gelu')
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        # Build MLP layers
        layers = []
        current_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self._get_activation(activation))

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        # Final batch norm
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))

        # Final activation
        layers.append(self._get_activation(activation))

        self.mlp = nn.Sequential(*layers)

        logger.info(f"TabularEncoder initialized: input_dim={input_dim}, "
                   f"hidden_dims={hidden_dims}, output_dim={output_dim}")

    def _get_activation(self, activation: str) -> nn.Module:
        """
        Get activation function.

        Args:
            activation: Activation name

        Returns:
            Activation module
        """
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sensor data (B, input_dim)

        Returns:
            Encoded features (B, output_dim)
        """
        return self.mlp(x)

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TabTransformerEncoder(nn.Module):
    """
    Alternative: Transformer-based encoder for tabular data.
    Uses self-attention to capture feature interactions.
    """

    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize TabTransformer encoder.

        Args:
            input_dim: Number of input features
            embed_dim: Embedding dimension for each feature
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            output_dim: Output feature dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Feature embeddings
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(embed_dim * input_dim, output_dim)

        logger.info(f"TabTransformerEncoder initialized: input_dim={input_dim}, "
                   f"embed_dim={embed_dim}, num_layers={num_layers}, output_dim={output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sensor data (B, input_dim)

        Returns:
            Encoded features (B, output_dim)
        """
        batch_size = x.size(0)

        # Embed each feature separately
        embedded_features = []
        for i in range(self.input_dim):
            feature = x[:, i:i+1]  # (B, 1)
            embedded = self.feature_embeddings[i](feature)  # (B, embed_dim)
            embedded_features.append(embedded)

        # Stack embeddings (B, input_dim, embed_dim)
        x_embedded = torch.stack(embedded_features, dim=1)

        # Apply transformer
        transformed = self.transformer(x_embedded)  # (B, input_dim, embed_dim)

        # Flatten and project
        flattened = transformed.view(batch_size, -1)  # (B, input_dim * embed_dim)
        output = self.output_proj(flattened)  # (B, output_dim)

        return output


if __name__ == "__main__":
    # Test tabular encoder
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing TabularEncoder...")
    print("=" * 80)

    # Test MLP encoder
    print("\nTest 1: MLP Encoder")
    print("-" * 80)

    encoder = TabularEncoder(
        input_dim=3,
        hidden_dims=[64],
        output_dim=128,
        use_batch_norm=True,
        dropout=0.2
    )

    # Create dummy input (batch of sensor data)
    batch_size = 16
    dummy_input = torch.randn(batch_size, 3)  # 3 features: temp, humidity, moisture

    output = encoder(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {encoder.get_trainable_params():,}")

    # Test with different configurations
    print("\n\nTest 2: Different hidden layer configurations")
    print("-" * 80)

    configs = [
        ([64], "Single hidden layer"),
        ([64, 128], "Two hidden layers"),
        ([128, 256, 128], "Three hidden layers")
    ]

    for hidden_dims, description in configs:
        encoder = TabularEncoder(
            input_dim=3,
            hidden_dims=hidden_dims,
            output_dim=128
        )

        output = encoder(dummy_input)
        params = encoder.get_trainable_params()

        print(f"\n{description}: {hidden_dims}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {params:,}")

    # Test TabTransformer
    print("\n\nTest 3: TabTransformer Encoder")
    print("-" * 80)

    transformer_encoder = TabTransformerEncoder(
        input_dim=3,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        output_dim=128
    )

    output = transformer_encoder(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {sum(p.numel() for p in transformer_encoder.parameters()):,}")

    print("\n" + "=" * 80)
    print("TabularEncoder tests completed!")
