"""
Fusion module for combining image and sensor features.
Supports concatenation, cross-attention, and gated fusion strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation-based fusion with MLP projection.
    Concatenates image and sensor features, then projects to output dimension.
    """

    def __init__(
        self,
        image_dim: int = 512,
        sensor_dim: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 512,
        activation: str = 'relu',
        use_layer_norm: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize concatenation fusion.

        Args:
            image_dim: Image feature dimension
            sensor_dim: Sensor feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout rate
        """
        super().__init__()

        self.image_dim = image_dim
        self.sensor_dim = sensor_dim
        self.output_dim = output_dim

        concat_dim = image_dim + sensor_dim

        # Projection layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        )

        logger.info(f"ConcatenationFusion initialized: image_dim={image_dim}, "
                   f"sensor_dim={sensor_dim}, output_dim={output_dim}")

    def forward(
        self,
        image_features: torch.Tensor,
        sensor_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image_features: Image features (B, image_dim)
            sensor_features: Sensor features (B, sensor_dim)

        Returns:
            Fused features (B, output_dim)
        """
        # Concatenate
        concatenated = torch.cat([image_features, sensor_features], dim=1)

        # Project
        fused = self.fusion_layers(concatenated)

        return fused


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion.
    Uses attention mechanism to allow modalities to interact.
    """

    def __init__(
        self,
        image_dim: int = 512,
        sensor_dim: int = 128,
        num_heads: int = 8,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention fusion.

        Args:
            image_dim: Image feature dimension
            sensor_dim: Sensor feature dimension
            num_heads: Number of attention heads
            output_dim: Output feature dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.image_dim = image_dim
        self.sensor_dim = sensor_dim
        self.output_dim = output_dim

        # Project sensor features to match image dimension
        self.sensor_proj = nn.Linear(sensor_dim, image_dim)

        # Cross-attention: sensor attends to image
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm
        self.layer_norm1 = nn.LayerNorm(image_dim)
        self.layer_norm2 = nn.LayerNorm(image_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(image_dim, image_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(image_dim * 2, output_dim)
        )

        logger.info(f"CrossAttentionFusion initialized: image_dim={image_dim}, "
                   f"sensor_dim={sensor_dim}, output_dim={output_dim}")

    def forward(
        self,
        image_features: torch.Tensor,
        sensor_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            image_features: Image features (B, image_dim)
            sensor_features: Sensor features (B, sensor_dim)

        Returns:
            Tuple of (fused_features, attention_weights)
        """
        batch_size = image_features.size(0)

        # Project sensor features
        sensor_proj = self.sensor_proj(sensor_features)  # (B, image_dim)

        # Reshape for attention (add sequence dimension)
        image_seq = image_features.unsqueeze(1)  # (B, 1, image_dim)
        sensor_seq = sensor_proj.unsqueeze(1)  # (B, 1, image_dim)

        # Cross-attention: sensor (query) attends to image (key, value)
        attended, attention_weights = self.cross_attention(
            query=sensor_seq,
            key=image_seq,
            value=image_seq
        )  # attended: (B, 1, image_dim)

        # Residual connection and layer norm
        sensor_attended = self.layer_norm1(sensor_seq + attended)

        # Combine with image features
        combined = (image_seq + sensor_attended) / 2  # Simple averaging

        # Layer norm
        combined = self.layer_norm2(combined)

        # Feed-forward
        fused = self.ffn(combined.squeeze(1))  # (B, output_dim)

        return fused, attention_weights


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism.
    Learns to weight the importance of each modality dynamically.
    """

    def __init__(
        self,
        image_dim: int = 512,
        sensor_dim: int = 128,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize gated fusion.

        Args:
            image_dim: Image feature dimension
            sensor_dim: Sensor feature dimension
            output_dim: Output feature dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.image_dim = image_dim
        self.sensor_dim = sensor_dim
        self.output_dim = output_dim

        # Project features to same dimension
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.sensor_proj = nn.Linear(sensor_dim, output_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(image_dim + sensor_dim, output_dim),
            nn.Sigmoid()
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

        logger.info(f"GatedFusion initialized: image_dim={image_dim}, "
                   f"sensor_dim={sensor_dim}, output_dim={output_dim}")

    def forward(
        self,
        image_features: torch.Tensor,
        sensor_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image_features: Image features (B, image_dim)
            sensor_features: Sensor features (B, sensor_dim)

        Returns:
            Fused features (B, output_dim)
        """
        # Project to same dimension
        image_proj = self.image_proj(image_features)  # (B, output_dim)
        sensor_proj = self.sensor_proj(sensor_features)  # (B, output_dim)

        # Compute gate
        concat = torch.cat([image_features, sensor_features], dim=1)
        gate = self.gate(concat)  # (B, output_dim)

        # Weighted fusion
        fused = gate * image_proj + (1 - gate) * sensor_proj

        # Final projection
        output = self.output_proj(fused)

        return output


def create_fusion_module(
    method: str = 'concatenation',
    **kwargs
) -> nn.Module:
    """
    Factory function to create fusion module.

    Args:
        method: Fusion method ('concatenation', 'cross_attention', 'gated')
        **kwargs: Arguments for the fusion module

    Returns:
        Fusion module
    """
    if method == 'concatenation':
        return ConcatenationFusion(**kwargs)
    elif method == 'cross_attention':
        return CrossAttentionFusion(**kwargs)
    elif method == 'gated':
        return GatedFusion(**kwargs)
    else:
        raise ValueError(f"Unsupported fusion method: {method}")


if __name__ == "__main__":
    # Test fusion modules
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Fusion Modules...")
    print("=" * 80)

    batch_size = 8
    image_features = torch.randn(batch_size, 512)
    sensor_features = torch.randn(batch_size, 128)

    # Test Concatenation Fusion
    print("\nTest 1: Concatenation Fusion")
    print("-" * 80)

    fusion1 = ConcatenationFusion(
        image_dim=512,
        sensor_dim=128,
        hidden_dim=512,
        output_dim=512
    )

    output1 = fusion1(image_features, sensor_features)
    print(f"Image features: {image_features.shape}")
    print(f"Sensor features: {sensor_features.shape}")
    print(f"Fused output: {output1.shape}")
    print(f"Parameters: {sum(p.numel() for p in fusion1.parameters()):,}")

    # Test Cross-Attention Fusion
    print("\n\nTest 2: Cross-Attention Fusion")
    print("-" * 80)

    fusion2 = CrossAttentionFusion(
        image_dim=512,
        sensor_dim=128,
        num_heads=8,
        output_dim=512
    )

    output2, attention = fusion2(image_features, sensor_features)
    print(f"Fused output: {output2.shape}")
    print(f"Attention weights: {attention.shape}")
    print(f"Parameters: {sum(p.numel() for p in fusion2.parameters()):,}")

    # Test Gated Fusion
    print("\n\nTest 3: Gated Fusion")
    print("-" * 80)

    fusion3 = GatedFusion(
        image_dim=512,
        sensor_dim=128,
        output_dim=512
    )

    output3 = fusion3(image_features, sensor_features)
    print(f"Fused output: {output3.shape}")
    print(f"Parameters: {sum(p.numel() for p in fusion3.parameters()):,}")

    # Test factory function
    print("\n\nTest 4: Factory function")
    print("-" * 80)

    for method in ['concatenation', 'cross_attention', 'gated']:
        fusion = create_fusion_module(
            method=method,
            image_dim=512,
            sensor_dim=128,
            output_dim=512
        )
        output = fusion(image_features, sensor_features)
        if isinstance(output, tuple):
            output = output[0]
        print(f"{method}: output shape = {output.shape}")

    print("\n" + "=" * 80)
    print("Fusion module tests completed!")
