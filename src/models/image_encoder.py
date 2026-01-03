"""
Image encoder for extracting visual features from leaf images.
Uses pretrained CNN (ResNet, EfficientNet) with optional fine-tuning.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """
    Image encoder using pretrained CNN backbone.
    Extracts fixed-size feature vectors from images.
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        freeze_layers: int = 7,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize image encoder.

        Args:
            backbone: Backbone architecture ('resnet50', 'resnet34', 'efficientnet_b0', etc.)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_layers: Number of initial layers to freeze (0 = no freezing)
            output_dim: Output feature dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self.output_dim = output_dim

        # Load backbone
        self.backbone, self.feature_dim = self._load_backbone(backbone, pretrained)

        # Freeze layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Projection head to output_dim
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        logger.info(f"ImageEncoder initialized: {backbone}, pretrained={pretrained}, "
                   f"freeze_layers={freeze_layers}, output_dim={output_dim}")

    def _load_backbone(self, backbone: str, pretrained: bool) -> tuple:
        """
        Load CNN backbone and return model + feature dimension.

        Args:
            backbone: Backbone name
            pretrained: Whether to load pretrained weights

        Returns:
            Tuple of (model, feature_dim)
        """
        weights = 'IMAGENET1K_V1' if pretrained else None

        if backbone == 'resnet50':
            model = models.resnet50(weights=weights)
            feature_dim = 2048
            # Remove final FC layer
            model = nn.Sequential(*list(model.children())[:-2])

        elif backbone == 'resnet34':
            model = models.resnet34(weights=weights)
            feature_dim = 512
            model = nn.Sequential(*list(model.children())[:-2])

        elif backbone == 'resnet18':
            model = models.resnet18(weights=weights)
            feature_dim = 512
            model = nn.Sequential(*list(model.children())[:-2])

        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
            # Remove classifier
            model.classifier = nn.Identity()

        elif backbone == 'efficientnet_b1':
            model = models.efficientnet_b1(weights=weights)
            feature_dim = 1280
            model.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        return model, feature_dim

    def _freeze_layers(self, num_layers: int):
        """
        Freeze the first num_layers of the backbone.

        Args:
            num_layers: Number of layers to freeze
        """
        layers_frozen = 0

        for name, param in self.backbone.named_parameters():
            if layers_frozen < num_layers:
                param.requires_grad = False
                layers_frozen += 1

        logger.info(f"Froze {layers_frozen} layers in {self.backbone_name}")

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

        logger.info(f"All layers unfrozen in {self.backbone_name}")

    def unfreeze_layers(self, num_layers: int):
        """
        Unfreeze the last num_layers for gradual unfreezing.

        Args:
            num_layers: Number of layers to unfreeze from the end
        """
        all_params = list(self.backbone.parameters())
        total_params = len(all_params)

        for i in range(max(0, total_params - num_layers), total_params):
            all_params[i].requires_grad = True

        logger.info(f"Unfroze last {num_layers} layers in {self.backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Feature vectors (B, output_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (B, feature_dim, H', W')

        # Project to output dimension
        output = self.projection(features)  # (B, output_dim)

        return output

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test image encoder
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing ImageEncoder...")
    print("=" * 80)

    # Test different backbones
    backbones = ['resnet50', 'resnet34', 'efficientnet_b0']

    for backbone in backbones:
        print(f"\nTesting {backbone}:")
        print("-" * 80)

        encoder = ImageEncoder(
            backbone=backbone,
            pretrained=True,
            freeze_layers=7,
            output_dim=512
        )

        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        output = encoder(dummy_input)

        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Total parameters: {encoder.get_total_params():,}")
        print(f"Trainable parameters: {encoder.get_trainable_params():,}")

    # Test freezing/unfreezing
    print("\n\nTesting freeze/unfreeze:")
    print("-" * 80)

    encoder = ImageEncoder(backbone='resnet50', freeze_layers=10)
    print(f"After initialization - Trainable: {encoder.get_trainable_params():,}")

    encoder.unfreeze_all()
    print(f"After unfreeze_all - Trainable: {encoder.get_trainable_params():,}")

    print("\n" + "=" * 80)
    print("ImageEncoder tests completed!")
