"""
Integrated multi-modal model for leaf disease detection with text generation.
Combines image encoder, sensor encoder, fusion, decoder, and classification head.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .image_encoder import ImageEncoder
from .tabular_encoder import TabularEncoder
from .fusion_module import create_fusion_module
from .transformer_decoder import TransformerTextDecoder
from .classification_head import ClassificationHead

logger = logging.getLogger(__name__)


class MultiModalLeafDiseaseModel(nn.Module):
    """
    Complete multi-modal model for leaf disease detection and explanation generation.

    Architecture:
        Image → ImageEncoder → \
                                Fusion → Classification Head (disease class)
        Sensors → TabularEncoder → /    \
                                         → Transformer Decoder (text explanation)
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        image_encoder_config: Optional[Dict] = None,
        tabular_encoder_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        pad_idx: int = 0
    ):
        """
        Initialize multi-modal model.

        Args:
            vocab_size: Vocabulary size for text generation
            num_classes: Number of disease classes
            image_encoder_config: Configuration for image encoder
            tabular_encoder_config: Configuration for tabular encoder
            fusion_config: Configuration for fusion module
            decoder_config: Configuration for transformer decoder
            classifier_config: Configuration for classification head
            pad_idx: Padding token index
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.pad_idx = pad_idx

        # Default configurations
        image_config = image_encoder_config or {
            'backbone': 'resnet50',
            'pretrained': True,
            'freeze_layers': 7,
            'output_dim': 512,
            'dropout': 0.1
        }

        tabular_config = tabular_encoder_config or {
            'input_dim': 3,
            'hidden_dims': [64],
            'output_dim': 128,
            'dropout': 0.2
        }

        fusion_conf = fusion_config or {
            'method': 'concatenation',
            'image_dim': 512,
            'sensor_dim': 128,
            'output_dim': 512
        }

        decoder_conf = decoder_config or {
            'embed_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'ff_dim': 2048,
            'dropout': 0.1,
            'max_seq_len': 100
        }

        classifier_conf = classifier_config or {
            'input_dim': 512,
            'hidden_dims': [256],
            'dropout': 0.3
        }

        # Initialize components
        self.image_encoder = ImageEncoder(**image_config)

        self.tabular_encoder = TabularEncoder(**tabular_config)

        self.fusion_module = create_fusion_module(**fusion_conf)

        self.text_decoder = TransformerTextDecoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            **decoder_conf
        )

        self.classification_head = ClassificationHead(
            num_classes=num_classes,
            **classifier_conf
        )

        logger.info("MultiModalLeafDiseaseModel initialized")
        logger.info(f"  Vocabulary size: {vocab_size}")
        logger.info(f"  Number of classes: {num_classes}")
        logger.info(f"  Total parameters: {self.count_parameters():,}")

    def forward(
        self,
        images: torch.Tensor,
        sensors: torch.Tensor,
        text_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images (B, 3, H, W)
            sensors: Sensor data (B, 3)
            text_ids: Target text token IDs (B, seq_len) - optional, for training
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - class_logits: Disease classification logits (B, num_classes)
                - text_logits: Text generation logits (B, seq_len, vocab_size) if text_ids provided
                - image_features: Encoded image features (B, image_dim)
                - sensor_features: Encoded sensor features (B, sensor_dim)
                - fused_features: Fused features (B, fusion_dim)
        """
        # Encode modalities
        image_features = self.image_encoder(images)  # (B, 512)
        sensor_features = self.tabular_encoder(sensors)  # (B, 128)

        # Fuse features
        fused_output = self.fusion_module(image_features, sensor_features)

        # Handle different fusion outputs
        if isinstance(fused_output, tuple):
            fused_features, attention_weights = fused_output
        else:
            fused_features = fused_output
            attention_weights = None

        # Classification
        class_logits = self.classification_head(fused_features)  # (B, num_classes)

        # Prepare output
        outputs = {
            'class_logits': class_logits,
            'image_features': image_features,
            'sensor_features': sensor_features,
            'fused_features': fused_features
        }

        # Text generation (if target text provided)
        if text_ids is not None:
            # Create padding mask
            tgt_key_padding_mask = (text_ids == self.pad_idx)

            # Generate text
            text_logits = self.text_decoder(
                tgt_tokens=text_ids,
                memory=fused_features,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            outputs['text_logits'] = text_logits

        # Return attention if requested
        if return_attention and attention_weights is not None:
            outputs['fusion_attention'] = attention_weights

        return outputs

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        sensors: torch.Tensor,
        start_token_idx: int,
        end_token_idx: int,
        max_text_length: int = 100,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions (class and text explanation).

        Args:
            images: Input images (B, 3, H, W)
            sensors: Sensor data (B, 3)
            start_token_idx: Start-of-sequence token index
            end_token_idx: End-of-sequence token index
            max_text_length: Maximum text generation length
            temperature: Sampling temperature

        Returns:
            Dictionary containing:
                - class_predictions: Predicted classes (B,)
                - class_probabilities: Class probabilities (B, num_classes)
                - generated_text_ids: Generated text token IDs (B, generated_len)
        """
        self.eval()

        # Encode and fuse
        image_features = self.image_encoder(images)
        sensor_features = self.tabular_encoder(sensors)

        fused_output = self.fusion_module(image_features, sensor_features)
        if isinstance(fused_output, tuple):
            fused_features = fused_output[0]
        else:
            fused_features = fused_output

        # Classification
        class_logits = self.classification_head(fused_features)
        class_probs = torch.softmax(class_logits, dim=1)
        class_preds = torch.argmax(class_logits, dim=1)

        # Text generation
        generated_text = self.text_decoder.generate(
            memory=fused_features,
            start_token_idx=start_token_idx,
            end_token_idx=end_token_idx,
            max_len=max_text_length,
            temperature=temperature
        )

        return {
            'class_predictions': class_preds,
            'class_probabilities': class_probs,
            'generated_text_ids': generated_text
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_image_encoder(self):
        """Freeze image encoder for transfer learning."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        logger.info("Image encoder frozen")

    def unfreeze_image_encoder(self):
        """Unfreeze image encoder for fine-tuning."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        logger.info("Image encoder unfrozen")


def create_model_from_config(config: Dict, vocab_size: int) -> MultiModalLeafDiseaseModel:
    """
    Create model from configuration dictionary.

    Args:
        config: Configuration dictionary
        vocab_size: Vocabulary size

    Returns:
        Initialized model
    """
    model_config = config.get('model', {})

    model = MultiModalLeafDiseaseModel(
        vocab_size=vocab_size,
        num_classes=config.get('dataset', {}).get('num_classes', 4),
        image_encoder_config=model_config.get('image_encoder'),
        tabular_encoder_config=model_config.get('tabular_encoder'),
        fusion_config=model_config.get('fusion'),
        decoder_config=model_config.get('transformer_decoder'),
        classifier_config=model_config.get('classification_head'),
        pad_idx=0
    )

    return model


if __name__ == "__main__":
    # Test multimodal model
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing MultiModalLeafDiseaseModel...")
    print("=" * 80)

    # Parameters
    batch_size = 4
    vocab_size = 1000
    num_classes = 4

    # Create model
    model = MultiModalLeafDiseaseModel(
        vocab_size=vocab_size,
        num_classes=num_classes
    )

    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test forward pass
    print("\nTest 1: Training forward pass")
    print("-" * 80)

    images = torch.randn(batch_size, 3, 224, 224)
    sensors = torch.randn(batch_size, 3)
    text_ids = torch.randint(0, vocab_size, (batch_size, 50))

    outputs = model(images, sensors, text_ids)

    print(f"Images shape: {images.shape}")
    print(f"Sensors shape: {sensors.shape}")
    print(f"Text IDs shape: {text_ids.shape}")
    print(f"\nOutputs:")
    print(f"  Class logits: {outputs['class_logits'].shape}")
    print(f"  Text logits: {outputs['text_logits'].shape}")
    print(f"  Fused features: {outputs['fused_features'].shape}")

    # Test prediction
    print("\n\nTest 2: Inference/prediction")
    print("-" * 80)

    predictions = model.predict(
        images=images,
        sensors=sensors,
        start_token_idx=1,
        end_token_idx=2,
        max_text_length=30
    )

    print(f"Class predictions: {predictions['class_predictions']}")
    print(f"Class probabilities shape: {predictions['class_probabilities'].shape}")
    print(f"Generated text shape: {predictions['generated_text_ids'].shape}")
    print(f"Sample generated text: {predictions['generated_text_ids'][0].tolist()}")

    print("\n" + "=" * 80)
    print("MultiModalLeafDiseaseModel tests completed!")
