"""
Inference script for multi-modal leaf disease detection model.
Run predictions on individual samples.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging

from src.utils.config_loader import ConfigLoader
from src.utils.logging_utils import setup_logging
from src.utils.device import get_device
from src.models.multimodal_model import MultiModalLeafDiseaseModel
from src.templates.vocabulary import Vocabulary
from src.data.transforms import get_validation_transforms, SensorNormalizer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference on leaf disease images'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )

    parser.add_argument(
        '--sensors',
        type=str,
        required=True,
        help='Sensor readings as comma-separated values: "temperature,humidity,soil_moisture" (e.g., "22.5,80.0,45.0")'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save visualization (optional)'
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip visualization'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for text generation sampling'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling for text generation'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    return parser.parse_args()


def load_and_preprocess_image(image_path: str, transform):
    """
    Load and preprocess image.

    Args:
        image_path: Path to image
        transform: Image transform

    Returns:
        Preprocessed image tensor and original image
    """
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)

    # Apply transform
    transformed = transform(image=original_image)
    image_tensor = transformed['image']

    return image_tensor.unsqueeze(0), original_image


def parse_sensor_values(sensor_string: str):
    """
    Parse sensor values from string.

    Args:
        sensor_string: Comma-separated sensor values

    Returns:
        Tuple of (temperature, humidity, soil_moisture)
    """
    try:
        values = [float(x.strip()) for x in sensor_string.split(',')]
        if len(values) != 3:
            raise ValueError("Expected 3 values (temperature, humidity, soil_moisture)")
        return tuple(values)
    except Exception as e:
        raise ValueError(f"Invalid sensor values: {e}")


def visualize_prediction(
    image: np.ndarray,
    predicted_class: str,
    confidence: float,
    generated_text: str,
    sensor_values: tuple,
    save_path: str = None
):
    """
    Visualize prediction results.

    Args:
        image: Original image
        predicted_class: Predicted class name
        confidence: Prediction confidence
        generated_text: Generated explanation text
        sensor_values: (temperature, humidity, soil_moisture)
        save_path: Path to save visualization (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Display image
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}',
                      fontsize=14, fontweight='bold')

    # Display text information
    axes[1].axis('off')

    info_text = f"""
PREDICTION RESULTS
{'=' * 40}

Disease Class: {predicted_class}
Confidence: {confidence:.2%}

Sensor Readings:
  Temperature: {sensor_values[0]:.1f}°C
  Humidity: {sensor_values[1]:.1f}%
  Soil Moisture: {sensor_values[2]:.1f}%

{'=' * 40}

EXPLANATION:

{generated_text}

{'=' * 40}
    """

    axes[1].text(
        0.05, 0.95,
        info_text.strip(),
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")

    plt.show()


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 80)
    logger.info("Multi-Modal Leaf Disease Detection - Inference")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config_loader = ConfigLoader(args.config)
    config = config_loader.load()

    # Setup device
    if args.device is not None:
        use_cuda = (args.device == 'cuda')
    else:
        use_cuda = config.get('device', {}).get('use_cuda', True)

    device = get_device(use_cuda=use_cuda)
    logger.info(f"Using device: {device}")

    # Load vocabulary
    logger.info("Loading vocabulary...")
    vocab_path = Path(config['paths']['processed_data']) / 'vocabulary.json'
    vocabulary = Vocabulary.load_from_json(vocab_path)
    logger.info(f"Vocabulary loaded: {len(vocabulary)} tokens")

    # Load class names
    class_names = config['data'].get('class_names', [
        'Healthy', 'Alternaria', 'Stemphylium', 'Marssonina'
    ])

    # Create model
    logger.info("Creating model...")
    model = MultiModalLeafDiseaseModel(
        vocab_size=len(vocabulary),
        num_classes=len(class_names),
        image_config=config.get('model', {}).get('image_encoder', {}),
        tabular_config=config.get('model', {}).get('tabular_encoder', {}),
        fusion_config=config.get('model', {}).get('fusion', {}),
        decoder_config=config.get('model', {}).get('decoder', {}),
        classifier_config=config.get('model', {}).get('classifier', {})
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    # Load and preprocess image
    logger.info(f"Loading image from {args.image}")
    image_size = config.get('data', {}).get('image_size', 224)
    transform = get_validation_transforms(image_size=image_size)

    image_tensor, original_image = load_and_preprocess_image(args.image, transform)
    image_tensor = image_tensor.to(device)

    # Parse sensor values
    logger.info(f"Parsing sensor values: {args.sensors}")
    temperature, humidity, soil_moisture = parse_sensor_values(args.sensors)

    logger.info(f"  Temperature: {temperature:.1f}°C")
    logger.info(f"  Humidity: {humidity:.1f}%")
    logger.info(f"  Soil Moisture: {soil_moisture:.1f}%")

    # Normalize sensor values
    sensor_normalizer = SensorNormalizer()
    sensor_tensor = sensor_normalizer.normalize(temperature, humidity, soil_moisture)
    sensor_tensor = torch.FloatTensor(sensor_tensor).unsqueeze(0).to(device)

    # Run inference
    logger.info("\nRunning inference...")

    with torch.no_grad():
        # Forward pass
        outputs = model(
            images=image_tensor,
            sensors=sensor_tensor
        )

        # Classification
        class_logits = outputs['class_logits']
        class_probs = torch.softmax(class_logits, dim=1)
        predicted_class_idx = torch.argmax(class_probs, dim=1).item()
        confidence = class_probs[0, predicted_class_idx].item()

        predicted_class = class_names[predicted_class_idx]

        # Text generation
        logger.info("Generating explanation...")

        # Get fused features for text generation
        fused_features = outputs['fused_features'].unsqueeze(1)  # (B, 1, D)

        # Generate text
        generated_ids = model.text_decoder.generate(
            memory=fused_features,
            start_token_idx=vocabulary.get_token_index(vocabulary.SOS_TOKEN),
            end_token_idx=vocabulary.get_token_index(vocabulary.EOS_TOKEN),
            max_len=100,
            temperature=args.temperature,
            top_k=args.top_k
        )

        # Decode generated text
        generated_text = vocabulary.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=True)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nDisease Class: {predicted_class}")
    logger.info(f"Confidence: {confidence:.2%}")
    logger.info(f"\nSensor Readings:")
    logger.info(f"  Temperature: {temperature:.1f}°C")
    logger.info(f"  Humidity: {humidity:.1f}%")
    logger.info(f"  Soil Moisture: {soil_moisture:.1f}%")
    logger.info(f"\nGenerated Explanation:")
    logger.info(f"{generated_text}")
    logger.info("=" * 80)

    # Visualize results
    if not args.no_plot:
        logger.info("\nGenerating visualization...")
        visualize_prediction(
            image=original_image,
            predicted_class=predicted_class,
            confidence=confidence,
            generated_text=generated_text,
            sensor_values=(temperature, humidity, soil_moisture),
            save_path=args.output
        )


if __name__ == "__main__":
    main()
