"""
Simplified inference script for multi-modal leaf disease detection.
Runs predictions on individual samples without complex config files.

Usage:
    python scripts/inference_simple.py --image <path> --sensors "temp,humid,soil"

Example:
    python scripts/inference_simple.py \
        --image data/raw/DiaMOS_Plant/Healthy/u1.jpg \
        --sensors "22.5,65.0,50.0"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import logging

from src.models.multimodal_model import MultiModalLeafDiseaseModel
from src.templates.vocabulary import load_vocabulary
from src.data.transforms import get_validation_transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run inference on a leaf image with sensor data'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to leaf image'
    )
    parser.add_argument(
        '--sensors',
        type=str,
        required=True,
        help='Sensor values as "temperature,humidity,soil_moisture" (e.g., "22.5,65.0,50.0")'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/best_model.pt',
        help='Path to model checkpoint (default: outputs/checkpoints/best_model.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save visualization (default: show only)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )
    return parser.parse_args()


def parse_sensors(sensor_string):
    """Parse sensor values from comma-separated string."""
    try:
        values = [float(x.strip()) for x in sensor_string.split(',')]
        if len(values) != 3:
            raise ValueError("Expected 3 values: temperature,humidity,soil_moisture")
        return values
    except Exception as e:
        raise ValueError(f"Invalid sensor format: {e}")


def normalize_sensors(temperature, humidity, soil_moisture):
    """Normalize sensor values to [0, 1] range."""
    # Normalize based on expected ranges
    temp_norm = temperature / 100.0  # Assuming max temp ~100°C
    humid_norm = humidity / 100.0    # Humidity is already in %
    soil_norm = soil_moisture / 100.0  # Soil moisture is in %
    return np.array([temp_norm, humid_norm, soil_norm], dtype=np.float32)


def visualize_prediction(image_array, predicted_class, confidence,
                        class_probs, class_names, sensor_values,
                        generated_text, save_path=None):
    """Create visualization of prediction results."""
    fig = plt.figure(figsize=(16, 6))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Original image
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(image_array)
    ax1.axis('off')
    ax1.set_title(f'Input Image\nPrediction: {predicted_class}',
                  fontsize=12, fontweight='bold')

    # 2. Class probabilities bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['green' if i == class_probs.argmax() else 'lightblue'
              for i in range(len(class_names))]
    ax2.barh(class_names, class_probs, color=colors)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Class Probabilities', fontweight='bold')
    ax2.set_xlim(0, 1)
    for i, (name, prob) in enumerate(zip(class_names, class_probs)):
        ax2.text(prob + 0.02, i, f'{prob:.2%}', va='center')

    # 3. Sensor readings
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    sensor_text = f"""
SENSOR READINGS
{'='*30}

Temperature:   {sensor_values[0]:.1f}°C
Humidity:      {sensor_values[1]:.1f}%
Soil Moisture: {sensor_values[2]:.1f}%

PREDICTION
{'='*30}

Class:      {predicted_class}
Confidence: {confidence:.2%}
"""
    ax3.text(0.1, 0.5, sensor_text.strip(),
             fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 4. Generated explanation
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.axis('off')
    explanation_text = f"""
GENERATED EXPLANATION
{'='*40}

{generated_text}

{'='*40}
"""
    ax4.text(0.05, 0.95, explanation_text.strip(),
             fontsize=10, family='monospace',
             verticalalignment='top', wrap=True,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Multi-Modal Leaf Disease Detection Results',
                 fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")

    plt.show()


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Multi-Modal Leaf Disease Detection - Simple Inference")
    logger.info("=" * 80)

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load vocabulary
    logger.info("Loading vocabulary...")
    vocab_path = Path('data/processed/vocabulary.pkl')
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
    vocabulary = load_vocabulary(vocab_path)
    vocab_size = len(vocabulary)
    logger.info(f"Vocabulary loaded: {vocab_size} tokens")

    # Class names (must match training order)
    class_names = ['Healthy', 'Alternaria', 'Stemphylium', 'Marssonina']
    num_classes = len(class_names)

    # Create model (same architecture as training)
    logger.info("Creating model...")
    model = MultiModalLeafDiseaseModel(
        vocab_size=vocab_size,
        num_classes=num_classes,
        image_encoder_config={
            'backbone': 'resnet50',
            'pretrained': True,
            'freeze_layers': 7,
            'output_dim': 512
        },
        tabular_encoder_config={
            'input_dim': 3,
            'hidden_dims': [64, 128],
            'output_dim': 128,
            'activation': 'relu',
            'dropout': 0.1
        },
        fusion_config={
            'image_dim': 512,
            'sensor_dim': 128,
            'hidden_dim': 512,
            'output_dim': 512,
            'dropout': 0.1
        },
        decoder_config={
            'num_layers': 4,
            'num_heads': 8,
            'embed_dim': 512,
            'ff_dim': 2048,
            'dropout': 0.1
        },
        classifier_config={
            'input_dim': 512,
            'hidden_dims': [256],
            'dropout': 0.3
        },
        pad_idx=vocabulary.get_word_index('<PAD>')
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from epoch {checkpoint['epoch']}")

    # Load and preprocess image
    logger.info(f"\nLoading image from {args.image}")
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Apply transforms
    transform = get_validation_transforms(image_size=224)
    transformed = transform(image=image_array)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Parse and normalize sensor values
    logger.info(f"Parsing sensor values: {args.sensors}")
    sensor_values = parse_sensors(args.sensors)
    temperature, humidity, soil_moisture = sensor_values

    logger.info(f"  Temperature:   {temperature:.1f}°C")
    logger.info(f"  Humidity:      {humidity:.1f}%")
    logger.info(f"  Soil Moisture: {soil_moisture:.1f}%")

    sensor_normalized = normalize_sensors(temperature, humidity, soil_moisture)
    sensor_tensor = torch.FloatTensor(sensor_normalized).unsqueeze(0).to(device)

    # Run inference
    logger.info("\n" + "=" * 80)
    logger.info("Running inference...")
    logger.info("=" * 80)

    with torch.no_grad():
        # Forward pass (no text targets needed for inference)
        outputs = model(images=image_tensor, sensors=sensor_tensor)

        # Get classification results
        class_logits = outputs['class_logits']
        class_probs = torch.softmax(class_logits, dim=1)[0]
        predicted_idx = torch.argmax(class_probs).item()
        confidence = class_probs[predicted_idx].item()
        predicted_class = class_names[predicted_idx]

        # Generate text explanation
        logger.info("Generating explanation...")
        fused_features = outputs['fused_features'].unsqueeze(1)

        generated_ids = model.text_decoder.generate(
            memory=fused_features,
            start_token_idx=vocabulary.get_word_index('<SOS>'),
            end_token_idx=vocabulary.get_word_index('<EOS>'),
            max_len=100,
            temperature=1.0
        )

        generated_text = vocabulary.decode(
            generated_ids[0].cpu().tolist(),
            skip_special_tokens=True
        )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nPredicted Class: {predicted_class}")
    logger.info(f"Confidence:      {confidence:.2%}")
    logger.info(f"\nClass Probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, class_probs)):
        logger.info(f"  {name:<15} {prob:.4f} ({prob*100:.2f}%)")
    logger.info(f"\nGenerated Explanation:")
    logger.info(f"  {generated_text}")
    logger.info("=" * 80)

    # Visualize results
    logger.info("\nGenerating visualization...")
    visualize_prediction(
        image_array=image_array,
        predicted_class=predicted_class,
        confidence=confidence,
        class_probs=class_probs.cpu().numpy(),
        class_names=class_names,
        sensor_values=sensor_values,
        generated_text=generated_text,
        save_path=args.output
    )

    logger.info("\nInference completed successfully!")


if __name__ == "__main__":
    main()
