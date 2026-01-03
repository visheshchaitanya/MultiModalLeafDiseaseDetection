"""
End-to-End Validation Pipeline

This script demonstrates the complete workflow for validating the model
on new, unseen data. It simulates a production scenario where you receive
new leaf images with sensor data and need to get predictions.

Steps:
1. Load the trained model
2. Process new images and sensor data
3. Run inference
4. Generate predictions and explanations
5. Save results

Usage:
    python scripts/validate_pipeline.py --data-dir <path_to_new_images>

Example:
    python scripts/validate_pipeline.py --data-dir data/raw/DiaMOS_Plant/Healthy --num-samples 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

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
        description='End-to-end validation pipeline for new data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing new images to validate'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to process (default: 10)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/validation',
        help='Directory to save validation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use'
    )
    parser.add_argument(
        '--generate-sensors',
        action='store_true',
        help='Generate synthetic sensor data (use if you dont have real sensors)'
    )
    return parser.parse_args()


def normalize_sensors(temperature, humidity, soil_moisture):
    """Normalize sensor values."""
    temp_norm = temperature / 100.0
    humid_norm = humidity / 100.0
    soil_norm = soil_moisture / 100.0
    return np.array([temp_norm, humid_norm, soil_norm], dtype=np.float32)


def generate_random_sensors():
    """Generate random but realistic sensor values."""
    temperature = np.random.uniform(15.0, 30.0)
    humidity = np.random.uniform(60.0, 95.0)
    soil_moisture = np.random.uniform(35.0, 65.0)
    return temperature, humidity, soil_moisture


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    logger.info("Loading vocabulary...")
    vocab_path = Path('data/processed/vocabulary.pkl')
    vocabulary = load_vocabulary(vocab_path)
    vocab_size = len(vocabulary)

    class_names = ['Healthy', 'Alternaria', 'Stemphylium', 'Marssonina']
    num_classes = len(class_names)

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

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, vocabulary, class_names


def process_single_image(image_path, sensor_values, model, vocabulary,
                         class_names, transform, device):
    """Process a single image and return predictions."""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    transformed = transform(image=image_array)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Normalize sensors
    temperature, humidity, soil_moisture = sensor_values
    sensor_normalized = normalize_sensors(temperature, humidity, soil_moisture)
    sensor_tensor = torch.FloatTensor(sensor_normalized).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(images=image_tensor, sensors=sensor_tensor)

        # Classification
        class_logits = outputs['class_logits']
        class_probs = torch.softmax(class_logits, dim=1)[0]
        predicted_idx = torch.argmax(class_probs).item()
        confidence = class_probs[predicted_idx].item()
        predicted_class = class_names[predicted_idx]

        # Text generation
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

    return {
        'image_path': str(image_path),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'class_probabilities': {name: prob for name, prob in zip(class_names, class_probs.cpu().numpy())},
        'temperature': temperature,
        'humidity': humidity,
        'soil_moisture': soil_moisture,
        'explanation': generated_text,
        'image_array': image_array
    }


def visualize_results(results, output_dir):
    """Create summary visualization of all results."""
    num_samples = len(results)
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, result in enumerate(results):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        ax.imshow(result['image_array'])
        ax.axis('off')

        title = f"{result['predicted_class']}\n"
        title += f"Conf: {result['confidence']:.2%}"
        ax.set_title(title, fontsize=10, fontweight='bold')

    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    save_path = output_dir / 'validation_summary.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Summary visualization saved to {save_path}")


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("End-to-End Validation Pipeline")
    logger.info("=" * 80)

    # Setup
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, vocabulary, class_names = load_model(args.checkpoint, device)
    transform = get_validation_transforms(image_size=224)

    # Get image files
    data_dir = Path(args.data_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in data_dir.glob('*') if f.suffix.lower() in image_extensions]

    if not image_files:
        raise ValueError(f"No image files found in {data_dir}")

    # Limit number of samples
    image_files = image_files[:args.num_samples]
    logger.info(f"\nFound {len(image_files)} images to process")

    # Process each image
    logger.info("\n" + "=" * 80)
    logger.info("Processing Images")
    logger.info("=" * 80)

    results = []
    for image_path in tqdm(image_files, desc="Processing"):
        try:
            # Generate or load sensor data
            if args.generate_sensors:
                sensor_values = generate_random_sensors()
            else:
                # In real scenario, load actual sensor data here
                sensor_values = generate_random_sensors()
                logger.warning(f"No sensor data provided, using synthetic values")

            # Process image
            result = process_single_image(
                image_path=image_path,
                sensor_values=sensor_values,
                model=model,
                vocabulary=vocabulary,
                class_names=class_names,
                transform=transform,
                device=device
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            continue

    # Save results to CSV
    logger.info("\n" + "=" * 80)
    logger.info("Saving Results")
    logger.info("=" * 80)

    results_df = pd.DataFrame([
        {
            'image_path': r['image_path'],
            'predicted_class': r['predicted_class'],
            'confidence': r['confidence'],
            'temperature': r['temperature'],
            'humidity': r['humidity'],
            'soil_moisture': r['soil_moisture'],
            'explanation': r['explanation'],
            **{f'prob_{cls}': r['class_probabilities'][cls] for cls in class_names}
        }
        for r in results
    ])

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'validation_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nTotal samples processed: {len(results)}")
    logger.info(f"\nPrediction distribution:")
    for class_name in class_names:
        count = sum(1 for r in results if r['predicted_class'] == class_name)
        percentage = 100 * count / len(results) if results else 0
        logger.info(f"  {class_name:<15} {count:>3} ({percentage:.1f}%)")

    logger.info(f"\nAverage confidence: {results_df['confidence'].mean():.2%}")
    logger.info(f"Min confidence:     {results_df['confidence'].min():.2%}")
    logger.info(f"Max confidence:     {results_df['confidence'].max():.2%}")

    # Create visualization
    logger.info("\nGenerating visualization...")
    visualize_results(results, output_dir)

    # Save detailed report
    report_path = output_dir / f'validation_report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("END-TO-END VALIDATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model checkpoint: {args.checkpoint}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Total samples: {len(results)}\n\n")

        f.write("PREDICTION DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for class_name in class_names:
            count = sum(1 for r in results if r['predicted_class'] == class_name)
            percentage = 100 * count / len(results) if results else 0
            f.write(f"{class_name:<15} {count:>3} ({percentage:.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for idx, result in enumerate(results, 1):
            f.write(f"Sample {idx}:\n")
            f.write(f"  Image: {Path(result['image_path']).name}\n")
            f.write(f"  Prediction: {result['predicted_class']} ({result['confidence']:.2%})\n")
            f.write(f"  Sensors: T={result['temperature']:.1f}°C, "
                   f"H={result['humidity']:.1f}%, SM={result['soil_moisture']:.1f}%\n")
            f.write(f"  Explanation: {result['explanation']}\n\n")

    logger.info(f"Detailed report saved to {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Validation pipeline completed successfully!")
    logger.info(f"All results saved to {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
