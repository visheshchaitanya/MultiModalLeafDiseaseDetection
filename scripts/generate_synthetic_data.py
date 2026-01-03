"""
Script to generate synthetic sensor data for the DiaMOS dataset.
Creates temperature, humidity, and soil moisture data correlated with disease labels.
"""

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.synthetic_generator import SyntheticSensorGenerator
from src.data.preprocessing import DataPreprocessor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic sensor data for leaf images'
    )

    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/raw/DiaMOS_Plant',
        help='Directory containing disease class subdirectories with images'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default='data/synthetic/sensor_data.csv',
        help='Output path for sensor data CSV'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--add-noise',
        action='store_true',
        default=True,
        help='Add random noise to sensor data (default: True)'
    )

    parser.add_argument(
        '--noise-factor',
        type=float,
        default=0.1,
        help='Noise factor (fraction of std, default: 0.1)'
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info(f"{'Synthetic Sensor Data Generator':^80}")
    logger.info("=" * 80)

    # Check if image directory exists
    image_dir = Path(args.image_dir)

    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        logger.error("Please download the dataset first:")
        logger.error("  python scripts/download_data.py")
        return 1

    # Create image index
    logger.info(f"Scanning images in {image_dir}...")

    preprocessor = DataPreprocessor(data_dir=Path("data"))
    image_df = preprocessor.create_image_index(image_dir)

    if len(image_df) == 0:
        logger.error("No images found!")
        logger.error(f"Please check that {image_dir} contains subdirectories for each disease class")
        logger.error("Expected structure:")
        logger.error(f"  {image_dir}/Healthy/")
        logger.error(f"  {image_dir}/Alternaria/")
        logger.error(f"  {image_dir}/Stemphylium/")
        logger.error(f"  {image_dir}/Marssonina/")
        return 1

    logger.info(f"Found {len(image_df)} images")
    logger.info(f"Class distribution:\n{image_df['disease'].value_counts()}")

    # Generate synthetic sensor data
    logger.info("\nGenerating synthetic sensor data...")

    generator = SyntheticSensorGenerator(seed=args.seed)

    sensor_df = generator.generate_for_labels(
        labels=image_df['disease'].tolist(),
        add_noise=args.add_noise,
        noise_factor=args.noise_factor
    )

    # Combine image and sensor data
    combined_df = pd.concat([image_df, sensor_df[['temperature', 'humidity', 'soil_moisture']]], axis=1)

    # Display statistics
    logger.info("\nSensor data statistics:")
    logger.info("\nOverall:")
    logger.info(combined_df[['temperature', 'humidity', 'soil_moisture']].describe())

    logger.info("\nBy disease class:")
    grouped_stats = combined_df.groupby('disease')[['temperature', 'humidity', 'soil_moisture']].describe()
    logger.info(f"\n{grouped_stats}")

    # Save to file
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(output_file, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("✓ Synthetic sensor data generated successfully!")
    logger.info("=" * 80)
    logger.info(f"Output file: {output_file}")
    logger.info(f"Total samples: {len(combined_df)}")
    logger.info(f"Features: {', '.join(['temperature', 'humidity', 'soil_moisture'])}")

    logger.info("\nNext steps:")
    logger.info("1. Prepare dataset splits: python scripts/prepare_dataset.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
