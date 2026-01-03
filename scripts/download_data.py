"""
Script to download the DiaMOS Plant dataset.
"""

import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.download import DiaMOSDownloader, DIAMOS_DATASET_INFO


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
        description='Download DiaMOS Plant dataset'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory to download data to (default: data/raw)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Print dataset info
    logger.info("=" * 80)
    logger.info(f"{'DiaMOS Plant Dataset Downloader':^80}")
    logger.info("=" * 80)
    logger.info(f"Dataset: {DIAMOS_DATASET_INFO['name']}")
    logger.info(f"Description: {DIAMOS_DATASET_INFO['description']}")
    logger.info(f"Classes: {', '.join(DIAMOS_DATASET_INFO['classes'])}")
    logger.info(f"Expected samples: {DIAMOS_DATASET_INFO['num_samples']}")
    logger.info("=" * 80)

    # Create downloader
    data_dir = Path(args.data_dir)
    downloader = DiaMOSDownloader(data_dir=data_dir)

    # Check if already downloaded
    if not args.force:
        info = downloader.get_dataset_info()

        if info['status'] == 'downloaded':
            logger.info("\n✓ Dataset already exists!")
            logger.info(f"Location: {info['path']}")
            logger.info(f"Total images: {info['total_images']}")
            logger.info("\nImages per class:")
            for class_name, count in info['classes'].items():
                logger.info(f"  {class_name}: {count}")

            logger.info("\nUse --force flag to re-download")
            return 0

    # Download
    logger.info("\nStarting download...")
    success = downloader.download(force=args.force)

    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✓ Download successful!")
        logger.info("=" * 80)

        # Print dataset info
        info = downloader.get_dataset_info()
        logger.info(f"Location: {info['path']}")
        logger.info(f"Total images: {info['total_images']}")
        logger.info("\nImages per class:")
        for class_name, count in info['classes'].items():
            logger.info(f"  {class_name}: {count}")

        logger.info("\nNext steps:")
        logger.info("1. Generate synthetic sensor data: python scripts/generate_synthetic_data.py")
        logger.info("2. Prepare dataset splits: python scripts/prepare_dataset.py")

        return 0
    else:
        logger.error("\n" + "=" * 80)
        logger.error("✗ Download failed")
        logger.error("=" * 80)
        logger.error("Please follow the manual download instructions above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
