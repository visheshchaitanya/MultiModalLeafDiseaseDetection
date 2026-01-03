"""
Helper script to extract the manually downloaded Pear.zip dataset.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.download import DiaMOSDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Extract and organize the dataset."""
    archive_path = Path('data/raw/Pear.zip')

    # Check if file exists
    if not archive_path.exists():
        logger.error(f"Archive file not found: {archive_path}")
        logger.info("Please download Pear.zip from https://zenodo.org/records/5557313")
        logger.info(f"And place it at: {archive_path.absolute()}")
        return 1

    logger.info(f"Found archive: {archive_path}")
    logger.info(f"File size: {archive_path.stat().st_size / (1024**3):.2f} GB")

    # Create downloader
    downloader = DiaMOSDownloader(data_dir=Path('data/raw'))

    # Extract
    logger.info("Starting extraction (this may take several minutes for 13GB)...")
    extract_to = Path('data/raw')

    if not downloader.extract_archive(archive_path, extract_to):
        logger.error("Extraction failed!")
        return 1

    logger.info("Extraction successful!")

    # Check what was extracted
    logger.info("Checking extracted contents...")
    extracted_items = list(extract_to.iterdir())
    logger.info(f"Found {len(extracted_items)} items in {extract_to}:")
    for item in extracted_items:
        if item.is_dir():
            logger.info(f"  [DIR]  {item.name}")
        else:
            logger.info(f"  [FILE] {item.name}")

    # Try to organize dataset
    # Look for the extracted directory
    possible_dirs = [
        extract_to / 'DiaMOS_Plant',
        extract_to / 'Pear',
        extract_to / 'pear',
        extract_to / 'diamos_plant',
    ]

    dataset_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists() and dir_path.is_dir():
            dataset_dir = dir_path
            logger.info(f"Found dataset directory: {dataset_dir}")
            break

    if dataset_dir:
        logger.info("Organizing dataset structure...")
        downloader.organize_dataset(dataset_dir)

        # Get dataset info
        info = downloader.get_dataset_info()
        if info['status'] == 'downloaded':
            logger.info("\n" + "=" * 80)
            logger.info("✓ Dataset extracted and organized successfully!")
            logger.info("=" * 80)
            logger.info(f"Location: {info['path']}")
            logger.info(f"Total images: {info['total_images']}")
            logger.info("\nImages per class:")
            for class_name, count in info['classes'].items():
                logger.info(f"  {class_name}: {count}")
            logger.info("\nNext steps:")
            logger.info("1. Generate synthetic sensor data: .venv\\Scripts\\python.exe scripts\\generate_synthetic_data.py")
            logger.info("2. Prepare dataset splits: .venv\\Scripts\\python.exe scripts\\prepare_dataset.py")
            logger.info("=" * 80)
        else:
            logger.warning("Dataset structure may need manual organization")
            logger.info(f"Please check: {extract_to}")
    else:
        logger.warning("Could not find expected dataset directory")
        logger.info(f"Please check extracted contents in: {extract_to}")
        logger.info("Expected structure: DiaMOS_Plant/Healthy/, DiaMOS_Plant/Alternaria/, etc.")

    # Optionally clean up archive
    logger.info(f"\nArchive file still at: {archive_path}")
    logger.info("You can delete it manually to save space if extraction was successful")

    return 0

if __name__ == "__main__":
    sys.exit(main())
