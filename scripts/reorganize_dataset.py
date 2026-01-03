"""
Reorganize the DiaMOS Pear dataset to match expected structure.

Maps the actual folder names to scientific disease names:
- leaves/slug -> Alternaria (black spot/slug leaf)
- leaves/spot -> Stemphylium (spot leaf)
- leaves/curl -> Marssonina (leaf curl)
- leaves/healthy -> Healthy
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Reorganize dataset structure."""
    # Source and destination paths
    source_base = Path('data/raw/Pear/leaves')
    dest_base = Path('data/raw/DiaMOS_Plant')

    # Disease mapping
    disease_mapping = {
        'slug': 'Alternaria',      # Black spot/slug leaf
        'spot': 'Stemphylium',     # Spot leaf
        'curl': 'Marssonina',      # Leaf curl
        'healthy': 'Healthy'       # Healthy leaves
    }

    logger.info("=" * 80)
    logger.info("Reorganizing DiaMOS Pear Dataset")
    logger.info("=" * 80)

    # Check if source exists
    if not source_base.exists():
        logger.error(f"Source directory not found: {source_base}")
        logger.error("Please extract Pear.zip first!")
        return 1

    # Check if already reorganized
    if dest_base.exists():
        logger.warning(f"Destination {dest_base} already exists!")
        response = input("Do you want to recreate it? (yes/no): ").strip().lower()
        if response != 'yes':
            logger.info("Aborted. Using existing structure.")
            return 0
        logger.info("Removing existing directory...")
        shutil.rmtree(dest_base)

    # Create destination directory
    dest_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created: {dest_base}")

    # Copy and reorganize
    total_images = 0
    for old_name, new_name in disease_mapping.items():
        source_dir = source_base / old_name
        dest_dir = dest_base / new_name

        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            continue

        logger.info(f"\nProcessing: {old_name} -> {new_name}")

        # Count images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(source_dir.glob(f'*{ext}')))

        logger.info(f"  Found {len(image_files)} images in {source_dir}")

        if len(image_files) == 0:
            logger.warning(f"  No images found in {source_dir}, skipping...")
            continue

        # Create destination and copy files
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_file in image_files:
            dest_file = dest_dir / img_file.name
            shutil.copy2(img_file, dest_file)

        logger.info(f"  Copied {len(image_files)} images to {dest_dir}")
        total_images += len(image_files)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Reorganization Complete!")
    logger.info("=" * 80)
    logger.info(f"Total images: {total_images}")
    logger.info(f"Location: {dest_base.absolute()}")
    logger.info("\nClass distribution:")

    for old_name, new_name in disease_mapping.items():
        dest_dir = dest_base / new_name
        if dest_dir.exists():
            count = len(list(dest_dir.glob('*.jpg'))) + len(list(dest_dir.glob('*.JPG')))
            logger.info(f"  {new_name}: {count} images")

    logger.info("\nNext steps:")
    logger.info("1. Generate synthetic sensor data:")
    logger.info("   .venv\\Scripts\\python.exe scripts\\generate_synthetic_data.py")
    logger.info("2. Prepare dataset splits:")
    logger.info("   .venv\\Scripts\\python.exe scripts\\prepare_dataset.py")
    logger.info("=" * 80)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
