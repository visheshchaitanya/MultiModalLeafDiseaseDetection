"""
Clean CSV files to remove references to invalid/deleted images.
Removes entries for macOS metadata files (._*) that don't exist.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def clean_csv(csv_path):
    """
    Remove entries with invalid image paths.

    Args:
        csv_path: Path to CSV file
    """
    csv_path = Path(csv_path)
    logger.info(f"Cleaning {csv_path.name}...")

    # Read CSV
    df = pd.read_csv(csv_path)
    original_count = len(df)
    logger.info(f"  Original entries: {original_count}")

    # Check which images actually exist
    valid_indices = []
    for idx, row in df.iterrows():
        image_path = Path(row['image_path'])

        # Skip macOS metadata files
        if image_path.name.startswith('._'):
            logger.debug(f"  Skipping metadata file: {image_path.name}")
            continue

        # Check if file exists
        if image_path.exists():
            valid_indices.append(idx)
        else:
            logger.debug(f"  Missing file: {image_path}")

    # Filter to valid entries
    df_clean = df.loc[valid_indices].reset_index(drop=True)
    removed_count = original_count - len(df_clean)

    logger.info(f"  Removed entries: {removed_count}")
    logger.info(f"  Valid entries: {len(df_clean)}")

    # Save cleaned CSV
    df_clean.to_csv(csv_path, index=False)
    logger.info(f"  ✓ Saved cleaned CSV")

    return original_count, len(df_clean)


def main():
    logger.info("=" * 80)
    logger.info("Cleaning CSV Files")
    logger.info("=" * 80)

    csv_files = [
        'data/processed/train.csv',
        'data/processed/val.csv',
        'data/processed/test.csv'
    ]

    total_removed = 0
    total_remaining = 0

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if csv_path.exists():
            original, remaining = clean_csv(csv_path)
            total_removed += (original - remaining)
            total_remaining += remaining
        else:
            logger.warning(f"File not found: {csv_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Total removed:   {total_removed}")
    logger.info(f"Total remaining: {total_remaining}")
    logger.info("=" * 80)
    logger.info("✓ CSV files cleaned successfully!")


if __name__ == "__main__":
    main()
