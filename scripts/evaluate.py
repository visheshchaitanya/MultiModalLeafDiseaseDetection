"""
Evaluation script for multi-modal leaf disease detection model.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

from src.utils.config_loader import ConfigLoader
from src.utils.logging_utils import setup_logging
from src.utils.device import get_device, print_device_info
from src.data.dataloader import DataLoaderFactory
from src.models.multimodal_model import MultiModalLeafDiseaseModel
from src.evaluation.evaluator import Evaluator
from src.templates.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate multi-modal leaf disease detection model'
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
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Output directory for evaluation results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run evaluation on'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )

    parser.add_argument(
        '--no-text-metrics',
        action='store_true',
        help='Skip text generation metrics (faster)'
    )

    parser.add_argument(
        '--no-save-predictions',
        action='store_true',
        help='Skip saving predictions to file'
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting confusion matrix'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 80)
    logger.info("Multi-Modal Leaf Disease Detection - Evaluation")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config_loader = ConfigLoader(args.config)
    config = config_loader.load()

    # Override config with command line arguments
    if args.device is not None:
        config['device']['use_cuda'] = (args.device == 'cuda')
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size

    # Setup device
    use_cuda = config.get('device', {}).get('use_cuda', True)
    device = get_device(use_cuda=use_cuda)
    print_device_info(device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    logger.info("Loading vocabulary...")
    vocab_path = Path(config['paths']['processed_data']) / 'vocabulary.json'
    vocabulary = Vocabulary.load_from_json(vocab_path)
    logger.info(f"Vocabulary loaded: {len(vocabulary)} tokens")

    # Load class names
    class_names = config['data'].get('class_names', [
        'Healthy', 'Alternaria', 'Stemphylium', 'Marssonina'
    ])
    logger.info(f"Classes: {class_names}")

    # Create data loader
    logger.info(f"Creating data loader for {args.split} split...")
    loader_factory = DataLoaderFactory(
        data_dir=config['paths']['processed_data'],
        config=config,
        vocabulary=vocabulary
    )

    if args.split == 'train':
        train_loader, _, _ = loader_factory.create_all_dataloaders()
        eval_loader = train_loader
    elif args.split == 'val':
        _, val_loader, _ = loader_factory.create_all_dataloaders()
        eval_loader = val_loader
    else:  # test
        _, _, test_loader = loader_factory.create_all_dataloaders()
        eval_loader = test_loader

    logger.info(f"  Evaluation samples: {len(eval_loader.dataset)}")

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

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")

        # Print checkpoint metrics if available
        if 'metrics' in checkpoint:
            logger.info("  Checkpoint metrics:")
            for key, value in checkpoint['metrics'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {key}: {value:.4f}")
    else:
        # Try loading directly as state dict
        model.load_state_dict(checkpoint)
        logger.info("  Loaded model state dict")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = Evaluator(
        model=model,
        test_loader=eval_loader,
        vocabulary=vocabulary,
        class_names=class_names,
        device=device,
        output_dir=output_dir
    )

    # Run evaluation
    logger.info("\n" + "=" * 80)
    logger.info("Starting evaluation...")
    logger.info("=" * 80 + "\n")

    try:
        results = evaluator.evaluate(
            compute_text_metrics=not args.no_text_metrics,
            save_predictions=not args.no_save_predictions,
            plot_cm=not args.no_plot
        )

        logger.info("\n" + "=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info("=" * 80)

        logger.info(f"\nResults saved to {output_dir}")

        # Print summary
        logger.info("\nSummary:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  F1-Score (macro): {results['f1_macro']:.4f}")

        if 'bleu_4' in results:
            logger.info(f"  BLEU-4: {results['bleu_4']:.4f}")
            logger.info(f"  ROUGE-L: {results['rougeL']:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
