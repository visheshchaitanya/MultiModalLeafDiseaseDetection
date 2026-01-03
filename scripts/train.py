"""
Training script for multi-modal leaf disease detection model.
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
from src.utils.seed import set_seed
from src.utils.device import get_device, print_device_info
from src.data.dataloader import DataLoaderFactory
from src.models.multimodal_model import MultiModalLeafDiseaseModel
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer
from src.templates.vocabulary import Vocabulary, load_vocabulary

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train multi-modal leaf disease detection model'
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
        default=None,
        help='Path to checkpoint to resume training from'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to train on (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
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
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{Path(__file__).stem}.log"
    setup_logging(
        name="multimodal_leaf",
        log_level=args.log_level,
        log_file=log_file,
        console=True,
        file_logging=True
    )

    logger.info("=" * 80)
    logger.info("Multi-Modal Leaf Disease Detection - Training")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config_loader = ConfigLoader(args.config)
    config = config_loader.load()

    # Override config with command line arguments
    if args.seed is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['seed'] = args.seed
    if args.device is not None:
        if 'device' not in config:
            config['device'] = {}
        config['device']['use_cuda'] = (args.device == 'cuda')
    if args.batch_size is not None:
        if 'training' not in config:
            config['training'] = {}
        if 'data' not in config['training']:
            config['training']['data'] = {}
        config['training']['data']['batch_size'] = args.batch_size
    if args.epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['num_epochs'] = args.epochs
    if args.lr is not None:
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['learning_rate'] = args.lr
    if args.output_dir is not None:
        if 'paths' not in config:
            config['paths'] = {}
        config['paths']['outputs'] = args.output_dir

    # Set random seed
    seed = config.get('training', {}).get('seed', config.get('dataset', {}).get('random_seed', 42))
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # Setup device
    use_cuda = config.get('device', {}).get('use_cuda', True)
    device = get_device(use_cuda=use_cuda)
    print_device_info(device)

    # Create output directory
    output_dir = Path(config['paths'].get('outputs', 'outputs'))
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    logger.info("Loading vocabulary...")
    vocab_path = Path(config['paths']['processed_data']) / 'vocabulary.pkl'
    vocabulary = load_vocabulary(vocab_path)
    logger.info(f"Vocabulary loaded: {len(vocabulary)} tokens")

    # Load class names
    class_names = config.get('dataset', {}).get('class_names', [
        'Healthy', 'Alternaria', 'Stemphylium', 'Marssonina'
    ])
    logger.info(f"Classes: {class_names}")

    # Create data loaders
    logger.info("Creating data loaders...")
    loader_factory = DataLoaderFactory(
        data_dir=config['paths']['processed_data'],
        config=config,
        vocabulary=vocabulary
    )

    train_loader, val_loader, test_loader = loader_factory.create_all_dataloaders()

    logger.info(f"  Training samples: {len(train_loader.dataset)}")
    logger.info(f"  Validation samples: {len(val_loader.dataset)}")
    logger.info(f"  Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = MultiModalLeafDiseaseModel(
        vocab_size=len(vocabulary),
        num_classes=len(class_names),
        image_encoder_config=config.get('model', {}).get('image_encoder', {}),
        tabular_encoder_config=config.get('model', {}).get('tabular_encoder', {}),
        fusion_config=config.get('model', {}).get('fusion', {}),
        decoder_config=config.get('model', {}).get('decoder', {}),
        classifier_config=config.get('model', {}).get('classifier', {})
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Create loss function
    logger.info("Creating loss function...")
    loss_config = config.get('training', {}).get('loss', {})

    # Get class weights if available
    class_weights = None
    if loss_config.get('use_class_weights', False):
        # You can compute class weights from the dataset here
        # For now, use None (uniform weights)
        pass

    criterion = MultiTaskLoss(
        alpha=loss_config.get('alpha', 0.7),
        beta=loss_config.get('beta', 0.3),
        class_weights=class_weights,
        ignore_index=vocabulary.get_token_index(vocabulary.PAD_TOKEN),
        use_uncertainty_weighting=loss_config.get('use_uncertainty_weighting', False)
    )

    logger.info(f"  Multi-task loss: alpha={loss_config.get('alpha', 0.7)}, beta={loss_config.get('beta', 0.3)}")

    # Load checkpoint if resuming
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"  Resumed from epoch {checkpoint['epoch']}")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving current state...")

        # Save interrupted checkpoint
        interrupted_path = checkpoint_dir / 'interrupted.pt'
        torch.save({
            'epoch': trainer.current_epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'global_step': trainer.global_step
        }, interrupted_path)

        logger.info(f"Checkpoint saved to {interrupted_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("\n" + "=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)

    # Print best checkpoint location
    best_checkpoint = trainer.checkpoint_manager.get_best_checkpoint_path()
    if best_checkpoint and best_checkpoint.exists():
        logger.info(f"\nBest checkpoint: {best_checkpoint}")

    # Run evaluation on test set
    logger.info("\n" + "=" * 80)
    logger.info("Running evaluation on test set...")
    logger.info("=" * 80)

    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        model=trainer.model,
        test_loader=test_loader,
        vocabulary=vocabulary,
        class_names=class_names,
        device=device,
        output_dir=output_dir / 'evaluation'
    )

    results = evaluator.evaluate(
        compute_text_metrics=True,
        save_predictions=True,
        plot_cm=True
    )

    logger.info("\nTraining and evaluation complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
