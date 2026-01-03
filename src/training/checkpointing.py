"""
Checkpoint management for saving and loading model states.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, List
import logging
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints - saving, loading, and keeping best K checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        save_best: bool = True,
        save_last: bool = True,
        save_top_k: int = 3,
        monitor: str = 'val_loss',
        mode: str = 'min',
        filename_template: str = 'checkpoint_epoch_{epoch:03d}_{monitor:.4f}.pt'
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best checkpoint
            save_last: Whether to save latest checkpoint
            save_top_k: Number of best checkpoints to keep (0 = keep all)
            monitor: Metric to monitor for best checkpoint
            mode: 'min' or 'max' for the monitored metric
            filename_template: Template for checkpoint filenames
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_best = save_best
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.filename_template = filename_template

        self.best_score = None
        self.saved_checkpoints: List[Dict] = []

        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")
        logger.info(f"  save_best={save_best}, save_last={save_last}, save_top_k={save_top_k}")
        logger.info(f"  monitor='{monitor}', mode='{mode}'")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ):
        """
        Save checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            **kwargs: Additional items to save
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            **kwargs
        }

        monitor_value = metrics.get(self.monitor)

        if monitor_value is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics. Available: {list(metrics.keys())}")
            monitor_value = 0.0

        # Save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / 'last.pt'
            torch.save(checkpoint, last_path)
            logger.debug(f"Saved last checkpoint: {last_path}")

        # Check if this is the best checkpoint
        is_best = self._is_best_checkpoint(monitor_value)

        # Save best checkpoint
        if self.save_best and is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved BEST checkpoint: {best_path} ({self.monitor}={monitor_value:.4f})")

        # Save epoch checkpoint and manage top-k
        if self.save_top_k != 0:  # 0 means don't save epoch checkpoints
            filename = self.filename_template.format(epoch=epoch, monitor=monitor_value)
            checkpoint_path = self.checkpoint_dir / filename
            torch.save(checkpoint, checkpoint_path)

            # Track saved checkpoints
            self.saved_checkpoints.append({
                'path': checkpoint_path,
                'epoch': epoch,
                'score': monitor_value
            })

            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Remove old checkpoints if exceeding top-k
            if self.save_top_k > 0:
                self._cleanup_old_checkpoints()

    def _is_best_checkpoint(self, current_score: float) -> bool:
        """
        Check if current checkpoint is the best.

        Args:
            current_score: Current metric score

        Returns:
            True if this is the best checkpoint
        """
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == 'min':
            is_better = current_score < self.best_score
        else:  # mode == 'max'
            is_better = current_score > self.best_score

        if is_better:
            self.best_score = current_score
            return True

        return False

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only top-k."""
        if len(self.saved_checkpoints) <= self.save_top_k:
            return

        # Sort checkpoints by score
        if self.mode == 'min':
            self.saved_checkpoints.sort(key=lambda x: x['score'])
        else:  # mode == 'max'
            self.saved_checkpoints.sort(key=lambda x: x['score'], reverse=True)

        # Remove checkpoints beyond top-k
        to_remove = self.saved_checkpoints[self.save_top_k:]

        for checkpoint_info in to_remove:
            checkpoint_path = checkpoint_info['path']
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint_path}")

        # Keep only top-k in the list
        self.saved_checkpoints = self.saved_checkpoints[:self.save_top_k]

    def load_checkpoint(
        self,
        checkpoint_name: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Load checkpoint.

        Args:
            checkpoint_name: Name of checkpoint file ('best.pt', 'last.pt', or custom)
            model: Model to load weights into (optional)
            optimizer: Optimizer to load state into (optional)
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if model is not None and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model weights loaded from {checkpoint_path}")

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Optimizer state loaded from {checkpoint_path}")

        logger.info(f"Checkpoint loaded: epoch {checkpoint.get('epoch', 'unknown')}")

        return checkpoint

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / 'best.pt'
        return best_path if best_path.exists() else None

    def get_last_checkpoint_path(self) -> Optional[Path]:
        """Get path to last checkpoint."""
        last_path = self.checkpoint_dir / 'last.pt'
        return last_path if last_path.exists() else None

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint files."""
        return sorted(self.checkpoint_dir.glob('*.pt'))


if __name__ == "__main__":
    # Test checkpoint manager
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing CheckpointManager...")
    print("=" * 80)

    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())

    # Create checkpoint manager
    checkpoint_dir = Path("outputs/test_checkpoints")
    manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        save_best=True,
        save_last=True,
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    # Simulate training and saving checkpoints
    print("\nTest 1: Saving checkpoints")
    print("-" * 80)

    val_losses = [0.5, 0.4, 0.35, 0.36, 0.33, 0.34, 0.32]

    for epoch, val_loss in enumerate(val_losses):
        metrics = {
            'train_loss': 0.6 - epoch * 0.05,
            'val_loss': val_loss,
            'accuracy': 0.7 + epoch * 0.03
        }

        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics
        )

    # List checkpoints
    print("\n\nTest 2: List checkpoints")
    print("-" * 80)

    checkpoints = manager.list_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  {ckpt.name}")

    # Load best checkpoint
    print("\n\nTest 3: Load best checkpoint")
    print("-" * 80)

    best_path = manager.get_best_checkpoint_path()
    if best_path:
        checkpoint = manager.load_checkpoint('best.pt', model=model, optimizer=optimizer)
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        print(f"Metrics: {checkpoint['metrics']}")

    # Clean up test directory
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"\nCleaned up test directory: {checkpoint_dir}")

    print("\n" + "=" * 80)
    print("CheckpointManager tests completed!")
