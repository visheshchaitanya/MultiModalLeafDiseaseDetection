"""
Main trainer for multi-modal leaf disease detection model.
Orchestrates training, validation, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import logging

from .losses import MultiTaskLoss
from .optimizer import create_optimizer, create_scheduler, WarmupScheduler
from .early_stopping import EarlyStopping
from .checkpointing import CheckpointManager
from ..utils.logging_utils import TrainingLogger
from ..utils.device import get_device, move_to_device

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer class for multi-modal model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: Dict,
        checkpoint_dir: Path,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            config: Configuration dictionary
            checkpoint_dir: Directory for saving checkpoints
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or get_device(use_cuda=config.get('device', {}).get('use_cuda', True))

        # Move model to device
        self.model = self.model.to(self.device)

        # Training config
        training_config = config.get('training', {})
        self.num_epochs = training_config.get('num_epochs', 100)
        self.gradient_clip = training_config.get('gradient_clip', 1.0)
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        self.mixed_precision = training_config.get('mixed_precision', True)

        # Create optimizer
        optimizer_config = config.get('optimizer', {})
        self.optimizer = create_optimizer(self.model, optimizer_config)

        # Create scheduler
        scheduler_config = config.get('scheduler', {})
        num_training_steps = len(train_loader) * self.num_epochs
        base_scheduler = create_scheduler(self.optimizer, scheduler_config, num_training_steps)

        # Add warmup if specified
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            self.scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                base_scheduler=base_scheduler
            )
        else:
            self.scheduler = base_scheduler

        # Early stopping
        early_stopping_config = config.get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 15),
                min_delta=early_stopping_config.get('min_delta', 0.001),
                mode=early_stopping_config.get('mode', 'min')
            )
        else:
            self.early_stopping = None

        # Checkpoint manager
        checkpoint_config = config.get('checkpointing', {})
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            save_best=checkpoint_config.get('save_best', True),
            save_last=checkpoint_config.get('save_last', True),
            save_top_k=checkpoint_config.get('save_top_k', 3),
            monitor=checkpoint_config.get('monitor', 'val_loss'),
            mode=checkpoint_config.get('mode', 'min')
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and torch.cuda.is_available() else None

        # TensorBoard writer
        log_dir = Path(config.get('paths', {}).get('tensorboard', 'outputs/tensorboard'))
        self.writer = SummaryWriter(log_dir=log_dir)

        # Training logger
        self.training_logger = TrainingLogger(logger, log_frequency=10)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        logger.info("Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Num epochs: {self.num_epochs}")
        logger.info(f"  Mixed precision: {self.mixed_precision}")
        logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def train(self):
        """Run full training loop."""
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            self.training_logger.on_epoch_start(epoch + 1, self.num_epochs)
            train_metrics = self.train_epoch()

            # Validate epoch
            val_metrics = self.validate_epoch()

            # Log epoch results
            self.training_logger.on_epoch_end(
                epoch=epoch + 1,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_metrics=self._filter_metrics(train_metrics),
                val_metrics=self._filter_metrics(val_metrics)
            )

            # TensorBoard logging
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)

            # Learning rate scheduling
            self._step_scheduler(val_metrics['loss'])

            # Save checkpoint
            all_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
            all_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                metrics=all_metrics,
                global_step=self.global_step
            )

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['loss'], epoch + 1):
                    logger.info("Early stopping triggered - training complete")
                    break

        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info("=" * 80)

        # Close writer
        self.writer.close()

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_class_loss = 0.0
        total_text_loss = 0.0
        num_correct = 0
        num_samples = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = move_to_device(batch, self.device)

            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        images=batch['images'],
                        sensors=batch['sensors'],
                        text_ids=batch['text_ids']
                    )

                    # Compute loss
                    losses = self.criterion(
                        class_logits=outputs['class_logits'],
                        class_targets=batch['class_labels'],
                        text_logits=outputs.get('text_logits'),
                        text_targets=batch['text_ids']
                    )

                    loss = losses['total_loss'] / self.gradient_accumulation_steps

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Regular forward pass (no mixed precision)
                outputs = self.model(
                    images=batch['images'],
                    sensors=batch['sensors'],
                    text_ids=batch['text_ids']
                )

                # Compute loss
                losses = self.criterion(
                    class_logits=outputs['class_logits'],
                    class_targets=batch['class_labels'],
                    text_logits=outputs.get('text_logits'),
                    text_targets=batch['text_ids']
                )

                loss = losses['total_loss'] / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Accumulate metrics
            total_loss += losses['total_loss'].item()
            total_class_loss += losses.get('class_loss', torch.tensor(0.0)).item()
            total_text_loss += losses.get('text_loss', torch.tensor(0.0)).item()

            # Accuracy
            predictions = torch.argmax(outputs['class_logits'], dim=1)
            num_correct += (predictions == batch['class_labels']).sum().item()
            num_samples += batch['class_labels'].size(0)

            # Update progress bar
            pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})

            self.global_step += 1

        # Compute average metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_class_loss = total_class_loss / len(self.train_loader)
        avg_text_loss = total_text_loss / len(self.train_loader)
        accuracy = num_correct / num_samples

        return {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'text_loss': avg_text_loss,
            'accuracy': accuracy
        }

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_class_loss = 0.0
        total_text_loss = 0.0
        num_correct = 0
        num_samples = 0

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
            # Move batch to device
            batch = move_to_device(batch, self.device)

            # Forward pass
            outputs = self.model(
                images=batch['images'],
                sensors=batch['sensors'],
                text_ids=batch['text_ids']
            )

            # Compute loss
            losses = self.criterion(
                class_logits=outputs['class_logits'],
                class_targets=batch['class_labels'],
                text_logits=outputs.get('text_logits'),
                text_targets=batch['text_ids']
            )

            # Accumulate metrics
            total_loss += losses['total_loss'].item()
            total_class_loss += losses.get('class_loss', torch.tensor(0.0)).item()
            total_text_loss += losses.get('text_loss', torch.tensor(0.0)).item()

            # Accuracy
            predictions = torch.argmax(outputs['class_logits'], dim=1)
            num_correct += (predictions == batch['class_labels']).sum().item()
            num_samples += batch['class_labels'].size(0)

        # Compute average metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_class_loss = total_class_loss / len(self.val_loader)
        avg_text_loss = total_text_loss / len(self.val_loader)
        accuracy = num_correct / num_samples

        return {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'text_loss': avg_text_loss,
            'accuracy': accuracy
        }

    def _step_scheduler(self, val_loss: float):
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, WarmupScheduler):
                self.scheduler.step(self.current_epoch, val_loss)
            elif hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

    def _log_to_tensorboard(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to TensorBoard."""
        # Training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)

        # Validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)

        # Learning rate
        if self.scheduler is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('learning_rate', lr, epoch)

    def _filter_metrics(self, metrics: Dict) -> Dict:
        """Filter out loss from metrics for cleaner logging."""
        return {k: v for k, v in metrics.items() if k != 'loss'}


if __name__ == "__main__":
    # Test trainer
    print("Trainer module loaded successfully")
    print("To test, please run with actual model and data loaders")
