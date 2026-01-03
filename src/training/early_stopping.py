"""
Early stopping callback to prevent overfitting.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/metrics
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}, min_delta={abs(min_delta)}")

    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            current_score: Current metric value
            epoch: Current epoch

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            # First epoch
            self.best_score = current_score
            self.best_epoch = epoch
            if self.verbose:
                logger.info(f"Epoch {epoch}: Initial best score: {current_score:.4f}")
            return False

        # Check if current score is better
        if self.monitor_op(current_score, self.best_score + self.min_delta):
            # Improvement
            if self.verbose:
                logger.info(f"Epoch {epoch}: Metric improved from {self.best_score:.4f} to {current_score:.4f}")
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logger.info(f"Epoch {epoch}: No improvement for {self.counter}/{self.patience} epochs "
                          f"(best: {self.best_score:.4f} at epoch {self.best_epoch})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True

            return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        logger.info("EarlyStopping reset")

    def state_dict(self) -> dict:
        """
        Get state dictionary.

        Returns:
            State dictionary
        """
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load state dictionary.

        Args:
            state_dict: State dictionary
        """
        self.counter = state_dict.get('counter', 0)
        self.best_score = state_dict.get('best_score')
        self.early_stop = state_dict.get('early_stop', False)
        self.best_epoch = state_dict.get('best_epoch', 0)


if __name__ == "__main__":
    # Test early stopping
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing EarlyStopping...")
    print("=" * 80)

    # Test for minimization (loss)
    print("\nTest 1: Early stopping for loss (mode='min')")
    print("-" * 80)

    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')

    # Simulate training with improving then worsening loss
    losses = [1.0, 0.9, 0.85, 0.84, 0.86, 0.87, 0.88, 0.89]

    for epoch, loss in enumerate(losses):
        should_stop = early_stopping(loss, epoch)
        if should_stop:
            print(f"\nTraining stopped at epoch {epoch}")
            break

    # Test for maximization (accuracy)
    print("\n\nTest 2: Early stopping for accuracy (mode='max')")
    print("-" * 80)

    early_stopping_acc = EarlyStopping(patience=5, min_delta=0.001, mode='max')

    # Simulate training with improving then plateauing accuracy
    accuracies = [0.6, 0.7, 0.75, 0.78, 0.79, 0.79, 0.79, 0.79, 0.79, 0.79]

    for epoch, acc in enumerate(accuracies):
        should_stop = early_stopping_acc(acc, epoch)
        if should_stop:
            print(f"\nTraining stopped at epoch {epoch}")
            break

    # Test state dict save/load
    print("\n\nTest 3: State dict save/load")
    print("-" * 80)

    early_stopping_test = EarlyStopping(patience=3, mode='min')
    early_stopping_test(0.5, 0)
    early_stopping_test(0.4, 1)

    state = early_stopping_test.state_dict()
    print(f"Saved state: {state}")

    # Create new instance and load state
    early_stopping_loaded = EarlyStopping(patience=3, mode='min')
    early_stopping_loaded.load_state_dict(state)

    print(f"Loaded state: {early_stopping_loaded.state_dict()}")
    print(f"States match: {state == early_stopping_loaded.state_dict()}")

    print("\n" + "=" * 80)
    print("EarlyStopping tests completed!")
