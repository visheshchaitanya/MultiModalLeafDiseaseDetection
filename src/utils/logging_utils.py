"""
Logging utilities for the multi-modal leaf disease detection system.
Provides console and file logging with customizable formats and levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logging(
    name: str = "multimodal_leaf",
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    file_logging: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging with console and file handlers.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (created if doesn't exist)
        console: Enable console logging
        file_logging: Enable file logging
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format string
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = ColoredFormatter(format_string, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_logging and log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_formatter = logging.Formatter(format_string, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_config(logger: logging.Logger, config: dict, title: str = "Configuration") -> None:
    """
    Log configuration dictionary in a readable format.

    Args:
        logger: Logger instance
        config: Configuration dictionary
        title: Title for the configuration section
    """
    logger.info("=" * 80)
    logger.info(f"{title:^80}")
    logger.info("=" * 80)

    def log_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")

    log_dict(config)
    logger.info("=" * 80)


def log_metrics(logger: logging.Logger, metrics: dict, epoch: Optional[int] = None, prefix: str = "") -> None:
    """
    Log metrics in a formatted way.

    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        epoch: Optional epoch number
        prefix: Optional prefix for metric names (e.g., 'train/', 'val/')
    """
    if epoch is not None:
        logger.info(f"Epoch {epoch} - {prefix}Metrics:")
    else:
        logger.info(f"{prefix}Metrics:")

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


def log_model_summary(logger: logging.Logger, model, input_size=None) -> None:
    """
    Log model summary including architecture and parameter count.

    Args:
        logger: Logger instance
        model: PyTorch model
        input_size: Optional input size for detailed summary
    """
    logger.info("=" * 80)
    logger.info(f"{'Model Architecture':^80}")
    logger.info("=" * 80)

    # Log model structure
    logger.info(str(model))
    logger.info("-" * 80)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info("=" * 80)


class TrainingLogger:
    """Helper class for logging training progress."""

    def __init__(self, logger: logging.Logger, log_frequency: int = 10):
        """
        Initialize training logger.

        Args:
            logger: Logger instance
            log_frequency: Log every N batches
        """
        self.logger = logger
        self.log_frequency = log_frequency
        self.epoch_start_time = None
        self.batch_times = []

    def on_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch_start_time = datetime.now()
        self.batch_times = []
        self.logger.info("=" * 80)
        self.logger.info(f"Epoch {epoch}/{total_epochs}")
        self.logger.info("=" * 80)

    def on_batch_end(self, batch_idx: int, total_batches: int, loss: float, metrics: Optional[dict] = None):
        """Log batch progress."""
        if (batch_idx + 1) % self.log_frequency == 0 or (batch_idx + 1) == total_batches:
            progress = (batch_idx + 1) / total_batches * 100
            log_msg = f"Batch [{batch_idx+1}/{total_batches}] ({progress:.1f}%) - Loss: {loss:.4f}"

            if metrics:
                metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                log_msg += f" | {metric_str}"

            self.logger.info(log_msg)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                     train_metrics: Optional[dict] = None, val_metrics: Optional[dict] = None):
        """Log epoch end summary."""
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()

        self.logger.info("-" * 80)
        self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        self.logger.info(f"Train Loss: {train_loss:.4f}")

        if train_metrics:
            train_metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            self.logger.info(f"Train Metrics: {train_metric_str}")

        if val_loss is not None:
            self.logger.info(f"Val Loss: {val_loss:.4f}")

        if val_metrics:
            val_metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger.info(f"Val Metrics: {val_metric_str}")

        self.logger.info("=" * 80)


# Global logger instance
_global_logger = None


def get_global_logger() -> logging.Logger:
    """Get or create the global logger instance."""
    global _global_logger

    if _global_logger is None:
        # Setup default logger
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        _global_logger = setup_logging(
            name="multimodal_leaf",
            log_level="INFO",
            log_file=log_file,
            console=True,
            file_logging=True
        )

    return _global_logger


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(
        name="test_logger",
        log_level="DEBUG",
        log_file=Path("outputs/logs/test.log")
    )

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test config logging
    test_config = {
        "model": {
            "backbone": "resnet50",
            "hidden_dim": 512
        },
        "training": {
            "batch_size": 32,
            "lr": 0.001
        }
    }
    log_config(logger, test_config, "Test Configuration")

    # Test metrics logging
    test_metrics = {
        "accuracy": 0.9234,
        "loss": 0.1567,
        "f1_score": 0.9012
    }
    log_metrics(logger, test_metrics, epoch=5, prefix="val/")
