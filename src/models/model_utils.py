"""
Utilities for model initialization, loading, saving, and analysis.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def initialize_weights(module: nn.Module, method: str = 'xavier_uniform'):
    """
    Initialize model weights.

    Args:
        module: PyTorch module
        method: Initialization method ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif method == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        else:
            raise ValueError(f"Unsupported initialization method: {method}")

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)

    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_summary(model: nn.Module) -> Dict[str, any]:
    """
    Get model summary statistics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with summary statistics
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    }

    return summary


def print_model_summary(model: nn.Module, detailed: bool = False):
    """
    Print model summary.

    Args:
        model: PyTorch model
        detailed: Whether to print detailed layer-by-layer information
    """
    summary = get_model_summary(model)

    logger.info("=" * 80)
    logger.info(f"{'Model Summary':^80}")
    logger.info("=" * 80)
    logger.info(f"Total parameters: {summary['total_parameters']:,}")
    logger.info(f"Trainable parameters: {summary['trainable_parameters']:,}")
    logger.info(f"Frozen parameters: {summary['frozen_parameters']:,}")
    logger.info(f"Model size: {summary['model_size_mb']:.2f} MB")
    logger.info("=" * 80)

    if detailed:
        logger.info("\nLayer-by-layer breakdown:")
        logger.info("-" * 80)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    logger.info(f"{name:50s} {num_params:>15,}")


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    path: Union[str, Path],
    **kwargs
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        epoch: Current epoch
        metrics: Dictionary of metrics
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        **kwargs
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to load to

    Returns:
        Checkpoint dictionary
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)

    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model weights loaded from {path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Optimizer state loaded from {path}")

    logger.info(f"Checkpoint loaded from {path} (epoch {checkpoint.get('epoch', 'unknown')})")

    return checkpoint


def freeze_module(module: nn.Module, freeze: bool = True):
    """
    Freeze or unfreeze a module.

    Args:
        module: Module to freeze/unfreeze
        freeze: Whether to freeze (True) or unfreeze (False)
    """
    for param in module.parameters():
        param.requires_grad = not freeze

    status = "frozen" if freeze else "unfrozen"
    logger.info(f"Module {module.__class__.__name__} {status}")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Set learning rate for optimizer.

    Args:
        optimizer: PyTorch optimizer
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.info(f"Learning rate set to {lr}")


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by norm.

    Args:
        model: PyTorch model
        max_norm: Maximum norm
        norm_type: Type of norm (2.0 for L2)

    Returns:
        Total norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    )
    return total_norm.item()


if __name__ == "__main__":
    # Test model utilities
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Model Utilities...")
    print("=" * 80)

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 20)
            self.fc3 = nn.Linear(20, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleModel()

    # Test parameter counting
    print("\nTest 1: Parameter counting")
    print("-" * 80)
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test model summary
    print("\n\nTest 2: Model summary")
    print("-" * 80)
    print_model_summary(model, detailed=True)

    # Test weight initialization
    print("\n\nTest 3: Weight initialization")
    print("-" * 80)
    model.apply(lambda m: initialize_weights(m, method='xavier_uniform'))
    print("Weights initialized with xavier_uniform")

    # Test freezing
    print("\n\nTest 4: Freezing/unfreezing")
    print("-" * 80)
    freeze_module(model.fc1, freeze=True)
    trainable_after_freeze = count_parameters(model, trainable_only=True)
    print(f"Trainable parameters after freezing fc1: {trainable_after_freeze:,}")

    freeze_module(model.fc1, freeze=False)
    trainable_after_unfreeze = count_parameters(model, trainable_only=True)
    print(f"Trainable parameters after unfreezing fc1: {trainable_after_unfreeze:,}")

    # Test checkpoint save/load
    print("\n\nTest 5: Checkpoint save/load")
    print("-" * 80)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    save_path = Path("outputs/test_checkpoint.pt")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=10,
        metrics={'loss': 0.5, 'accuracy': 0.9},
        path=save_path
    )

    # Load checkpoint
    checkpoint = load_checkpoint(save_path, model=model, optimizer=optimizer)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Metrics: {checkpoint['metrics']}")

    print("\n" + "=" * 80)
    print("Model utilities tests completed!")
